import torch,math
import torch.nn as nn
import torch.nn.functional as F


class Flatten(nn.Module):
    "Flatten `x` to a single dimension, often used at the end of a model. `full` for rank-1 tensor"
    def __init__(self, full:bool=False): 
        super().__init__()
        self.full = full
    def forward(self, x): return x.view(-1) if self.full else x.view(x.size(0), -1)

def bn_drop_lin(n_in:int, n_out:int, bn:bool=True, p:float=0., actn=None):
    "Sequence of batchnorm (if `bn`), dropout (with `p`) and linear (`n_in`,`n_out`) layers followed by `actn`."
    layers = [nn.BatchNorm1d(n_in)] if bn else []
    if p != 0: layers.append(nn.Dropout(p))
    layers.append(nn.Linear(n_in, n_out))
    if actn is not None: layers.append(actn)
    return layers

def cd_adaptiveconcatpool(relevant, irrelevant, module):
    mpr, mpi = module.mp.attrib(relevant,irrelevant)
    apr, api = module.ap.attrib(relevant,irrelevant)
    return torch.cat([mpr, apr], 1), torch.cat([mpi, api], 1)
def attrib_adaptiveconcatpool(self,relevant,irrelevant):
    return cd_adaptiveconcatpool(relevant,irrelevant,self)
class AdaptiveConcatPool1d(nn.Module):
    "Layer that concats `AdaptiveAvgPool1d` and `AdaptiveMaxPool1d`."
    def __init__(self, sz=None):
        "Output will be 2*sz or 2 if sz is None"
        super().__init__()
        sz = sz or 1
        self.ap,self.mp = nn.AdaptiveAvgPool1d(sz), nn.AdaptiveMaxPool1d(sz)
    def forward(self, x): return torch.cat([self.mp(x), self.ap(x)], 1)
    def attrib(self,relevant,irrelevant):
        return attrib_adaptiveconcatpool(self,relevant,irrelevant)
        


def create_head1d(nf:int, nc:int, lin_ftrs=None, ps=0.5, bn_final=False, bn=True, act="relu", concat_pooling=True):
    "Model head that takes `nf` features, runs through `lin_ftrs`, and about `nc` classes; added bn and act here"
    lin_ftrs = [2*nf if concat_pooling else nf, nc] if lin_ftrs is None else [2*nf if concat_pooling else nf] + lin_ftrs + [nc] #was [nf, 512,nc]
    ps = list([ps])
    if len(ps)==1: ps = [ps[0]/2] * (len(lin_ftrs)-2) + ps
    actns = [nn.ReLU(inplace=True) if act=="relu" else nn.ELU(inplace=True)] * (len(lin_ftrs)-2) + [None]
    layers = [AdaptiveConcatPool1d() if concat_pooling else nn.MaxPool1d(2), Flatten()]
    for ni,no,p,actn in zip(lin_ftrs[:-1],lin_ftrs[1:],ps,actns):
        layers += bn_drop_lin(ni,no,bn,p,actn)
    if bn_final: layers.append(nn.BatchNorm1d(lin_ftrs[-1], momentum=0.01))
    return nn.Sequential(*layers)

#深度可分离卷积
class SeparableConv1d(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,stride,padding,dilation=1,bias=False):
        super(SeparableConv1d,self).__init__()
        
        #逐通道卷积：groups=in_channels=out_channels
        self.conv1 = nn.Conv1d(in_channels,in_channels,kernel_size,stride,padding,dilation,groups=in_channels,bias=bias)
        #逐点卷积：普通1x1卷积
        self.pointwise = nn.Conv1d(in_channels,out_channels,kernel_size=1,stride=1,padding=0,dilation=1,groups=1,bias=bias)
    
    def forward(self,x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x

class Block(nn.Module):
    def __init__(self,in_filters,out_filters,reps,strides=1,start_with_relu=True,grow_first=True):
        #:parm reps:块重复次数
        super(Block, self).__init__()

        if out_filters != in_filters or strides!=1:
            self.skip = nn.Conv1d(in_filters,out_filters,kernel_size=1,stride=strides, bias=False)
            self.skipbn = nn.BatchNorm1d(out_filters)
        else:
            self.skip=None
        
        self.relu = nn.ReLU(inplace=True)
        rep=[]

        filters=in_filters
        if grow_first:
            rep.append(self.relu)
            #这里的卷积不改变特征图尺寸
            rep.append(SeparableConv1d(in_filters,out_filters,kernel_size=3,stride=1,padding=1,bias=False))
            rep.append(nn.BatchNorm1d(out_filters))
            filters = out_filters

        for i in range(reps-1):
            rep.append(self.relu)
            #这里的卷积不改变特征图尺寸
            rep.append(SeparableConv1d(filters,filters,kernel_size=3,stride=1,padding=1,bias=False))
            rep.append(nn.BatchNorm1d(filters))
        
        if not grow_first:
            rep.append(self.relu)
            #这里的卷积不改变特征图尺寸
            rep.append(SeparableConv1d(in_filters,out_filters,kernel_size=3,stride=1,padding=1,bias=False))
            rep.append(nn.BatchNorm1d(out_filters))

        if not start_with_relu:
            rep = rep[1:]
        else:
            rep[0] = nn.ReLU(inplace=False)
        
        #Middle flow 的stride恒为1，因此无需做池化，而其余块需要
        #其余块的stride=2，因此这里的最大池化可以将特征图尺寸减半
        if strides != 1:
            rep.append(nn.MaxPool1d(kernel_size=3,stride=strides,padding=1))
        self.rep = nn.Sequential(*rep)

    def forward(self,inp):
        x = self.rep(inp)

        if self.skip is not None:
            skip = self.skip(inp)
            skip = self.skipbn(skip)
        else:
            skip = inp

        x+=skip
        return x
class Xception(nn.Module):
    def __init__(self, num_classes=1000,input_channels=12, ps_head=0.5,lin_ftrs_head=[128]):
        super(Xception, self).__init__()
        self.num_classes = num_classes#总分类数

        ################################## 定义 Entry flow ###############################################################
        self.conv1 = nn.Conv1d(in_channels=input_channels, out_channels=32, kernel_size=3,stride=2,padding=0,bias=False)
        self.bn1 = nn.BatchNorm1d(32)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv1d(in_channels=32,out_channels=64, kernel_size=3,stride=1,padding=0,bias=False)
        self.bn2 = nn.BatchNorm1d(64)
        #do relu here

        # Block中的参数顺序：in_filters,out_filters,reps,stride,start_with_relu,grow_first
        self.block1=Block(64,128,2,2,start_with_relu=False,grow_first=True)
        self.block2=Block(128,256,2,2,start_with_relu=True,grow_first=True)
        self.block3=Block(256,728,2,2,start_with_relu=True,grow_first=True)
         
        
        ################################### 定义 Middle flow ############################################################
        self.block4=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block5=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block6=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block7=Block(728,728,3,1,start_with_relu=True,grow_first=True)

        self.block8=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block9=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block10=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block11=Block(728,728,3,1,start_with_relu=True,grow_first=True)

        
        #################################### 定义 Exit flow ###############################################################
        self.block12=Block(728,1024,2,2,start_with_relu=True,grow_first=False)

        self.conv3 = SeparableConv1d(1024,1536,3,1,1)
        self.bn3 = nn.BatchNorm1d(1536)

        #do relu here
        self.conv4 = SeparableConv1d(1536,2048,3,1,1)
        self.bn4 = nn.BatchNorm1d(2048)
        
        self.fc = nn.Linear(2048, num_classes)
        self.head = create_head1d(2048, nc=num_classes, lin_ftrs= lin_ftrs_head, ps= ps_head, bn_final=False, bn=True, act='relu', concat_pooling=True)
        #256 71 [128] 0.5 False  True 'relu' True

        ###################################################################################################################



        #------- init weights --------
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                n = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        #-----------------------------





    def forward(self, x):
        ################################## 定义 Entry flow ###############################################################
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        
        ################################### 定义 Middle flow ############################################################
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)
        x = self.block11(x)
        
        #################################### 定义 Exit flow ###############################################################
        x = self.block12(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)

        # x = F.adaptive_avg_pool1d(x, 1)
        # x = x.view(x.size(0), -1)
        x = self.head(x)

        return x


def xception(**kwargs): return Xception( **kwargs)


model = xception(num_classes =71)
model.eval()
from torchsummary import summary
summary(model, input_size=[(12 ,5000)], batch_size=1, device="cpu")
