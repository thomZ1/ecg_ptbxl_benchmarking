
from experiments.scp_experiment import SCP_Experiment
from utils import utils
# model configs
from configs.fastai_configs import *
from configs.wavelet_configs import *
import os
import pandas as pd
import numpy as np
from numpy.core.fromnumeric import argmax
from sklearn.metrics import precision_recall_curve
import pickle
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import precision_recall_fscore_support
def main():
    path_list = ['/root/zhangyi/ecg_ptbxl_benchmarking/output/exp0/',
    # '/root/zhangyi/ecg_ptbxl_benchmarking/output/exp1/',
    # '/root/zhangyi/ecg_ptbxl_benchmarking/output/exp1.1/',
    # '/root/zhangyi/ecg_ptbxl_benchmarking/output/exp1.1.1/',
    # '/root/zhangyi/ecg_ptbxl_benchmarking/output/exp2/',
    # '/root/zhangyi/ecg_ptbxl_benchmarking/output/exp3/',

    ]
    #exp0  exp1  exp1.1  exp1.1.1  exp2  exp3
    for outpath in path_list :
        with open(outpath +'data/'+'mlb.pkl', 'rb') as tokenizer:
            mlb = pickle.load( tokenizer)
            scp_list = list(mlb.classes_)

        y_test = np.load(outpath+'/data/y_test.npy', allow_pickle=True)
        y_val = np.load(outpath+'/data/y_val.npy', allow_pickle=True)
        for m in ["fastai_dsc_xresnet1d101"]:#sorted(os.listdir(outpath + 'models')):
            print(m)
            my_te_df = pd.DataFrame(columns = ['scp','precision','recall','f1','specificity','test_freq','thresholds'])
            mpath = outpath+ 'models/'+m+'/'
            rpath = outpath+ 'models/'+m+'/results/'
            # load predictions
            y_train_pred = np.load(mpath+'y_train_pred.npy', allow_pickle=True)
            y_val_pred = np.load(mpath+'y_val_pred.npy', allow_pickle=True)
            y_test_pred = np.load(mpath+'y_test_pred.npy', allow_pickle=True)

            # cal thresholds
            precision,recall,f1,specificity,threshold,tsf = [0]*y_test_pred[0],[0]*y_test_pred[0],[0]*y_test_pred[0],[0]*y_test_pred[0],[0]*y_test_pred[0],[0]*y_test_pred[0]

            for i in range(len(y_test_pred[0])):
                y_val_tmp  =y_val[:,i]
                y_val_pred_tmp=y_val_pred[:,i]
                p, r, thresholds = precision_recall_curve(y_val_tmp, y_val_pred_tmp)
                f = np.nan_to_num(2*p*r/(p+r))
                index = argmax(f)
                threshold[i] = thresholds[index]

                y_test_pred_tmp=y_test_pred[:,i]
                pos_ind = np.where(y_test_pred_tmp > threshold[i])
                neg_ind = np.where(y_test_pred_tmp <= threshold[i])
                y_test_pred_tmp[pos_ind] =1
                y_test_pred_tmp[neg_ind] =0
                y_test_tmp = y_test[:,i]
                pre, rec, f, sup = precision_recall_fscore_support(y_test_tmp, y_test_pred_tmp)
                precision[i] =pre[1]
                recall[i] = rec[1]
                f1[i] = f[1]
                specificity[i]= rec[0]
                tsf[i] = sum(y_test_tmp)/len(y_test_tmp)

            my_te_df["scp"] = scp_list
            my_te_df["precision"] = precision
            my_te_df["recall"] = recall
            my_te_df["f1"] = f1
            my_te_df['test_freq'] = tsf
            my_te_df['specificity'] = specificity
            my_te_df["thresholds"] = threshold
            my_te_df.to_csv(rpath+'my_results.csv')


if __name__ == "__main__":
    main()
