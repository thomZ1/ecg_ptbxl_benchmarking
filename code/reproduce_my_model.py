from experiments.scp_experiment import SCP_Experiment
from utils import utils
# model configs
from configs.fastai_configs import *
from configs.wavelet_configs import *


def main():
    
    datafolder = '../data/ptbxl/'
    datafolder_icbeb = '../data/ICBEB/'
    outputfolder = '../output/'

    conf_my_dsc_xresnet1d101 = {'modelname':'fastai_dsc_xresnet1d101', 'modeltype':'fastai_model', 
    'parameters':dict()}
    conf_fastai_xception = {'modelname':'fastai_xception', 'modeltype':'fastai_model', 
    'parameters':dict()}
    fastai_se_resnext101 = {'modelname':'fastai_se_resnext101', 'modeltype':'fastai_model', 
    'parameters':dict()}
    
    models = [
            fastai_se_resnext101,
            conf_fastai_xception,
            conf_my_dsc_xresnet1d101,
            
        ]

    ##########################################
    # STANDARD SCP EXPERIMENTS ON PTBXL
    ##########################################

    experiments = [
        ('exp0', 'all'),
        # ('exp1', 'diagnostic'),
        # ('exp1.1', 'subdiagnostic'),
        # ('exp1.1.1', 'superdiagnostic'),
        # ('exp2', 'form'),
        # ('exp3', 'rhythm')
       ]

    for name, task in experiments:
        e = SCP_Experiment(name, task, datafolder, outputfolder, models)
        e.prepare()
        e.perform()
        e.evaluate()

    # generate greate summary table
    utils.generate_ptbxl_summary_table()

    ##########################################
    # EXPERIMENT BASED ICBEB DATA
    ##########################################

    e = SCP_Experiment('exp_ICBEB', 'all', datafolder_icbeb, outputfolder, models)
    e.prepare()
    e.perform()
    e.evaluate()

    # generate greate summary table
    utils.ICBEBE_table()

if __name__ == "__main__":
    main()
