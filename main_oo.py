import os
import glob
import SimpleITK as sitk
import matplotlib.pyplot as plt
import torch

# Personal libraries
from dataset import get_data_cv
from cv_class import CrossValidation
import analysis
from XAI_methods import evaluate_XAI


""" One-Output model"""
## Dataset ##

base = os.getcwd()
dir_lesion = os.path.join(base, "images/TRACE-VCI-DOWNSIZED")
dir_experiments = os.path.join(base, 'experiments/one_output_3D')
dir_models = os.path.join(base, 'experiments/one_output_3D/weights/no_noise')
output = "one"
xai_method = "ggc"
dataset = get_data_cv(dir_experiments, dir_lesion, output)


""" Proof-of-concept Experiment"""
# Default
config = {
    'model': 'CNN2',
    'learning_rate': 0.001,
    'optimizer': 'AdamW',
    'lr_scheduler': 'None',
    'weight_decay': 0.01, 
    'device_ids': [0],
    'path_save_model': dir_models,
    'n_classes': 1,
    'project_wandb': 'cv_one-output-3d',
    'noise_weight': None,
    'xai_method': xai_method,
    'xai_path': "saved_xai_methods/"+xai_method,
    'num_epochs': None # None or string, depends on the fold name, if you want to keep track of the epochs used then write it down
}

print("CNN with without noise")
model = CrossValidation(config=config)
# model.cross_validation(dataset, k_folds=5, num_epochs=60) # Uncomment for training the model
model.test_crossval(dataset, k_folds=5, make_xai=True) # Uncomment to test it again or recalculate XAI
torch.cuda.empty_cache()


""" Noise Experiments"""

dir_lesion = os.path.join(base, "images/TRACE-VCI-DOWNSIZED")
dir_experiments = os.path.join(base, 'experiments/one_output_3D')
dir_models = os.path.join(base, 'experiments/one_output_3D/weights/gaussian')
torch.cuda.empty_cache()
# 0.1
print("CNN with 10% noise")
noise_weight =  '0.1'
noise_dataset = get_data_cv(dir_experiments, dir_lesion, output, True, noise_weight)
config.update({'project_wandb': 'cv_'+ output + "_gaussian_0.1", 'noise_weight': noise_weight, 'path_save_model': dir_models})
model = CrossValidation(config=config)
# model.cross_validation(noise_dataset, k_folds=5, num_epochs=80)
model.test_crossval(dataset, k_folds=5, make_xai=True)
torch.cuda.empty_cache()

# 0.2
print("CNN with 20% noise")
noise_weight =  '0.2'
noise_dataset = get_data_cv(dir_experiments, dir_lesion, output, True, noise_weight)
config.update({'project_wandb': 'cv_'+ output + "_gaussian_0.2", 'noise_weight': noise_weight})
model = CrossValidation(config=config)
# model.cross_validation(noise_dataset, k_folds=5, num_epochs=80)
model.test_crossval(dataset, k_folds=5, make_xai=True)
torch.cuda.empty_cache()

# 0.3
print("CNN with 30% noise")
noise_weight =  '0.3'
noise_dataset = get_data_cv(dir_experiments, dir_lesion, output, True, noise_weight)
config.update({'project_wandb': 'cv_'+ output + "_gaussian_0.3", 'noise_weight': noise_weight})
model = CrossValidation(config=config)
# model.cross_validation(noise_dataset, k_folds=5, num_epochs=80)
model.test_crossval(dataset, k_folds=5, make_xai=True)
torch.cuda.empty_cache()

# 0.4
print("CNN with 40% noise")
noise_weight =  '0.4'
noise_dataset = get_data_cv(dir_experiments, dir_lesion, output, True, noise_weight)
config.update({'project_wandb': 'cv_'+ output + "_gaussian_0.4", 'noise_weight': noise_weight})
model = CrossValidation(config=config)
# model.cross_validation(noise_dataset, k_folds=5, num_epochs=80)
model.test_crossval(dataset, k_folds=5, make_xai=True)
torch.cuda.empty_cache()

# 0.5
print("CNN with 50% noise")
noise_weight =  '0.5'
noise_dataset = get_data_cv(dir_experiments, dir_lesion, output, True, noise_weight)
config.update({'project_wandb': 'cv_'+ output + "_gaussian_0.5", 'noise_weight': noise_weight})
model = CrossValidation(config=config)
# model.cross_validation(noise_dataset, k_folds=5, num_epochs=80)
model.test_crossval(dataset, k_folds=5, make_xai=True)
torch.cuda.empty_cache()


""" XAI evaluation """

model_name = 'cv_CNN2_AdamW_lr0.001sch_None'
save_analysis_path =  os.path.join(base, 'experiments/one_output_3D/analysis')
xai_path = os.path.join(base, 'experiments/one_output_3D/weights/no_noise/') # Chosen model
path_noise = os.path.join(base, 'experiments/one_output_3D/weights/gaussian/') # Chosen model

evaluate_XAI(xai_path+model_name, model_name, output, save_analysis_path, make_analysis=True, xai=xai_method, path_noise=path_noise+model_name, noise=True, score=0, name="CNN", dependency=None, layer="_conv1") 
