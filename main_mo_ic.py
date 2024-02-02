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


"""Dependency Multi-Output model"""

## Dataset ##
base = os.getcwd()

# Configuration 
dependency_weight =  '0.6' # 0.9, 0.8, 0.7
output = "multi"
xai_method = "gs"
dir_lesion = os.path.join(base, "images/TRACE-VCI-DOWNSIZED")
dir_experiment = os.path.join(base, 'experiments/ic_multi_output_3D')
dir_models = os.path.join(base, 'experiments/ic_multi_output_3D/weights', dependency_weight)
epochs = 70
model_type = "CNN2" 

print("Using ", model_type," and dependency weight: "+ dependency_weight)

# CNN
config = {
    'model': model_type,
    'learning_rate': 0.001,
    'optimizer': 'AdamW',
    'lr_scheduler': 'None',
    'weight_decay': 0.01,
    'device_ids': [0],
    'path_save_model': dir_models,
    'n_classes': 3,
    'project_wandb': 'cv_multi-output-3d',
    'noise_weight': None,
    'xai_method': xai_method,
    'xai_path': "saved_xai_methods/"+xai_method,
    'num_epochs': str(epochs)
}

dependency_dataset = get_data_cv(dir_experiment, dir_lesion, output, False, dependency_weight, True)
config.update({'project_wandb': 'cv_'+ output + "_dependency_"+dependency_weight})
model = CrossValidation(config=config)
model.cross_validation(dependency_dataset, k_folds=5, num_epochs=epochs) # Uncommon to train
# model.test_crossval(dependency_dataset, k_folds=5)
torch.cuda.empty_cache()


""" XAI evaluation """
model_name = "cv_"+model_type+"_AdamW_lr0.001sch_None_epochs"+str(epochs)
save_analysis_path =  os.path.join(base, 'experiments/ic_multi_output_3D/analysis')
xai_path = os.path.join(base, 'experiments/ic_multi_output_3D/weights/'+dependency_weight+"/"+model_name+"/") # Chosen model
roi_path = os.path.join(base, "images")

analysis.get_RC_rate_roi(xai_path, roi_path, save_analysis_path, model_name, xai_method, score=1, dependency_weight=dependency_weight)
analysis.get_RC_rate_roi(xai_path, roi_path, save_analysis_path, model_name, xai_method, score=2, dependency_weight=dependency_weight)
analysis.get_RC_rate_roi(xai_path, roi_path, save_analysis_path, model_name, xai_method, score=3, dependency_weight=dependency_weight)

