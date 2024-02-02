# Prepare data library ANA
""""
This library contains the following classes and functions:
- Get cognition scores (z-scores) from SPSS file
- Normalize z-scores
- Create lesion matrix
- Subject threshold
"""

import json
import math
import numpy as np
import os
import random
import torch
import SimpleITK as sitk
import warnings
from tqdm.notebook import tqdm
from seed import seed_everything
# seeds
np.random.seed(0) # seed for NumPy
# random.seed(0) # for python
torch.manual_seed(0) # seed for PyTorch
# torch.cuda.manual_seed(0)
# torch.cuda.manual_seed_all(0) # if you are using multi-GPU.
print("Global seed", seed_everything(0))

mni_cor = [2, 154, 21, 200, 19, 161]  # Voxels containing only brain tissue in MNI-space
crop_dim = [mni_cor[1] - mni_cor[0], mni_cor[3] - mni_cor[2], mni_cor[5] - mni_cor[4]]


#
# def get_noise():
#     # Temporary function to generate noise. Once satisfied with the method, this should be generated and saved as noisy scores.
#     # Noise implementation as done in Pustina et al. (2018) https://github.com/dorianps/LESYMAP/blob/master/R/simulateBehavior.R


def downsize_data(dir_lesion, dir_downsample, mni_coordinates=mni_cor):
    # Create lesion matrix filled with 3D image arrays
    if mni_coordinates is None:
        mni_coordinates = mni_cor

    # Create down sampled dataset directory
    if not os.path.exists(dir_downsample):
        os.mkdir(dir_downsample)

    # Loop through each file and concatenate the data
    for i, file_name in enumerate(dir_lesion):
        lesion_nii = sitk.ReadImage(file_name)
        crop_lesion = sitk.GetArrayFromImage(lesion_nii)[mni_coordinates[0]:mni_coordinates[1],
                      mni_coordinates[2]:mni_coordinates[3], mni_coordinates[4]:mni_coordinates[5]]
        image_lesion = sitk.GetImageFromArray(crop_lesion)

        lesion_name = file_name.split("\\", 1)[-1]
        ds_lesion_path = os.path.join(dir_downsample, lesion_name)
        sitk.WriteImage(image_lesion, ds_lesion_path)
        
        
def get_information(json_path, roi_config, noise_test=False):
    if not os.path.exists(json_path):
        warnings.warn("Json file does not exists in this path...")

    file = open(json_path)  # Open json file
    data = json.load(file)
    if noise_test:
        scores = data["Score 1"]
    else:
        scores = data["Score 1"][roi_config]
    included = data["Patient"]

    for i, pt in enumerate(included):
        if pt == "20253":
            del included[i]
            del scores[i]
    file.close()
    return included, scores


def get_selected_information(json_path, roi_config, selection):
    if not os.path.exists(json_path):
        warnings.warn("Json file does not exists in this path...")

    file = open(json_path)  # Open json file
    data = json.load(file)

    all_patients = data["Patient"]
    selected_pts, selected_score = [], []
    for i, pt in enumerate(all_patients):
        if pt in selection:
            # print(f"Patient {pt} is in selection")
            selected_pts.append(pt)
            selected_score.append(data["scores"][roi_config][i])

    file.close()
    return selected_pts, selected_score


def get_overfit_information(json_path, roi_config, overfit_ids, noise_test=False):
    if not os.path.exists(json_path):
        warnings.warn("Json file does not exists in this path...")

    file = open(json_path)  # Open json file
    data = json.load(file)
    overfit_indices, scores = [], []

    total_patients = data["patients"]
    for patient in overfit_ids:
        for i in range(len(total_patients)):
            if total_patients[i] == patient:
                overfit_indices.append(i)

    print(f'Overfit indices: {overfit_indices}')
    if noise_test:
        for index in overfit_indices:
            score = data["scores"][index]
            scores.append(score)
    else:
        for index in overfit_indices:
            score = data["scores"][roi_config][index]
            scores.append(score)
    print(f'Scores: {scores}')
    file.close()
    return overfit_ids, scores


def get_ids(dir_lesion):
    if not os.path.exists(dir_lesion):
        warnings.warn("Lesion directory does not exists...")
    ids = []
    img_paths = [f for f in sorted(os.listdir(dir_lesion)) if os.path.isfile(os.path.join(dir_lesion, f))]
    for i, lesion_path in enumerate(tqdm(img_paths)):
        if lesion_path.endswith(".nii.gz"):
            # Split the filename by underscores and extract the desired part
            patient_id = str(lesion_path.split('_')[1])
            ids.append(patient_id)
    return ids


def lesion_matrix(pt_ids, dir_lesion): 
    # Create lesion matrix filled with 3D image arrays
    if not os.path.exists(dir_lesion):
        warnings.warn("Lesion directory does not exists, will load only zero matrices...")
    raw_lesion_matrix = np.zeros((len(pt_ids), 152, 179, 142), np.int8)
    print("Creating a lesion matrix with the following dimensions: {}".format(np.shape(raw_lesion_matrix)))

    for i, patient_number in enumerate(tqdm(pt_ids)):
        # load and crop the lesion map
        lesion_path = os.path.join(dir_lesion, f"TRACEVCI_{str(patient_number)}_WMHinMNImasked.nii.gz")
        if os.path.exists(lesion_path):
            raw_lesion_matrix[i] = sitk.GetArrayFromImage(sitk.ReadImage(lesion_path))
            
    all_zeros = not np.any(raw_lesion_matrix)
    if all_zeros:
        warnings.warn("Lesion matrix is empty...")
    return raw_lesion_matrix


def repeat_data(data_size, included_pts, scores, raw_lesion_matrix):
    nr_patients = len(included_pts)

    patient_ids = np.zeros(data_size)
    idx = 0
    for i in range(nr_patients):
        for j in range(math.ceil(data_size / nr_patients)):
            if idx == len(patient_ids): continue
            patient_ids[idx] = int(included_pts[i])
            idx += 1

    cognitive_score = np.zeros(data_size)
    idx = 0
    for i in range(nr_patients):
        for j in range(math.ceil(data_size / nr_patients)):
            if idx == len(cognitive_score): continue
            cognitive_score[idx] = scores[i]
            idx += 1

    if raw_lesion_matrix.ndim > 2:
        matrix = np.zeros(
            (data_size, np.shape(raw_lesion_matrix)[1], np.shape(raw_lesion_matrix)[2], np.shape(raw_lesion_matrix)[3]))
    else:
        matrix = np.zeros((data_size, np.shape(raw_lesion_matrix)[1]))
    idx = 0
    for i in range(nr_patients):
        for j in range(math.ceil(data_size / nr_patients)):
            if idx == len(matrix): continue
            matrix[idx] = raw_lesion_matrix[i]
            idx += 1

    print(f'New matrix size: {np.shape(matrix)}')

    return patient_ids, cognitive_score, matrix


def pt_threshold(raw_lesion_matrix, threshold, reduce_voxels=False, feature_selection=False,
                 feature_selection_path=None, alpha=0.05):
    
    # Create mask where lesion prevalence >= threshold
    lesion_prevalence = np.sum(raw_lesion_matrix, 0)
    if reduce_voxels:
        if feature_selection:
            p_arr = sitk.GetArrayFromImage(sitk.ReadImage(feature_selection_path)).ravel()
            prevalence_mask = (lesion_prevalence >= threshold) & (p_arr > (1 - alpha))
        else:
            prevalence_mask = lesion_prevalence >= threshold

        prev_mask_indices = np.nonzero(prevalence_mask)

        if len(prev_mask_indices) > 1:
            lesion_matrix = np.zeros(np.shape(raw_lesion_matrix), np.int8)
            lesion_matrix[:, prev_mask_indices[0], prev_mask_indices[1], prev_mask_indices[2]] = raw_lesion_matrix[:,
                                                                                                 prev_mask_indices[0],
                                                                                                 prev_mask_indices[1],
                                                                                                 prev_mask_indices[2]]
        else:
            lesion_matrix = raw_lesion_matrix[:, np.argwhere(prevalence_mask).ravel()]

    else:
        prevalence_mask = lesion_prevalence >= threshold
        prev_mask_indices = np.nonzero(prevalence_mask)
        lesion_matrix = np.zeros(np.shape(raw_lesion_matrix), np.int8)

        if len(prev_mask_indices) > 1:
            lesion_matrix[:, prev_mask_indices[0], prev_mask_indices[1], prev_mask_indices[2]] = raw_lesion_matrix[:,
                                                                                                 prev_mask_indices[0],
                                                                                                 prev_mask_indices[1],
                                                                                                 prev_mask_indices[2]]
        else:
            lesion_matrix[:, prev_mask_indices[0]] = raw_lesion_matrix[:, prev_mask_indices[0]]

    return lesion_matrix.astype(np.float32), prev_mask_indices, prevalence_mask


def volume_correction(lesion_matrix, mode):
    lesion_summed = np.sum(lesion_matrix, axis=1)
    if 'noscale' == mode:
        return lesion_matrix, lesion_summed
    elif 'sqrt' == mode:
        rtSumsqr = np.sqrt(np.sum(lesion_matrix, axis=1))
        # rtSumsqr = np.sqrt(lesion_summed)
        result = np.divide(lesion_matrix.astype(np.float32), rtSumsqr.astype(np.float32)[:, None],
                           out=np.zeros_like(lesion_matrix, np.float32), where=rtSumsqr[:, None] != 0)

        for i, normalised in enumerate(result):
            norm = np.linalg.norm(normalised)
            # print(f"{i} norm (should be 1: {norm})")

        return result, lesion_summed

    #     elif 'sqrt' == mode:
    #         rtSumsqr = np.sqrt(np.sum(lesion_matrix, axis=1))

    else:
        raise NotImplementedError(f'Mode "{mode}" is not supported')
        
        
def toy_prevalence(raw_lesion_matrix):
    # Create a prevalence map
    lesion_prevalence = np.sum(raw_lesion_matrix, 0)
    prevalence_mask = lesion_prevalence >= 0
    prev_mask_indices = np.nonzero(lesion_prevalence)

    return raw_lesion_matrix.astype(np.float32), prev_mask_indices, prevalence_mask


def toy_matrix(pt_ids, dir_lesion, toy_size):
    # Create lesion matrix filled with 3D image arrays
    if not os.path.exists(dir_lesion):
        warnings.warn("Lesion directory does not exists, will load only zero matrices...")

    size = int(toy_size[0])
    raw_lesion_matrix = np.zeros((len(pt_ids), size, size, size), np.int8)  # 3,3,3) , np.int8)#
    print("Creating a lesion matrix with the following dimensions: {}".format(np.shape(raw_lesion_matrix)))

    for i, patient_number in enumerate(tqdm(pt_ids)):
        # load and crop the lesion map
        lesion_path = os.path.join(dir_lesion, "toy_image_" + str(patient_number) + ".nii.gz")
        if os.path.exists(lesion_path):
            lesion_nii = sitk.ReadImage(lesion_path)
            raw_lesion_matrix[i] = sitk.GetArrayFromImage(lesion_nii)

    all_zeros = not np.any(raw_lesion_matrix)
    if all_zeros:
        warnings.warn("Lesion matrix is empty...")

    return raw_lesion_matrix



def lesion_matrix_old(pt_ids, dir_lesion, mni_coordinates=mni_cor): 
    raw_lesion_matrix = np.zeros((pt_ids, crop_dim[0], crop_dim[1], crop_dim[2]), np.int8)
    print("Creating a lesion matrix with the following dimensions: {}".format(np.shape(raw_lesion_matrix)))

    # Sort the file list to ensure consistent order
    dir_lesion.sort()
    lesion_path = []

    # Loop through each file and concatenate the data
    for i, file_name in enumerate(dir_lesion):
        lesion_nii = sitk.ReadImage(file_name)
        raw_lesion_matrix[i] = sitk.GetArrayFromImage(lesion_nii)[mni_coordinates[0]:mni_coordinates[1],
                               mni_coordinates[2]:mni_coordinates[3], mni_coordinates[4]:mni_coordinates[5]]
        lesion_file = file_name.split("\\", 1)[-1]
        lesion_path.append(lesion_file)

    all_zeros = not np.any(raw_lesion_matrix)
    if all_zeros:
        warnings.warn("Lesion matrix is empty...")
    return raw_lesion_matrix, lesion_path


