# Simulation library
"""
Simulation library based on the SVR paper simulation method
 https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4213345/

This library contains the following classes and functions:
-  
"""
import json
import random
import numpy as np
import os
import torch
import pandas as pd
import scipy
import SimpleITK as sitk
import sklearn.preprocessing as preproc
from math import ceil
from seed import seed_everything
import matplotlib.pyplot as plt
# seeds
np.random.seed(0) # seed for NumPy
# random.seed(0) # for python
torch.manual_seed(0) # seed for PyTorch
# torch.cuda.manual_seed(0)
# torch.cuda.manual_seed_all(0) # if you are using multi-GPU.
print("Global seed", seed_everything(0))


######################### Define three ROIs #########################

########## ROI helper functions ##########

def get_random_roi_origin(indices):
    # Return a random [x,y,z] coordinate from the the list of indices
    random_index = np.random.choice(range(len(indices[0])))
    random_x = indices[0][random_index]
    random_y = indices[1][random_index]
    random_z = indices[2][random_index]
    coordinate = [random_x, random_y, random_z]

    return coordinate


def remove_roi_from_map(roi_origin, roi_radius, binary_map):
    # ROI origin is a coordinate of shape [x,y,z]
    x = roi_origin[0]
    y = roi_origin[1]
    z = roi_origin[2]

    new_map = binary_map

    # Marge moet misschien eerder 2*roi_size zijn ################################### controleren
    for x_idx in range(x - ceil(1.5 * roi_radius), x + ceil(1.5 * roi_radius)):
        for y_idx in range(y - ceil(1.5 * roi_radius), y + ceil(1.5 * roi_radius)):
            for z_idx in range(z - ceil(1.5 * roi_radius), z + ceil(1.5 * roi_radius)):
                new_map[x_idx, y_idx, z_idx] = 0

    return new_map


def get_roi_origin(indices, roi_radius, roi_map):
    # Find a random ROI origin from the list of indices, remove the ROI + margin of 1/2 ROI size from ROI_map
    roi_origin = get_random_roi_origin(indices)
    updated_map = remove_roi_from_map(roi_origin, roi_radius, roi_map)
    updated_indices = np.nonzero(updated_map)

    return roi_origin, updated_indices, updated_map


def get_three_roi_origins(indices, roi_radius, roi_map):
    # Define_ROI three times and return the origin from all ROIs
    origin_1, indices_1, map_1 = get_roi_origin(indices, roi_radius, roi_map)
    origin_2, indices_2, map_2 = get_roi_origin(indices_1, roi_radius, map_1)
    origin_3, indices_3, map_3 = get_roi_origin(indices_2, roi_radius, map_2)

    return origin_1, origin_2, origin_3


def set_roi(roi_matrix, roi_origin, roi_radius, set_val=1):
    # Mark the ROI in roi_matrix based on the roi_origin with a value of set_val
    x = roi_origin[0]
    y = roi_origin[1]
    z = roi_origin[2]

    for x_idx in range(x - roi_radius, x + roi_radius):
        for y_idx in range(y - roi_radius, y + roi_radius):
            for z_idx in range(z - roi_radius, z + roi_radius):
                roi_matrix[x_idx, y_idx, z_idx] = set_val

    return roi_matrix


def save_rois_image(lesion_mask, origin_1, origin_2, origin_3, roi_radius, path):
    # Save an image with the three ROIs marked at paths
    roi_matrix = np.zeros(np.shape(lesion_mask))

    updated_map_1 = set_roi(roi_matrix, origin_1, roi_radius)
    updated_map_2 = set_roi(updated_map_1, origin_2, roi_radius)
    updated_map_3 = set_roi(updated_map_2, origin_3, roi_radius)

    updated_map_3 = updated_map_3.astype(np.double)

    roi_img = sitk.GetImageFromArray(updated_map_3)
    sitk.WriteImage(roi_img, path)
    print('Saved ROI image')

    return updated_map_3

def save_roi_image(lesion_mask, origin, roi_radius, path):
    # Save an image with the three ROIs marked at paths
    roi_matrix = np.zeros(np.shape(lesion_mask))

    updated_map_1 = set_roi(roi_matrix, origin, roi_radius)
    updated_map_2 = updated_map_1.astype(np.double)

    roi_img = sitk.GetImageFromArray(updated_map_2)
    sitk.WriteImage(roi_img, path)
    print('Saved ROI image')

    return updated_map_2


def get_roi_indices(lesion_mask, roi_origin, roi_radius, set_val=1):
    # get ROI indices from lesion mask
    matrix = set_roi(np.zeros(np.shape(lesion_mask)), roi_origin, roi_radius)
    return np.nonzero(matrix)


def get_separate_indices(lesion_mask, roi_origins, roi_radius):
    """" A list of indices is returned indices[0] is the list of indices for ROI 1, 
    indices[1] is for ROI 2, etc """""

    indices = []
    indices_1 = get_roi_indices(lesion_mask, roi_origins[0], roi_radius)
    indices_2 = get_roi_indices(lesion_mask, roi_origins[1], roi_radius)
    indices_3 = get_roi_indices(lesion_mask, roi_origins[2], roi_radius)
    indices.append(indices_1)
    indices.append(indices_2)
    indices.append(indices_3)
    return indices

    ########## Other helper functions ##########


def def_lesion_mask(lesion_matrix, roi_threshold, mask_path=None):
    lesion_prevalence = np.sum(lesion_matrix, 0)
    lesion_mask = lesion_prevalence >= roi_threshold

    # Save lesion image
    #     lesion_img = sitk.GetImageFromArray(lesion_prevalence.astype(np.double))
    #     sitk.WriteImage(lesion_img, r"../simulation/lesion_matrix.nii.gz")

    # Save Mask image
    mask_path = os.path.join(os.getcwd(), "images/lesion_mask.nii.gz")
    print(mask_path)
    lesion_mask_img = sitk.GetImageFromArray(lesion_mask.astype(np.double))
    sitk.WriteImage(lesion_mask_img, mask_path)
    print('Saved lesion mask image')

    return lesion_mask


def erosion(lesion_mask, roi_radius):
    # Perform an erosion function on the lesion mask
    eroded = scipy.ndimage.binary_erosion(lesion_mask, structure=np.ones(
        (roi_radius * 2 + 1, roi_radius * 2 + 1, roi_radius * 2 + 1)))  # Controleren
    eroded_indices = np.nonzero(eroded)

    return eroded, eroded_indices


    ########## Main function ##########


def get_rois(lesion_matrix, roi_threshold, roi_radius, folder, i=0):
    # Return three random ROI origins for ROIs with diameter of roi_size; and a masked ROI array (only ROI locations
    # are 1) All voxels within the three ROIs have lesions in >roi_threshold patients

    lesion_mask = def_lesion_mask(lesion_matrix, roi_threshold)
    eroded, eroded_indices = erosion(lesion_mask, roi_radius)
    roi_1, roi_2, roi_3 = get_three_roi_origins(eroded_indices, roi_radius, eroded)
    print('ROI origins: {}, {}, {}'.format(roi_1, roi_2, roi_3))
    roi_array = save_rois_image(lesion_mask, roi_1, roi_2, roi_3, roi_radius,
                                path=os.path.join(os.getcwd(), "images", "roi_image_v2_{}.nii.gz".format(i)))
    
    roi_array1 = save_roi_image(lesion_mask, roi_1, roi_radius,
                                path=os.path.join(os.getcwd(), "images", "roi_image_1.nii.gz"))
    roi_array2 = save_roi_image(lesion_mask, roi_2, roi_radius,
                                path=os.path.join(os.getcwd(), "images", "roi_image_2.nii.gz"))
    roi_array3 = save_roi_image(lesion_mask, roi_3, roi_radius,
                                path=os.path.join(os.getcwd(), "images", "roi_image_3.nii.gz"))
    # "../simulation/roi_image_{}.nii.gz".format(i))

    origins = [roi_1, roi_2, roi_3]

    return origins, roi_array  # roi_1, roi_2, roi_3, roi_array


# ######################## Artificial cognition scores ########################
# Noise implementation as done in
# Pustina et al. (2018) https://github.com/dorianps/LESYMAP/blob/master/R/simulateBehavior.R

def save_df(data1, data2, path, noise_type, score):
    # save noise
    dn = {'Errormat': data1, 'Noise': data2}
    df_n = pd.DataFrame(data=dn)
    df_n.to_pickle(os.path.join(path, "ds_" + noise_type + "_noise" + score + ".pkl"))


def save_simulation_data(data_df, lesion_matrix, roi_origins, results_path, json_type=False, noise=False):
    os.makedirs(results_path, exist_ok=True)
    if json_type:  # Save data as JSON
        with open(os.path.join(results_path, "ds_simulation_scores.json"), 'w') as json_file:
            json.dump(data_df, json_file)
    else:
        # Save data into local files
        data_df.to_pickle(os.path.join(results_path, "ds_simulation_scores.pkl"))
        if not noise:
            np.save(os.path.join(results_path, 'ds_lesion_matrix.npy'), lesion_matrix)
            np.save(os.path.join(results_path, 'ds_roi_array.npy'), roi_origins)


def add_uniform_noise(patient_scores, roi_scores, error_weight):
    if not (0 <= error_weight <= 1):
        raise ValueError("error_weight must be between 0 and 1")

    score_weight = 1 - error_weight
    # calculate amount of artificial score
    one_score = [value * score_weight for value in patient_scores]
    multi_score = [[value * score_weight for value in sublist] for sublist in roi_scores]

    # calculate noise
    errormat = np.random.uniform(0, 1, size=len(one_score))

    noise = [value * error_weight for value in errormat]

    # obtain final score
    oo_score = [x + y for x, y in zip(one_score, noise)]
    mo_score = [[n + sublist[0], n + sublist[1], n + sublist[2]] for n, sublist in zip(noise, multi_score)]

    # save noise
    path_mo = "experiments/multi_output_3D/noise/uniform/" + str(error_weight)
    path_oo = "experiments/one_output_3D/noise/uniform/" + str(error_weight)
    os.makedirs(path_oo, exist_ok=True)
    os.makedirs(path_mo, exist_ok=True)
    save_df(errormat, noise, path_oo, "uniform", "1")
    save_df(errormat, noise, path_mo, "uniform", "1")

    return oo_score, mo_score


def add_gaussian_noise(patient_scores, roi_scores, error_weight):
    if not (0 <= error_weight <= 1):
        raise ValueError("error_weight must be between 0 and 1")

    score_weight = 1 - error_weight
    one_score = [value * score_weight for value in patient_scores]
    multi_score = [[value * score_weight for value in sublist] for sublist in roi_scores]

    std_score_oo = np.std(patient_scores)
    errormat_oo = np.random.normal(0, std_score_oo, size=len(one_score))
    noise_oo = [value * error_weight for value in errormat_oo]
#     Uncomment to plot the Noise distribution
#     bins = 50  # fixed bin size
#     plt.hist(noise_oo, bins=bins, alpha=0.5)
#     plt.title('Noise distribution')
#     plt.xlabel('Score')
#     plt.ylabel('count')
#     plt.show()
    
    std_score1_mo, std_score2_mo, std_score3_mo = np.std(np.array(roi_scores), axis=0)
    errormat1_mo = np.random.normal(0, std_score1_mo, size=np.shape(multi_score)[0])
    noise1_mo = [value * error_weight for value in errormat1_mo]

    errormat2_mo = np.random.normal(0, std_score2_mo, size=len(one_score))
    noise2_mo = [value * error_weight for value in errormat2_mo]

    errormat3_mo = np.random.normal(0, std_score3_mo, size=len(one_score))
    noise3_mo = [value * error_weight for value in errormat3_mo]

    oo_score = [x + y for x, y in zip(one_score, noise_oo)]
    
    # different noise to each ROIs, depends on their std
    mo_score = [[n1 + sublist[0], n2 + sublist[1], n3 + sublist[2]]
                for n1, n2, n3, sublist in zip(noise1_mo, noise2_mo, noise3_mo, multi_score)]

    # save noise
    path_mo = "experiments/multi_output_3D/noise/gaussian/" + str(error_weight)
    path_oo = "experiments/one_output_3D/noise/gaussian/" + str(error_weight)
    os.makedirs(path_oo, exist_ok=True)
    os.makedirs(path_mo, exist_ok=True)

    save_df(errormat_oo, noise_oo, path_oo, "gaussian", "1")
    save_df(errormat1_mo, noise1_mo, path_mo, "gaussian", "1")
    save_df(errormat2_mo, noise2_mo, path_mo, "gaussian", "2")
    save_df(errormat3_mo, noise3_mo, path_mo, "gaussian", "3")

    return oo_score, mo_score


def normalize_oo_noisy_scores(noisy_scores_oo):
    min_value = np.min(noisy_scores_oo)
    max_value = np.max(noisy_scores_oo)
    range_value = max_value - min_value
    normalized_noisy_scores_oo = np.array([(x - min_value) / range_value if range_value != 0 else 0.0 for x in noisy_scores_oo])
    return normalized_noisy_scores_oo


def normalize_mo_noisy_scores(noisy_scores_mo):
    # Find the minimum and maximum values for each column
    min_value_column0 = min(sublist[0] for sublist in noisy_scores_mo)
    max_value_column0 = max(sublist[0] for sublist in noisy_scores_mo)

    min_value_column1 = min(sublist[1] for sublist in noisy_scores_mo)
    max_value_column1 = max(sublist[1] for sublist in noisy_scores_mo)

    min_value_column2 = min(sublist[2] for sublist in noisy_scores_mo)
    max_value_column2 = max(sublist[2] for sublist in noisy_scores_mo)

    # Normalize the values in each sublist between 0 and 1
    normalized_noisy_scores_mo = np.array([
        [(value[0] - min_value_column0) / (max_value_column0 - min_value_column0),
         (value[1] - min_value_column1) / (max_value_column1 - min_value_column1),
         (value[2] - min_value_column2) / (max_value_column2 - min_value_column2)]
        for value in noisy_scores_mo
    ])

    return normalized_noisy_scores_mo


def calculate_lvr(pt_matrix: np.array, roi_indices: np.array, roi_radius: int) -> float:
    # Calculate lesion volume ratio
    # nr of lesion voxels in ROI / total nr of voxels in ROI
    total_voxels = (2 * roi_radius) * (2 * roi_radius) * (2 * roi_radius)
    lesion_voxels = 0

    x = roi_indices[0]
    y = roi_indices[1]
    z = roi_indices[2]

    for i in range(len(x)):
        if pt_matrix[x[i]][y[i]][z[i]]:
            lesion_voxels += 1

    result = lesion_voxels / total_voxels
    return result


def calculate_percentage_lvr(pt_matrix: np.array, roi_indices: np.array, roi_radius: int) -> float:
    # Calculate lesion volume ratio
    # nr of lesion voxels in ROI / total nr of voxels in ROI
    total_voxels = (2 * roi_radius) * (2 * roi_radius) * (2 * roi_radius)
    lesion_voxels = 0

    x = roi_indices[0]
    y = roi_indices[1]
    z = roi_indices[2]

    for i in range(len(x)):
        if pt_matrix[x[i]][y[i]][z[i]]:
            lesion_voxels += 1

    result = lesion_voxels / total_voxels
    return result

def get_artificial_score(pt_matrix, roi_weights, indices, roi_radius, a, b):
    """"
    Parameters
    - pt_matrix: patient's lesion matrix, for whom the artificial cognition score should be calculated
    - roi_weights: the weights for the three ROIs
    - lv_ratio: lesion volume ratio

    """
    # one score per patients: sum the amount of lesion there is in all ROIs
    sum_rois = 0
    sum_rois += roi_weights[0] * calculate_lvr(pt_matrix, indices[0], roi_radius)
    sum_rois += roi_weights[1] * calculate_lvr(pt_matrix, indices[1], roi_radius)
    sum_rois += roi_weights[2] * calculate_lvr(pt_matrix, indices[2], roi_radius)

    # one score per roi, sum amount of lesion there is in each roi separately
    score_roi_1 = roi_weights[0] * calculate_lvr(pt_matrix, indices[0], roi_radius)
    score_roi_2 = roi_weights[1] * calculate_lvr(pt_matrix, indices[1], roi_radius)
    score_roi_3 = roi_weights[2] * calculate_lvr(pt_matrix, indices[2], roi_radius)
    
    # score intercorrelation 
    score_ic_1 = score_roi_1 # ROI which is not correlated to others as a baseline check. 
    score_ic_2 = a*score_roi_2+(1-a)*score_roi_3
    score_ic_3 = b*score_roi_3+(1-b)*score_roi_1
    
    # save them in alist
    scores = [score_roi_1, score_roi_2, score_roi_3]
    intercorrelated_scores = [score_ic_1, score_ic_2, score_ic_3]
    return sum_rois, scores, intercorrelated_scores


def get_all_artificial_scores(lesion_matrix, roi_weights, indices, roi_radius, noise, error_weight, noise_type, a, b):
    # Call get_artificial_score for all included patients
    patient_scores, roi_scores, ic_scores = [], [], []

    # loop through all patient's lesion to obtain artificial score
    for pt_matrix in lesion_matrix:
        sum_rois, r_score, i_score = get_artificial_score(pt_matrix, roi_weights, indices, roi_radius, a, b)
        patient_scores.append(sum_rois)  # one-output
        roi_scores.append(r_score)  # multi-output: one score per roi
        ic_scores.append(i_score)
        
    # if noise is True, recalculate the artificial scores
    if noise:
        if noise_type == "uniform":
            patient_scores, roi_scores = add_uniform_noise(patient_scores, roi_scores, error_weight)
        elif noise_type == "gaussian":
            patient_scores, roi_scores = add_gaussian_noise(patient_scores, roi_scores, error_weight)
        else:
            raise ValueError("noise_type not specified")
    return np.array(patient_scores), np.array(roi_scores), np.array(ic_scores)


def get_lesion_matrix(directory):
    # Sort the file list to ensure consistent order
    directory.sort()
    lesion_path = []
    # Loop through each file and concatenate the data
    for i, file_name in enumerate(directory):
        img = sitk.ReadImage(file_name)
        data = sitk.GetArrayFromImage(img)
        lesion_file = file_name.split("\\", 1)[-1]

        lesion_path.append(lesion_file)

        # Initialize an empty array to store the concatenated data
        if i == 0:
            size = img.GetSize()
            lesion_matrix = np.empty((len(directory), size[2], size[1], size[0]))

        lesion_matrix[i] = data

    return lesion_matrix, lesion_path


########## Main function ##########


def get_scores(lesion_matrix, roi_origins, roi_threshold, roi_weights, roi_radius,
               noise=False, error_weight=0.3, noise_type="gaussian", normalize=True, a=0.7, b=0.7):
    """
    :param error_weight:
    :param noise_type:
    :param noise:
    :param lesion_matrix: 821 patients, with 152x179x142 matrix
    :param roi_origins: origins from the roi inn pixel/voxel
    :param roi_threshold: 10 patients at least
    :param roi_weights: by now 1,c1,1 same weight for every roi
    :param roi_radius: 5
    :param normalize:
    :return:
    """
    # Artificial scores are ordered the same way as the lesion matrix is ordered
    lesion_prevalence = np.sum(lesion_matrix, 0)  # sum all lesion maps together
    lesion_mask = lesion_prevalence >= roi_threshold  # take those lesion regions that are at least in 10 patients

    # Get list of indices is returned indices[0] is the list of indices for ROI 1, indices[1] is for ROI 2
    roi_indices = get_separate_indices(lesion_mask, roi_origins, roi_radius)  # Get all the ROI indices

    # Obtain artificial scores one/multi-output (respectively)
    artificial_scores_patient, roi_artificial_scores, ic_artificial_scores = get_all_artificial_scores(lesion_matrix, roi_weights,
                                                                                                       roi_indices, roi_radius,
                                                                                                       noise, error_weight,
                                                                                                       noise_type, a, b)

    # artificial scores between 0-1
    if normalize: 
        artificial_scores_patient = preproc.normalize([artificial_scores_patient], norm='max')
        
    return artificial_scores_patient, roi_artificial_scores, ic_artificial_scores


######################### Get patient ID #########################
mni_cor = [2, 154, 21, 200, 19, 161]


def get_all_ids(dir_lesion):
    patient_numbers = os.listdir(dir_lesion)  # List of patients with available images
    pt_ids = []
    for i in range(len(patient_numbers)):
        if patient_numbers[i][9:14].isnumeric():
            pt_ids.append(int(patient_numbers[i][9:14]))

    return pt_ids


def get_dataset_ids(dir_lesion, folder, included_pts):
    dir_path = os.path.join(dir_lesion, folder)
    img_paths = [f for f in sorted(os.listdir(dir_path)) if os.path.isfile(os.path.join(dir_path, f))]

    ids = [int(path[9:14]) for path in img_paths]
    ids = [patient_id for patient_id in ids if patient_id in included_pts]

    return ids

# %%
