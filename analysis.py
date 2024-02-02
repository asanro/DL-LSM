   # Analysis library
"""
This library contains the following classes and functions:
- Perform ordinary least squares analysis based on volume alone
- Functions to compute an ROC curve
- Functions to compute a PR curve
"""
import json
import random
import numpy as np
import os
import SimpleITK as sitk
import statsmodels.api as sm
import torch
import numpy as np
from sklearn import metrics
from tqdm.notebook import tqdm
from seed import seed_everything
from matplotlib import pyplot as plt
from scipy.stats import pearsonr

# seeds
# np.random.seed(0) # seed for NumPy
# # random.seed(0) # for python
# torch.manual_seed(0) # seed for PyTorch
# torch.cuda.manual_seed(0)
# torch.cuda.manual_seed_all(0) # if you are using multi-GPU.
print("Global seed", seed_everything(0))

torch.cuda.empty_cache()

device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


""" Analysis for dependency/intercorrelation model experiment """ 
            
def calculate_RC_score(prediction, ground_truth):
    common = np.sum(prediction * ground_truth)
    sum_prediction = np.sum(prediction)
    sum_ground_truth = np.sum(ground_truth)
    common_RC_score = (common) / (sum_prediction + sum_ground_truth) if (sum_prediction + sum_ground_truth) > 0 else 0.0
    
    return common_RC_score

def get_RC_rate_roi(xai_path, roi_path, save_path, model_name, xai_method, score, dependency_weight):
    
    common_RC = {'ROI 1': [], 'ROI 2': [], 'ROI 3': []}
    
    stack_xai = np.zeros((152, 179, 142))
    print(f"Score{score}")        
    for fold in os.listdir(xai_path):
        fold_path = os.path.join(xai_path, fold)
        xai_fold_path = os.path.join(fold_path, f'saved_xai_methods/{xai_method}/Score{score}/{xai_method}_group_level_positive.nii.gz')  
        xai_array = sitk.GetArrayFromImage(sitk.ReadImage(xai_fold_path))
        stack_xai += xai_array
    
    # Save stack XAI
    updated_xai = stack_xai.astype(np.double)
    updated_xai = sitk.GetImageFromArray(updated_xai)
    sitk.WriteImage(updated_xai, os.path.join(fold_path, xai_method+f"_test_Score{score}.nii.gz"))
    
     # Get ROI masks
    roi_paths = [os.path.join(roi_path, f"roi_image_{i}.nii.gz") for i in range(1, 4)]
    roi_masks = [sitk.GetArrayFromImage(sitk.ReadImage(roi_path)) for roi_path in roi_paths]
            
    # Common RC scores
    for idx, roi_mask in enumerate(roi_masks):
        common_RC_score = calculate_RC_score(stack_xai, roi_mask)  # Use binary masks for the calculation
        common_RC[f'ROI {idx + 1}'].append(np.round(common_RC_score, 3))
        
    sum_common_RC = sum([sum(values) for values in common_RC.values()])
    rate_common_RC = {key: [score / sum_common_RC for score in values] for key, values in common_RC.items()}

    for roi, values in rate_common_RC.items():
        print(f"Common RC rate for {roi}: {values[0]}")
    print("-----")
    
    # Save
    data = {
        "analysis": "RC Rate inside ROI",
        "RC Rate": rate_common_RC
    }
    
    path_dir = os.path.join(save_path, "RC-rate", model_name+xai_method+f"/Score {score}"+"/")
    path = os.path.join(path_dir, f"{dependency_weight}_rc_rate.json")

    if not os.path.exists(path_dir):
        os.makedirs(path_dir, exist_ok=True)
#         print(f"Folder '{path_dir}' created successfully!")
        
    with open(path, 'w') as f:
        path
        json.dump(data, f)  # , indent=2
        # print(f"False positive rates are saved in {path}")
        
    return data
    

""" Analysis for single-multi output and noise experiments """

def determine_volume(dataloader):
    iterator = iter(dataloader)
    volume, scores, ids = [], [], []
    for i in range(len(dataloader)):
        image, score, id = next(iterator)
        volume.append(torch.sum(image[0]))
        scores.append(score[0])
        ids.append(id[0])
        if len(image) > 1:
            volume.append(torch.sum(image[1]))
            scores.append(score[1])
            ids.append(id[1])
    return volume, scores, ids


def determine_ols_volume(dataloader, true_values, predictions):
    # Perform ordinary least squares analysis based on volume alone
    volumes, scores, ids = determine_volume(dataloader)

    print(f"This is a check, these values should be the same: {len(scores)}, {len(true_values)}")
    for i, value in enumerate(true_values):
        print(f"id {i}: {value}, {scores[i]}")

    x = np.column_stack((predictions, volumes))
    X = sm.add_constant(x)
    y = true_values

    model = sm.OLS(y, X)
    results = model.fit()
    print(results.summary())
    print(f"R2: {results.rsquared}")
    print(f"Resid_Pearson: {results.resid_pearson}")


def count_voxels(image, indices, flat=True):
    # Return the number of non-zero voxels at locations corresponding to the indices within the image
    non_zero_counter = 0
    if flat:
        for i in indices:  # f[0]:
            if image[i] != 0:
                non_zero_counter += 1
    else:
        for i in range(indices.shape[1]):
            if image[indices[0][i], indices[1][i], indices[2][i]]:
                non_zero_counter += 1
    return non_zero_counter


def get_pos_rate(thresholded_img, indices, nr_roi_voxels):
    # Calculate the true or false positive rate 
    # TPR: proportion suprathreshold in ROIs
    # FPR: proportion suprathreshold outside ROIs

    if len(thresholded_img) > 152:
        supra = count_voxels(thresholded_img, indices)
    else:
        supra = count_voxels(thresholded_img, indices, flat=False)

    pos_rate = supra / len(indices)  # nr_roi_voxels #
    return pos_rate


def get_p_and_r(thresholded_img, roi_indices, bg_indices, i=0, method="SVR"):
    # Calculate precision and recall
    # TP: number of suprathreshold voxels in ROIs
    # FP: number of suprathreshold voxels outside ROIs
    # FN: number of subthreshold voxels in ROIs

    if thresholded_img.shape[0] > 152:
        tp = count_voxels(thresholded_img, roi_indices)
        fp = count_voxels(thresholded_img, bg_indices)
    else:
        tp = count_voxels(thresholded_img, roi_indices, flat=False) # It was False
        fp = count_voxels(thresholded_img, bg_indices, flat=False)

    if method == "NN":
        fn = len(roi_indices[0]) - tp
    else:
        # fn = len(roi_indices) - tp # Ryanne
        fn = np.shape(roi_indices)[1] - tp
    if tp == 0:
        precision = 0
        recall = 0
    else:
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)

    if i == 0 or i % 100 == 0:
        print(f'True positives: {tp} at {i}/{1000}')
        print(f'False positives: {fp} at {i}/{1000}')
    return precision, recall


def calculate_iou(thresholded, roi_mask):
    intersection = np.logical_and(thresholded, roi_mask)
    union = np.logical_or(thresholded, roi_mask)
    iou = np.sum(intersection) / np.sum(union)
    return iou


def get_torf_rates(image, roi_path, roi_folder, filename, save_path, model_name, analysis, output="one", score=1, 
                   xai='gs', dependency="", method="SVR", mask_indices=None, tpr=True):
    
    # Threshold (descending order) from max to min values in map
    precision, recall, pos, iou, dice = [], [], [], [], []

    if method == "NN":
        image = sitk.GetArrayFromImage(sitk.ReadImage(image))

    thresholds = np.unique(image)  # Determine unique values in image that can be used as a threshold (returns a sorted list)
    thresholds[::-1].sort()
    sampled_thresholds = np.linspace(max(thresholds), min(thresholds), num=1000)
    
    # Get ROI mask indices
    roi_mask = sitk.GetArrayFromImage(sitk.ReadImage(roi_path))

    if method == "VLSM" or method == "GLM":
        roi_reshape = roi_mask.reshape(-1)
        roi_mask = [roi_reshape[i] for i in mask_indices[0]]
    else:
        roi_mask = roi_mask.reshape(np.shape(image))

    # if method == "SVR":
    #     roi_indices = np.array(np.nonzero(roi_mask))[0]
    # else:
    roi_indices = np.array(np.nonzero(roi_mask))
    print(f'ROI indices: {np.shape(roi_indices)}')

    if analysis == "roc" and tpr:
        rate = "tpr"

    else:
        bg_indices = []  # Background
        if method == "VLSM" or method == "GLM":
            for i, value in enumerate(roi_mask):
                if value == 0: bg_indices.append(i)
        # elif method == "NN":
        else:
            bg_indices = np.array(np.where(roi_mask == 0))
        # else:
        #     bg_indices = np.array(np.where(roi_mask == 0))[0]

    print(f'BG indices: {np.shape(bg_indices)}')

    for i, t in enumerate(tqdm(sampled_thresholds)):
        thresholded = image >= t  # Threshold image

        if analysis == "roc":
            # Get positive or false positive rate
            # Determine true positive rate (proportion suprathreshold in ROI)
            if tpr:
                pos.append(get_pos_rate(thresholded, roi_indices, len(roi_indices[0])))
            # Determine false positive rate (proportion suprathreshold outside ROI)
            else:
                rate = "fpr"
                pos.append(get_pos_rate(thresholded, bg_indices, len(roi_indices[0])))

        elif analysis == "p_and_r":
            pre, rec = get_p_and_r(thresholded, roi_indices, bg_indices, i, method)
            precision.append(pre)
            recall.append(rec)

        elif analysis == "iou":
            iou_score = calculate_iou(thresholded, roi_mask)
            iou.append(iou_score)

        elif analysis == "dice":
            dice_score = calculate_dice(thresholded, roi_mask)
            dice.append(dice_score)

    if analysis == "roc":  # we don't use this because it's not sensitive to the little ROIs
        # Save positive or false positive rate
        data = {
            "method": method,
            "analysis": "ROC",
            "volume_correction": True,
            'rate': pos,
            "thresholds": sampled_thresholds.tolist()
        }
        if output == "one":
            path_dir = os.path.join(save_path, "ROC", model_name, xai+'/')
        elif output == "multi":
            path_dir = os.path.join(save_path, "ROC", model_name, f'Score{score}', xai+'/')

        path = os.path.join(path_dir, f"{filename}.json")
        if not os.path.exists(path_dir):
            os.makedirs(path_dir, exist_ok=True)
            print(f"Folder '{path_dir}' created successfully!")
            
    elif analysis == "p_and_r":
        # Save precision and recall
        data = {
            "method": method,
            "analysis": "precision and recall",
            "volume_correction": False,
            "precision": precision,
            "recall": recall,
            "thresholds": sampled_thresholds.tolist()
        }
        
        if output == "one":
            path_dir = os.path.join(save_path, "PR-curve", model_name, xai+'/')
        elif output == "multi":
            path_dir = os.path.join(save_path, "PR-curve", dependency, model_name, f'Score{score}', xai+'/')

        path = os.path.join(path_dir, f"{filename}.json")
        if not os.path.exists(path_dir):
            os.makedirs(path_dir, exist_ok=True)
            print(f"Folder '{path_dir}' created successfully!")

    elif analysis == "iou":
        # Save precision and recall
        data = {
            "method": method,
            "analysis": "IoU",
            "volume_correction": False,
            "iou": iou,
            "threshold": sampled_thresholds.tolist()
        }
        
        if output == "one":
            path_dir = os.path.join(save_path, "IoU", model_name, xai+'/')
        elif output == "multi":
            path_dir = os.path.join(save_path, "IoU", model_name, f'/Score{score}', xai+'/')

        path = os.path.join(path_dir, f"{filename}.json")
        if not os.path.exists(path_dir):
            os.makedirs(path_dir, exist_ok=True)
            print(f"Folder '{path_dir}' created successfully!")
            

    elif analysis == "dice":
        # Save precision and recall
        data = {
            "method": method,
            "analysis": "dice score",
            "volume_correction": False,
            "dice": dice,
            "threshold": sampled_thresholds.tolist()
        }

        if output == "one":
            path_dir = os.path.join(save_path, "DiceScore", model_name, xai+'/')
        elif output == "multi":
            path_dir = os.path.join(save_path, "DiceScore", model_name, f'/Score{score}', xai+'/')

        path = os.path.join(path_dir, f"{filename}.json")
        if not os.path.exists(path_dir):
            os.makedirs(path_dir, exist_ok=True)
            print(f"Folder '{path_dir}' created successfully!")

    with open(path, 'w') as f:
        path
        json.dump(data, f)  # , indent=2
        if tpr:
            print(f"True positive rates are saved in {path}")
        else:
            print(f"False positive rates are saved in {path}")

    return data


def roc_curve(true_pos, false_pos):
    plt.plot(false_pos, true_pos)
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC analysis')
    plt.show()


def get_roc_curve(filename, roi_folder):
    # The ROC curve are obtained by plotting the true positive rates versus the false positive rates
    # for all possible thresholds (within map values) in beta- or t-maps

    # Opening JSON files containing true and false positive rates
    # path_base = Path(r"U:\ryanne\Z_Scratch\Analyses\LSM\ROC")
    path = os.path.join(os.getcwd(), "experiments/one_output_3D/analysis/ROC-curve/"+model_name+'/')
    f_tpr = open(path + filename + "_tpr" + ".json")
    f_fpr = open(path + filename + "_fpr" + ".json")

    # Turn json files into dictionary
    data_tpr = json.load(f_tpr)
    data_fpr = json.load(f_fpr)

    # Access true and false positive rates
    tpr = data_tpr["tpr"]
    fpr = data_fpr["fpr"]
    print(f"true positive rate shape {np.shape(tpr)}")
    print(f"false positive rate shape {np.shape(fpr)}")

    # Closing files
    f_tpr.close()
    f_fpr.close()

    roc_curve(tpr, fpr)
    return tpr, fpr


def pr_curve(precision, recall, method):
   
    plt.plot(recall, precision, label=method)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves')
    plt.legend()
    plt.show()


def get_pr_curve(model_name, save_path, filename, method, output, score, xai, dependency=""):
    # The precision-recall curve is obtained by plotting precision versus recall values
    # for all possible thresholds (within map values) in beta- or t-maps

    # Opening JSON files containing true and false positive rates
    if output == "one":
        path = os.path.join(save_path, "PR-curve", model_name, xai+'/')
    elif output == "multi":
        path = os.path.join(save_path, "PR-curve", dependency, model_name, f'Score{score}', xai+'/')
    
    pr_file = open(path + filename)

    # Turn json files into dictionary
    data = json.load(pr_file)

    # Access precision and recall values
    precision = data["precision"]
    recall = data["recall"]
    pr_file.close()  # Closing files
    
    auc = metrics.auc(recall, precision)
    print(f'Area under the PR curve: {"%.5f" % auc} for {method}')

    # method = method+f' (AUC: {auc})'
    # pr_curve(precision, recall, method)
    
    return precision, recall, auc


def iou_curve(iou, thresholds, method):
    plt.plot(thresholds, iou)
    plt.xlabel('Threshold')
    plt.ylabel('IoU')
    plt.title(f'IoU Curve: {method}')
    plt.show()


def get_iou_curve(filename, method, output):
    
    # Opening JSON files containing true and false positive rates
    if output == "one":
        path = os.path.join(save_path, "IoU", model_name, xai+'/')
    elif output == "multi":
        path = os.path.join(save_path, "IoU", dependency, model_name, f'Score{score}', xai+'/')

    pr_file = open(path + filename)

    # Turn json files into dictionary
    data = json.load(pr_file)

    # Access precision and recall values
    iou = data["iou"]
    thresholds = data["threshold"]
    iou_file.close()  # Closing files

    iou_curve(iou, thresholds, method)
    auc = metrics.auc(thresholds, iou)
    print(f'Area under the IoU curve: {"%.5f" % auc} for {method}')

    return iou


def dice_curve(dice, thresholds, method):
    plt.plot(thresholds, dice)
    plt.xlabel('Threshold')
    plt.ylabel('dice')
    plt.title(f'Dice Score Curve: {method}')
    plt.show()


def get_dice_curve(filename, method, output):
    # Opening JSON files containing true and false positive rates
    if output == "one":
        path = os.path.join(save_path, "Dice", model_name, xai+'/')
    elif output == "multi":
        path = os.path.join(save_path, "Dice", dependency, model_name, f'Score{score}', xai+'/')

    pr_file = open(path + filename)

    # Turn json files into dictionary
    data = json.load(pr_file)
    # Access precision and recall values
    dice = data["dice"]
    thresholds = data["threshold"]
    dice_file.close()  # Closing files

    dice_curve(dice, thresholds, method)
    auc = metrics.auc(thresholds, dice)
    print(f'Area under the Dice Score curve: {"%.5f" % auc} for {method}')
    
    return dice


""" Analysis for comparing different XAI methods """

def map_wise_correlation(model_name, xai_path):
    # load att maos
    files = os.listdir(xai_path)
    attribution_maps = [file_name for file_name in files if file_name.endswith('.nii.gz')]

    images_by_score = {}
    for attmap in attribution_maps:
        img = sitk.ReadImage(os.path.join(xai_path, attmap))
        parts = attmap.split('_')
        score = parts[-1].split('.')[0]
        if score not in images_by_score:
            images_by_score[score] = []
        # save att maps by score
        images_by_score[score].append(img)

    map_wise_correlation = {
        "analysis": "Attribution map-wise Correlation Coefficient",
        "methods": "Occlusion vs Gradient Shap"
        }

    # Calculate and print correlation coefficients within each score group
    for score, image_list in images_by_score.items():
        print(f"Score: {score}")
        att_map1 = sitk.GetArrayFromImage(image_list[0])
        att_map2 = sitk.GetArrayFromImage(image_list[1])

        correlation_coefficient = np.corrcoef(att_map1.flatten(), att_map2.flatten())[0, 1]  
        # correlation_coefficient = scipy.stats.pearsonr(att_map1.flatten(), att_map2.flatten())
        # correlation_coefficient = scipy.signal.correlate(att_map1, att_map2)

        if score not in map_wise_correlation:
            map_wise_correlation[score] = []
        map_wise_correlation[score].append(correlation_coefficient)
        print(f"Score {score[-1]}: Correlation Coefficient: {correlation_coefficient}")


    path = os.path.join(xai_path, f"XAIs_comparison.json")
    with open(path, 'w') as f:
        json.dump(map_wise_correlation, f)
        
       
    
""" Used in analysis for models when parameter tunning """

def correlation_predictions(folder_path):
    all_predictions = []
    all_true_labels = []

    # Iterate over each file in the folder
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.json'):
            file_path = os.path.join(folder_path, file_name)

            # Read the JSON file
            with open(file_path, 'r') as json_file:
                data = json.load(json_file)

            # Append the prediction and true labels to the lists
            all_predictions.extend(data["prediction"])
            all_true_labels.extend(data["true"])

    return all_predictions, all_true_labels

def plot(predictions, true_labels, score):
    y_pred = torch.tensor(predictions)
    y_true = torch.tensor(true_labels)

    # Convert tensors to numpy arrays for plotting
    y_true_np = y_true.detach().cpu().numpy()
    y_pred_np = y_pred.detach().cpu().numpy()

    # Calculate the correlation coefficient
    # correlation = np.corrcoef(y_true_np, y_pred_np)[0, 1]
    correlation, _ = pearsonr(y_true_np, y_pred_np)

    # Create the scatter plot
    plt.scatter(y_true_np, y_pred_np, label='True vs. Predicted', color='b', marker='o')

    # Plot the line of correlation
    plt.plot([y_true_np.min(), y_true_np.max()], [y_true_np.min(), y_true_np.max()], color='r',
             label=f'Correlation: {correlation:.2f}')

    # Add labels and legend
    plt.xlabel(f'Sccore {score}')
    plt.ylabel('Predicted Values')
    plt.title("Scatter plot of predicted values ")
    plt.legend()

    # Show the plot
    plt.show()
    
def correlation_scatter_plot(model, test_loader, mode="multi"):
    # Create an empty list to store predictions
    pred, pred2, label, label2 = [], [], [], []
    model.eval()

    with torch.no_grad():
        if mode == 'one':
            # Make simulation score predictions
            for images, score in test_loader:
                images, score = images.cuda().float(), score.cuda().float()
                outputs = model(images)
                pred.extend(outputs.flatten().cpu().tolist())
                label.extend(score.flatten().cpu().tolist())
            plot(pred, label, "1")

        elif mode == 'multi':
            # Make simulation score predictions
            for images, score1, score2 in test_loader:
                images, score1, score2 = images.cuda().float(), score1.cuda().float(), score2.cuda().float()
                outputs = model(images)
                o1, o2 = torch.unbind(outputs, dim=1)
                pred.extend(o1.flatten().cpu().tolist())
                pred2.extend(o2.flatten().cpu().tolist())

                label.extend(score1.flatten().cpu().tolist())
                label2.extend(score2.flatten().cpu().tolist())

            plot(pred, label, "1")
            plt.title("Score 2 Scatter plot of predicted values ")
            plot(pred2, label2, "2")

        else:
            raise ValueError(f"Invalid argument {mode}")