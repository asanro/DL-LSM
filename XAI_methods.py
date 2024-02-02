import torch.nn as nn
import numpy as np
import SimpleITK as sitk
import torch
import warnings
import os
import random
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from seed import seed_everything
from captum.attr import Occlusion, IntegratedGradients, DeepLiftShap, GradientShap, NoiseTunnel, Saliency, DeepLift, \
    InputXGradient, GuidedBackprop, GuidedGradCam
from captum.attr import visualization as viz
import analysis

# seeds
# np.random.seed(0) # seed for NumPy
# # random.seed(0) # for python
# torch.manual_seed(0) # seed for PyTorch
# torch.cuda.manual_seed(0) # seed for cuda
# torch.cuda.manual_seed_all(0) # if you are using multi-GPU.
print("Global seed", seed_everything(0))


""""  Standard functions used for explanations """


def load_model(model, model_path, device):
    net = model.load_from_checkpoint(model_path)
    return net.to(device).eval()


def prep_visualisation(attribution, group=False):
    attribution = attribution.cpu().detach().numpy()
    return np.transpose(attribution, (1, 2, 3, 0))  # Transpose attributions for visualisation purposes


def _normalize_scale(attribution, scale_factor):
    # assert scale_factor != 0, "Cannot normalize by scale factor = 0"
    if scale_factor == 0:
        scale_factor = 1e-10
    if abs(scale_factor) < 1e-5:
        warnings.warn(
            "Attempting to normalize by value approximately 0, visualized results"
            "may be misleading. This likely means that attribution values are all"
            "close to 0."
        )
    attr_norm = attribution / scale_factor
    return np.clip(attr_norm, -1, 1)


def save_attribution(attribution, path_base, pt_id, pt_label, method, group=False, sign="all", layer="conv1"):
    
    if group:
        if method == "ggc":
            path = path_base + "/" + f"{method}_group_level_" + sign + "_"+ layer + ".nii.gz"
        else:
            path = path_base + "/" + f"{method}_group_level_" + sign + ".nii.gz"
        # attribution = attribution.detach().cpu().numpy() 
    else:
        path = path_base + "/" + f"{method}_pt_" + str(pt_id) + "_label_" + str(pt_label) + "_" + ".nii.gz"
        attribution = attribution.detach().cpu().numpy() # .numpy() .detach().cpu().numpy()
    
    # print(f'Save path: {path}')
    attribute_image = sitk.GetImageFromArray(attribution.squeeze())
    sitk.WriteImage(attribute_image, path)
    return


""""  Determine attributions """

def determine_attribution_cv(net, method, device, device_id, dataloader, nr_pts, save=False, save_path=None, 
                             value=0, n_classes=1, group_level=True, sign="positive"):
    
    if len(device_id) == 2: # if dataparallel
        print("Dataparallel ON")
        net = net.module
    # Select image from dataloader to determine attributions
    dataiter = iter(dataloader)
    group_attributions = []
    for i in tqdm(range(nr_pts)):
        torch.cuda.empty_cache()
        if n_classes == 1:
            images, pt_label, pt_id = next(dataiter)
            images = images.to(device)
        elif n_classes == 3:
            images, pt_label, pt_label2, pt_label3, pt_id = next(dataiter)
            images = images.to(device)
            
        if method == "ig":
            integrated_gradients = IntegratedGradients(net)
            zeros = torch.zeros(np.shape(images)).to(device)  # Create baseline image
            if len(pt_label) == 1:
                attributions_1 = integrated_gradients.attribute(images, baselines=zeros,
                                                                n_steps=10, target=value)  # Calculate attributions
            else:
                attributions_1, attributions_2 = integrated_gradients.attribute(images, baselines=zeros, n_steps=10,
                                                                target=value) 
        elif method == "gs":
            gradient_shap = GradientShap(net)
            zeros = torch.zeros(np.shape(images)).to(device)  # Create baseline image
            if len(pt_label) == 1:
                attributions_1 = gradient_shap.attribute(images, n_samples=10,
                                                         stdevs=0.01, baselines=zeros, target=value)  
            else:
                attributions_1, attributions_2 = gradient_shap.attribute(images, n_samples=30, stdevs=0.01,
                                                                         baselines=zeros, target=value)
        elif method == "occlusion":
            occlusion = Occlusion(net)
            if len(pt_label) == 1:
                attributions_1 = occlusion.attribute(images, sliding_window_shapes=(1, 5, 5, 5), strides=(1, 3, 3, 3),
                                                     perturbations_per_eval=10, show_progress=True, target=value)
            else:
                attributions_1, attributions_2 = occlusion.attribute(images, sliding_window_shapes=(1, 5, 5, 5), strides=(1, 3, 3, 3), 
                                                                     perturbations_per_eval=10, show_progress=True, target=value) 
        elif method == "ggc":
            layer = "conv1"
            guided_grad_cam = GuidedGradCam(net, net.conv1)
            images_grad = images.clone().detach().requires_grad_(True).to(device)
            if len(pt_label) == 1:
                attributions_1 = guided_grad_cam.attribute(images_grad, interpolate_mode='trilinear', target=value)  
            else:
                attributions_1, attributions_2 = guided_grad_cam.attribute(images_grad, interpolate_mode='nearest', target=value)
          
        # Uncomment  to save per patient  
        if save: 
            if len(pt_label) == 1:
                save_attribution(attributions_1, save_path, pt_id[0].item(), "{:.3f}".format(pt_label[0].item()), method)
            else:
                save_attribution(attributions_1, save_path, pt_id[0].item(), "{:.3f}".format(pt_label[0].item()), method)
                save_attribution(attributions_2, save_path, pt_id[1].item(), "{:.3f}".format(pt_label[1].item()), method)
            torch.cuda.empty_cache()
            
        if group_level:
            if sign == "absolute_value":
                if len(pt_label) == 1:
                    attributions_1 = torch.abs(attributions_1)
                else:
                    attributions_1 = torch.abs(attributions_1)
                    attributions_2 = torch.abs(attributions_2)
            elif sign == "positive":
                if len(pt_label) == 1:
                    attributions_1 = attributions_1.clone().detach().to(device)
                    attributions_1[attributions_1 < 0] = 0
                else:
                    attributions_1 = attributions_1.clone().detach().to(device)
                    attributions_2 = attributions_2.clone().detach().to(device)
                    attributions_1[attributions_1 < 0] = 0
                    attributions_2[attributions_2 < 0] = 0
#                     attributions_1[attributions_1 < 0].zero_()
#                     attributions_2[attributions_2 < 0].zero_()
  
            elif sign == "negative":
                if len(pt_label) == 1:
                    attributions_1[attributions_1 > 0] = 0
                else:
                    attributions_1[attributions_1 > 0] = 0
                    attributions_2[attributions_2 > 0] = 0
            else:
                print('No valid sign provided!')
                return
            
            if i == 0:
                group_attributions = attributions_1
            else:
                if attributions_1.dim() == 5:  # (nr_pts-1):
                    attributions_1 = attributions_1[0].clone().detach().to(device)
                group_attributions = torch.cat((group_attributions, attributions_1))
            
            if len(pt_label) == 2:
                group_attributions = torch.cat((group_attributions, attributions_2))
         
    if group_level:
        group = group_attributions.detach().cpu().numpy() # detach()
        sum_group = np.sum(group, axis=0)
        sum_group = sum_group[None, :, :, :]
        
        if save:
            if method=="ggc":
                save_attribution(sum_group, save_path, None, None, method, group=True, sign=sign, layer=layer)
            else:
                save_attribution(sum_group, save_path, None, None, method, group=True, sign=sign)
           
        
"""" Evaluate attributions """

def evaluate_XAI(xai_path, model_name, output, save_path, make_analysis=True, xai="ggc", path_noise=None, noise=True, score=0, name="CNN", dependency="", layer=''):
    
    roi_folder = os.path.join(os.getcwd(), "images/")
    if score==1:
        roi_path = os.path.join(os.getcwd(), "images/roi_image_1.nii.gz")
    elif score==2:
        roi_path = os.path.join(os.getcwd(), "images/roi_image_2.nii.gz")
    elif score==3:
        roi_path = os.path.join(os.getcwd(), "images/roi_image_3.nii.gz")
    else:
        roi_path = os.path.join(os.getcwd(), "images/roi_image_0.nii.gz")
        
    # model without noise
    print('Computing evaluation for non-noise saliency map...')
    stack_xai = np.zeros((152, 179, 142))
    print(xai_path)
    for fold in os.listdir(xai_path):
        fold_path = os.path.join(xai_path, fold)
        if output == "one":
            xai_fold_path = os.path.join(fold_path, 'saved_xai_methods/'+xai+'/'+xai+'_group_level_positive'+layer+'.nii.gz')
        elif output == "multi":
            xai_fold_path = os.path.join(fold_path, 'saved_xai_methods/'+xai+f'/Score{score}/'+xai+'_group_level_positive'+layer+'.nii.gz')
            
        xai_array = sitk.GetArrayFromImage(sitk.ReadImage(xai_fold_path))
        stack_xai += xai_array
    
    # Save stack XAI
    updated_xai = stack_xai.astype(np.double)
    updated_xai = sitk.GetImageFromArray(updated_xai)
    if output == "one":
        sitk.WriteImage(updated_xai, os.path.join(fold_path, xai+"_test.nii.gz"))
    elif output == "multi":
        sitk.WriteImage(updated_xai, os.path.join(fold_path, xai+layer+f"_test_Score{score}.nii.gz"))
    filename = 'no_noise_PR-curve'+layer
    if make_analysis:
        data = analysis.get_torf_rates(stack_xai, roi_path, roi_folder, filename, save_path, model_name, 'p_and_r', 
                                       output, score, xai, dependency)
    method = name+' without noise'
    
    precision, recall, auc = analysis.get_pr_curve(model_name, save_path, filename + '.json', method, output, score, xai, dependency)
    
    save_name = 'test_predictions.png'
    
    if noise:
        save_name = 'test_predictions_noise.png'
        print('\nComputing evaluation noisy saliency maps...')
        # one-output with noise
        noise = ['0.1', '0.2', '0.3', '0.4', '0.5']
        p, r, a = [], [], []
        for weight in noise:
            stack_xai = np.zeros((152, 179, 142))
            noise_path = os.path.join(path_noise, f'{weight}')
        
            for fold in os.listdir(noise_path):
                fold_path = os.path.join(noise_path, fold)
                if output == "one":
                    xai_fold_path = os.path.join(fold_path, 'saved_xai_methods/'+xai+'/'+xai+'_group_level_positive'+layer+'.nii.gz')
                elif output == "multi":
                    xai_fold_path = os.path.join(fold_path, 'saved_xai_methods/'+xai+f'/Score{score}/'+xai+'_group_level_positive'+layer+'.nii.gz')
                xai_array = sitk.GetArrayFromImage(sitk.ReadImage(xai_fold_path))
                stack_xai += xai_array
            
            # Save stack XAI
            updated_xai = stack_xai.astype(np.double)
            updated_xai = sitk.GetImageFromArray(updated_xai)
            if output == "one":
                sitk.WriteImage(updated_xai, os.path.join(fold_path, xai+"_test.nii.gz"))
            elif output == "multi":
                sitk.WriteImage(updated_xai, os.path.join(fold_path, xai+f"_test_Score{score}.nii.gz"))
            
            filename = f'{weight}_noise_PR-curve'
            if make_analysis:
                data = analysis.get_torf_rates(stack_xai, roi_path, roi_folder, filename, save_path, model_name, analysis='p_and_r',
                                              output=output, score=score, xai=xai)
            method1 = name+f' with {int(float(weight)*100)}% noise'
            precision1, recall1, auc1 = analysis.get_pr_curve(model_name, save_path, filename + '.json', method1, output, score, xai)
            p.append(precision1)
            r.append(recall1)
            a.append(auc1)
        
    # Plot results
    plt.plot(recall, precision, label=name+f' without noise (AUC: {round(auc,3)})')
    
    if noise:
        for i, (weight) in enumerate(noise):
            plt.plot(r[i], p[i], label=name+f' with {int(float(weight) * 100)}% noise (AUC: {round(a[i], 3)})')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    if output == "one":
        plt.title(f'Score {score+1} - Precision-Recall Curves')
    elif output == "multi":
        plt.title(f'Score {score} - Precision-Recall Curves')
    if noise:
        plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.4), ncol=2)
    else:
        plt.legend(loc='lower center')
    # Save plot
    if output == "one":
        plot_path = os.path.join(save_path, 'PR-curve', model_name, xai, save_name)
    elif output == "multi":
        plot_path = os.path.join(save_path, 'PR-curve', model_name, f'Score{score}', xai, save_name)

    plt.savefig(plot_path, bbox_inches='tight')
    plt.show()
    plt.close()