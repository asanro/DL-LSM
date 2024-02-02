import os
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, StratifiedKFold
from torch.utils.data import Dataset
import architectures
from architectures import BasicBlock, BasicBlock_, BasicBlock_7
import numpy as np
import torch
import wandb
import random
import torch.nn as nn
import XAI_methods as XAI
from seed import seed_everything
import torch.nn.init as init
import json
import sklearn
import scipy
from sklearn.metrics import r2_score
# seeds
# np.random.seed(0) # seed for NumPy
# random.seed(0) # for python
torch.manual_seed(0) # seed for PyTorch
torch.cuda.manual_seed(0)
# torch.cuda.manual_seed_all(0) # if you are using multi-GPU.
print("Global seed", seed_everything(0))


""" Helper functions """

def pearson_correlation(data, n_classes, test=False):
    if n_classes == 1:
        all_predictions = [float(item) for batch_info in data for item in batch_info.get('prediction', [])]
        all_true_labels = [float(item) for batch_info in data for item in batch_info.get('true', [])]

        correlations = np.corrcoef(all_predictions, all_true_labels)[0, 1]

    elif n_classes == 3:
        num_columns = len(data[0]['prediction'])
        correlations = []
        all_predictions = {}
        all_true_labels = {}

        for i in range(num_columns):
            all_predictions_list = []
            all_true_labels_list = []
            for batch_info in data:
                all_predictions_list.extend((batch_info['prediction'][i]).tolist())
                all_true_labels_list.extend(batch_info['true'][i].tolist())

            correlation = np.corrcoef(all_predictions_list, all_true_labels_list)[0, 1]
            correlations.append(correlation) 
            if test:
                all_predictions[f'Score{i + 1}'] = all_predictions_list
                all_true_labels[f'Score{i + 1}'] = all_true_labels_list
            
    return correlations, all_predictions, all_true_labels



def reset_weights(m):
    for layer in m.children():
        if hasattr(layer, 'reset_parameters'):
            print(f'Reset trainable parameters of layer = {layer}')
            layer.reset_parameters()

            
def save_predictions(fold_path, loss_test, corr_test, all_predictions, all_true_labels):
    if len(all_predictions) == 3:
        # Create a dictionary with dynamic keys
        fold_info = {
            "test_loss": {},
            'test_correlation': {},
            "predictions": {},
            "true_labels": {}
        }

        num_scores = len(all_predictions)
        for i in range(num_scores):
            key = f'Score{i + 1}'
            fold_info["test_loss"][key] = loss_test[i].item()
            fold_info["test_correlation"][key] = corr_test[i].item()
            fold_info["predictions"][key] = all_predictions[key]
            fold_info["true_labels"][key] = all_true_labels[key]
            
        # Save the dictionary as a JSON file
        json_path = os.path.join(fold_path, 'fold_testpred.json')
        with open(json_path, 'w', encoding='utf-8') as json_file:
            json.dump(fold_info, json_file)
        print(f"Fold test predictions saved to {json_path}")
    else:
        fold_info = {
            "test loss": loss_test,
            'test_correlation': corr_test,
            "prediction": all_predictions,
            "true": all_true_labels
        }
        # Save the dictionary as a JSON file
        json_path = os.path.join(fold_path, 'fold_testpred.json')
        with open(json_path, 'w') as json_file:
            json.dump(fold_info, json_file)
        print(f"Fold test predictions saved to {json_path}")
        

def show_test_prediction_simulation_new(all_predictions, all_true_labels, n_classes, save_path=None, full_test=False, m_name="CNN"):
    
    if n_classes == 1:
        x = [0, 0.5, 1]
        y = [0, 0.5, 1]
        title = f'{m_name} single-output model'
        plt.plot(all_true_labels, all_predictions, 'o', markersize=2, label='Score 1')
        plt.plot(x, y, '--')
        plt.xlabel("Ground Truth score")
        plt.ylabel("Predicted score")
        plt.title(title)
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.legend()
          
        # Save plot
        if full_test:
            plot_path = os.path.join(save_path, 'test_predictions.png')
            correlations = scipy.stats.pearsonr(all_true_labels, all_predictions)
            r2 = r2_score(all_true_labels, all_predictions)
            
            test_info = {
                "Coefficient of determination (r_squared)": r2,
                'Pearson Correlation R': correlations[0],
                'p value': correlations[1],
                'Pearson R squared:': correlations[0]**2
            }
            # Save the dictionary as a JSON file
            json_path = os.path.join(save_path, 'testpred.json')
            
            with open(json_path, 'w') as json_file:
                json.dump(test_info, json_file)
            print(f"Fold test predictions saved to {json_path}")
            
        else:
            plot_path = os.path.join(save_path, 'fold_test_predictions.png')
        plt.savefig(plot_path)
            
        plt.show()
        plt.close()
             
    elif n_classes == 3:
        x = [0, 0.5, 1]
        y = [0, 0.5, 1]
        title =  f'{m_name} multi-output model'
        plt.plot(all_true_labels["Score1"], all_predictions["Score1"], 'o', markersize=2, label='Score 1')
        plt.plot(all_true_labels["Score2"], all_predictions["Score2"], 'o', markersize=2, label='Score 2')
        plt.plot(all_true_labels["Score3"], all_predictions["Score3"], 'o', markersize=2, label='Score 3')
        plt.plot(x, y, '--')
        plt.xlabel("Ground Truth score")
        plt.ylabel("Predicted score")
        plt.title(title)
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.legend()
  
        # Save plot
        if full_test:
            plot_path = os.path.join(save_path, 'test_predictions.png')
            
            correlations1 = scipy.stats.pearsonr(all_true_labels["Score1"], all_predictions["Score1"])
            correlations2 = scipy.stats.pearsonr(all_true_labels["Score2"], all_predictions["Score2"])
            correlations3 = scipy.stats.pearsonr(all_true_labels["Score3"], all_predictions["Score3"])
            
            r2_1 = r2_score(all_true_labels["Score1"], all_predictions["Score1"])
            r2_2 = r2_score(all_true_labels["Score2"], all_predictions["Score2"])
            r2_3 = r2_score(all_true_labels["Score3"], all_predictions["Score3"])
            
            test_info = {
                "Coefficient of determination (r_squared)": [r2_1, r2_2, r2_3],
                'Pearson Correlation R': [correlations1[0], correlations2[0], correlations3[0]],
                'p value': [correlations1[1], correlations2[1], correlations3[1]],
                'Pearson R squared: ': [correlations1[0]**2, correlations2[0]**2, correlations3[0]**2]
            }
            # Save the dictionary as a JSON file
            json_path = os.path.join(save_path, 'testpred.json')
            
            with open(json_path, 'w') as json_file:
                json.dump(test_info, json_file)
            print(f"Fold test predictions saved to {json_path}")
            
        else:
            plot_path = os.path.join(save_path, 'fold_test_predictions.png')
        plt.savefig(plot_path)
            
        plt.show()
        plt.close()
       
    
""" Cross Validation"""  

class CrossValidation:
    def __init__(self, config=None):
        super().__init__()
        self.criterion = nn.MSELoss()
        self.config = config
        self.lr = self.config['learning_rate']
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device_id = self.config['device_ids']
        print(f"Using {self.device} device")
        if self.config['num_epochs']:
            self.path_model = self.config['path_save_model'] + "/" + 'cv_' + self.config['model'] + '_' + self.config[
                'optimizer'] + '_lr' + str(self.lr) + 'sch_' + self.config['lr_scheduler'] + '_epochs' + self.config['num_epochs']
        else:
            self.path_model = self.config['path_save_model'] + "/" + 'cv_' + self.config['model'] + '_' + self.config[
                'optimizer'] + '_lr' + str(self.lr) + 'sch_' + self.config['lr_scheduler']
        
    def configure_params(self):
        # Define model
        if self.config['model'] == "ResNet2":
            m = architectures.ResNet2(BasicBlock, [1, 1, 1, 1], num_classes=self.config['n_classes'])
        elif self.config['model'] == "ResidualCNN":
            m = architectures.ResidualCNN(BasicBlock_, [2, 2], num_classes=self.config['n_classes'])
            m_name = "Residual CNN"
        elif self.config['model'] == "CNN2ResNet_maxpool":
            m = architectures.CNN2ResNet_maxpool(BasicBlock_, [2, 2], num_classes=self.config['n_classes'])
        elif self.config['model'] == "CNN2":
            m = architectures.CNN2(num_classes=self.config['n_classes'])
            m_name = "CNN"
        elif self.config['model'] == "CNN3":
            m = architectures.CNN3(num_classes=self.config['n_classes'])
        else:
            raise ValueError(f"Unsupported model: {self.config['model']}")
        
        if len(self.device_id) > 1:
            print("Data parallelization")
            m = (nn.DataParallel(m, device_ids=self.device_id)).to(self.device) 
        elif len(self.device_id) == 1:
            m = m.to(self.device) 
                
        # Define optimizer            
        if self.config['optimizer'] == "Adam":
            optim_set = torch.optim.Adam(m.parameters(), lr=self.lr, weight_decay=self.config['weight_decay'])
        elif self.config['optimizer'] == "AdamW":
            optim_set = torch.optim.AdamW(m.parameters(), weight_decay=self.config['weight_decay'])
        elif self.config['optimizer'] == "SGD":
            optim_set = torch.optim.SGD(m.parameters(), lr=self.lr, weight_decay=self.config['weight_decay'])
        else:
            raise ValueError(f"Unsupported optimizer: {self.config['optimizer']}")

        # Define learning rate scheduler
        if self.config['lr_scheduler'] == "None":
            sch = "None"  # Change this line
        elif self.config['lr_scheduler'] == 'StepLR':
            sch = torch.optim.lr_scheduler.StepLR(optim_set, step_size=self.config["step_size"])
        elif self.config['lr_scheduler'] == "ReduceLROnPlateau":
            sch = torch.optim.lr_scheduler.ReduceLROnPlateau(optim_set, mode='min', cooldown=self.config["cooldown"])
        elif self.config['lr_scheduler'] == "cyclic":
            sch = torch.optim.lr_scheduler.CyclicLR(optim_set, base_lr=self.lr, max_lr=self.config['max_lr'],
                                                    step_size_up=5, cycle_momentum=False, mode='triangular2')
        elif self.config['lr_scheduler'] == "multiplicative":
            lmbda = lambda epoch: 0.95
            sch = torch.optim.lr_scheduler.MultiplicativeLR(optim_set, lr_lambda=lmbda)

        return m, optim_set, sch, m_name
    
    def get_foldpath(self, fold):
        if self.config['noise_weight']:
            path = self.path_model + '/' + self.config['noise_weight'] + f"/fold{fold + 1}"
        else:
            path = self.path_model + f"/fold{fold + 1}"

        os.makedirs(path, exist_ok=True)  # Create the directory if it doesn't exist
        return path
                
    def compute_XAI(self, xai_path, target_score, test_loader, model):
        torch.cuda.empty_cache()
        xai_path_s = os.path.join(xai_path, f'Score{target_score+1}')
        os.makedirs(xai_path_s, exist_ok=True)
        XAI.determine_attribution_cv(model, self.config["xai_method"], self.device, self.device_id, test_loader,
                                                 nr_pts=len(test_loader), save=True, save_path=xai_path_s, value=target_score,
                                                 n_classes=self.config['n_classes'])
    def test_analysis(self, m_name):
        if self.config["n_classes"] == 1:
            if self.config['noise_weight']:
                path = os.path.join(self.path_model, self.config['noise_weight'])
            else:
                path = self.path_model
            predictions = []
            true_labels = []
            for fold in sorted(os.listdir(path)):
                fold_path = os.path.join(path, fold)   
                for file_name in os.listdir(fold_path):
                    if file_name == 'fold_testpred.json':
                        file_path = os.path.join(fold_path, file_name)

                        # Read the JSON file
                        with open(file_path, 'r') as json_file:
                            data = json.load(json_file)

                        # Append the prediction and true labels to the lists
                        predictions.extend(data["prediction"])
                        true_labels.extend(data["true"])

            show_test_prediction_simulation_new(predictions, true_labels, self.config['n_classes'], fold_path, True, m_name)

        elif self.config["n_classes"] == 3: # all score 1,2,3 together in predictions
            if self.config['noise_weight']:
                path = os.path.join(self.path_model, self.config['noise_weight'])
            else:
                path = self.path_model
            # Initialize dictionaries to store scores for each class
            class_scores = {"prediction": {f"Score{i+1}": [] for i in range(3)}, "test": {f"Score{i+1}": [] for i in range(3)}}
        
            for fold in sorted(os.listdir(path)):
                fold_path = os.path.join(path, fold)   
                for file_name in os.listdir(fold_path):
                    if file_name == 'fold_testpred.json':
                        file_path = os.path.join(fold_path, file_name)

                        # Read the JSON file
                        with open(file_path, 'r') as json_file:
                            data = json.load(json_file)

                        for i in range(3):
                            score_key = f"Score{i+1}"   
                            class_scores["prediction"][score_key].extend(data["predictions"][score_key])
                            class_scores["test"][score_key].extend(data["true_labels"][score_key])


            show_test_prediction_simulation_new(class_scores["prediction"], class_scores["test"], self.config['n_classes'], fold_path, True, m_name)

        
    def get_batch_data(self, batch):
        if self.config['n_classes'] == 1:
            data, scores, _ = batch
            data, scores = data.to(self.device), scores.to(self.device)
            return data, scores
        elif self.config['n_classes'] == 3:
            data, s1, s2, s3, _ = batch
            data, s1, s2, s3 = data.to(self.device), s1.to(self.device), s2.to(self.device), s3.to(self.device)
            scores = [s1, s2, s3]
            return data, scores
  
    def train(self, loader, optimizer, model):
        model.train()
        all_batch_info = []
        
        for batch in loader:
            data, scores = self.get_batch_data(batch)
 
            output = model(data)
            
            if self.config['n_classes'] == 1:
                if output.shape[0] > 1:
                    output = torch.squeeze(output)
                else:
                    output = output[0]
                loss = self.criterion(output, scores)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                batch_info = {
                    "loss": loss.item(),
                    "prediction": output.detach().cpu().numpy(),
                    "true": scores.detach().cpu().numpy()
                }
                
            elif self.config['n_classes'] == 3:    
                output1, output2, output3 = torch.unbind(output, dim=1)
                loss1 = self.criterion(output1, scores[0])
                loss2 = self.criterion(output2, scores[1])
                loss3 = self.criterion(output3, scores[2])     
                loss = loss1 + loss2 + loss3
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                batch_info = {
                    "loss": loss.item(),
                    "prediction": [output1.detach().cpu().numpy(), output2.detach().cpu().numpy(), output3.detach().cpu().numpy()],
                    "true": [scores[0].detach().cpu().numpy(), scores[1].detach().cpu().numpy(), scores[2].detach().cpu().numpy()]
                }
                
            all_batch_info.append(batch_info)

        return all_batch_info

    def test(self, loader, model):
        model.eval()
        all_batch_info = []
       
        with torch.no_grad():
            
            for batch in loader:
                data, scores = self.get_batch_data(batch)
                output = model(data)
                
                if self.config['n_classes'] == 1:
                    if output.shape[0] > 1:
                        output = torch.squeeze(output)
                    else:
                        output = output[0]
                    loss = self.criterion(output, scores)
                    batch_info = {
                        "loss": loss.item(),
                        "prediction": output.detach().cpu().numpy(),
                        "true": scores.detach().cpu().numpy()
                    }
                elif self.config['n_classes'] == 3:
                    output1, output2, output3 = torch.unbind(output, dim=1)
                    loss1 = self.criterion(output1, scores[0])
                    loss2 = self.criterion(output2, scores[1])
                    loss3 = self.criterion(output3, scores[2])     
                    loss = loss1 + loss2 + loss3
                    batch_info = {
                        "loss": [loss1.item(), loss2.item(),loss3.item()], 
                        "prediction": [output1.detach().cpu().numpy(), output2.detach().cpu().numpy(), output3.detach().cpu().numpy()],
                        "true": [scores[0].detach().cpu().numpy(), scores[1].detach().cpu().numpy(), scores[2].detach().cpu().numpy()]
                    }
                all_batch_info.append(batch_info)
                            
        return all_batch_info 
    
    def cross_validation(self, dataset, k_folds, num_epochs):
        # Initialize wandb
        wandb.init(project=self.config['project_wandb'], entity="a-sanromangaitero", config=self.config)

        wandb.run.name = self.config['model'] + '_' + self.config['optimizer'] + '_lr' + str(self.lr) \
                         + 'sch_' + self.config['lr_scheduler'] + '_epochs' + str(num_epochs)

        # Initialize the k-fold cross validation
        kf = KFold(n_splits=k_folds, shuffle=True, random_state=0)

        # Loop through each fold
        for fold, (train_idx, test_idx) in enumerate(kf.split(dataset)):
            print(f"Fold {fold + 1}")
            print("-------")
            
            # Fold path
            fold_path = self.get_foldpath(fold)
            
            # Create dataloaders
            train_loader = torch.utils.data.DataLoader(dataset, batch_size=4,
                                                       sampler=torch.utils.data.SubsetRandomSampler(train_idx))
            test_loader = torch.utils.data.DataLoader(dataset, batch_size=2,
                                                      sampler=torch.utils.data.SubsetRandomSampler(test_idx))
            
            # Reset model at each epoch
            model, optimizer, scheduler, m_name = self.configure_params()
            
            epoch_train_losses, epoch_train_corr = [], []
            # Train the model on the current fold
            for epoch in range(0, num_epochs):
                print(f"Epoch {epoch+1}...")
                torch.cuda.empty_cache()  # Empty cache after every epoch
                data_train = self.train(train_loader, optimizer, model)
                avg_loss = np.mean([batch_info['loss'] for batch_info in data_train])
                avg_correlation, _, _ = pearson_correlation(data_train, self.config['n_classes'])
                if self.config['n_classes'] == 3:
                    avg_correlation = np.mean(avg_correlation)
                
                # Update scheduler
                if scheduler != "None":
                    if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        scheduler.step(avg_loss)
                    else:
                        scheduler.step()

                epoch_train_losses.append(avg_loss)
                epoch_train_corr.append(avg_correlation)

                wandb.log({"epoch train loss": avg_loss, 'custom_step': epoch})
                wandb.log({"epoch train correlation": avg_correlation, 'custom_step': epoch})

                print(f"train loss: {avg_loss:.4f}")
                print(f"train correlation: {avg_correlation:.4f}")
            
            model.eval()
            with torch.no_grad():
                data_test = self.test(test_loader, model)
                avg_corr_test, all_predictions, all_true_labels = pearson_correlation(data_test, self.config['n_classes'], test=True)

                print(f'\n----Summary fold {fold + 1}:----')

                if self.config['n_classes'] == 1:
                    avg_loss_test = np.mean([batch_info['loss'] for batch_info in data_test])
                    print(f"test loss: {avg_loss_test:.4f}")
                    print(f"test correlation: {avg_corr_test:.4f}\n")
                    wandb.log({"test loss": avg_loss_test, 'custom_step2': fold})
                    wandb.log({"test correlation": avg_corr_test, 'custom_step2': fold})

                    # Save prediction metrics in json file
                    save_predictions(fold_path, avg_loss_test, avg_corr_test, all_predictions, all_true_labels)
                                        
                elif self.config['n_classes'] == 3:
                    avg_loss_test = np.mean(np.array([batch_info['loss'] for batch_info in data_test]), axis=0)
                    print('Total MSE for: ')
                    print('Score 1 : ', avg_loss_test[0])
                    print('Score 2: ', avg_loss_test[1])
                    print('Score 3: ', avg_loss_test[2])

                    print("Pearson correlation")
                    print(f'Score 1 : {avg_corr_test[0]:.4f}')
                    print(f'Score 2: {avg_corr_test[1]:.4f}')
                    print(f'Score 3: {avg_corr_test[2]:.4f}')

                    # Save prediction metrics in json file
                    save_predictions(fold_path, avg_loss_test, avg_corr_test, all_predictions, all_true_labels)

                # Save model
                best_model_state = model.state_dict()
                
                path2 = os.path.join(fold_path, 'model_weights.pth')
                torch.save(best_model_state, path2)

                # Plot predictions
                show_test_prediction_simulation_new(all_predictions, all_true_labels, self.config['n_classes'], fold_path, m_name)
                
                # Determine Saliency map                
                xai_path = os.path.join(fold_path, self.config["xai_path"])
                os.makedirs(xai_path, exist_ok=True)

                if self.config['n_classes'] == 1:
                    XAI.determine_attribution_cv(model, self.config["xai_method"], self.device, self.device_id, test_loader,
                                                 nr_pts=len(test_loader), save=True, save_path=xai_path, value=0,
                                                 n_classes=self.config['n_classes'])
                elif self.config['n_classes'] == 3:
                    torch.cuda.empty_cache()
                    self.compute_XAI(xai_path, 0, test_loader, model)
                    torch.cuda.empty_cache()
                    self.compute_XAI(xai_path, 1, test_loader, model)
                    torch.cuda.empty_cache()
                    self.compute_XAI(xai_path, 2, test_loader, model)
        
        self.test_analysis()
        
    def test_crossval(self, dataset, k_folds, make_xai=False):
        # Initialize the k-fold cross validation
        kf = KFold(n_splits=k_folds, shuffle=True, random_state=0)
            
        # Loop through each fold
        for fold, (_, test_idx) in enumerate(kf.split(dataset)):
            # Create dataloaders
            test_loader = torch.utils.data.DataLoader(dataset, batch_size=2,
                                                      sampler=torch.utils.data.SubsetRandomSampler(test_idx))
            fold_path = self.get_foldpath(fold)
            print(fold_path)
            
            model_path = os.path.join(fold_path, 'model_weights.pth')
            model, _, _, m_name = self.configure_params()
            model.load_state_dict(torch.load(model_path))
            model.eval()
            
            # test in data
            data_test = self.test(test_loader, model)
            avg_corr_test, all_predictions, all_true_labels = pearson_correlation(data_test, self.config['n_classes'], test=True)
            
            print(f'\n----Summary fold {fold + 1}:----')
            
            if self.config['n_classes'] == 1:
                avg_loss_test = np.mean([batch_info['loss'] for batch_info in data_test])
                print(f"test loss: {avg_loss_test:.4f}")
                print(f"test correlation: {avg_corr_test:.4f}\n")
        
                # Save prediction metrics in json file
                save_predictions(fold_path, avg_loss_test, avg_corr_test, all_predictions, all_true_labels)
                
            elif self.config['n_classes'] == 3:
                avg_loss_test = np.mean(np.array([batch_info['loss'] for batch_info in data_test]), axis=0)
                print('Total MSE for: ')
                print('Score 1 : ', avg_loss_test[0])
                print('Score 2: ', avg_loss_test[1])
                print('Score 3: ', avg_loss_test[2])

                print("Pearson correlation")
                print(f'Score 1 : {avg_corr_test[0]:.4f}')
                print(f'Score 2: {avg_corr_test[1]:.4f}')
                print(f'Score 3: {avg_corr_test[2]:.4f}')
                
                # Save prediction metrics in json file
                save_predictions(fold_path, avg_loss_test, avg_corr_test, all_predictions, all_true_labels)
        
            show_test_prediction_simulation_new(all_predictions, all_true_labels, self.config['n_classes'], fold_path, False, m_name)
            
            if make_xai:
            # Determine saliency map
                xai_path = os.path.join(fold_path, self.config["xai_path"])
                os.makedirs(xai_path, exist_ok=True)

                if self.config['n_classes'] == 1:
                    XAI.determine_attribution_cv(model, self.config["xai_method"], self.device, self.device_id, test_loader,
                                                 nr_pts=len(test_loader), save=True, save_path=xai_path, value=0,
                                                 n_classes=self.config['n_classes'])
                elif self.config['n_classes'] == 3:
                    self.compute_XAI(xai_path, 0, test_loader, model)
                    self.compute_XAI(xai_path, 1, test_loader, model)
                    self.compute_XAI(xai_path, 2, test_loader, model)
                    
        # Calculate metrics for all test predictions
        self.test_analysis(m_name)