import os
import numpy as np
import torch
import random
import pandas as pd
from torch.utils.data import Dataset, DataLoader, random_split
from matplotlib import pyplot as plt
import prepare_data
from seed import seed_everything
# seeds
np.random.seed(0) # seed for NumPy
# random.seed(0) # for python
torch.manual_seed(0) # seed for PyTorch
torch.cuda.manual_seed(0) # for cuda
# torch.cuda.manual_seed_all(0) # if you are using multi-GPU.
print("Global seed", seed_everything(0))

""" Dataloader and dataset preparation """

# SIMULATION DATASET
class TRACE_VCISimulationDataset(Dataset):
    def __init__(self, scores_df=pd.DataFrame(), output='multi', lesion_matrix=np.zeros((821, 152, 179, 142)),
                 included_pts=None):
        if included_pts is None:
            included_pts = list()
        self.lesion_matrix = lesion_matrix
        self.included_pts = included_pts
        self.output = output
        self.scores_df = scores_df
        self.scores = self.extract_scores()

    def extract_scores(self):
        scores = []
        for pt_id in self.included_pts:
            if self.output == 'multi':
                score1 = self.scores_df[self.scores_df['Patient'] == pt_id]['Score 1'].values[0]
                score2 = self.scores_df[self.scores_df['Patient'] == pt_id]['Score 2'].values[0]
                score3 = self.scores_df[self.scores_df['Patient'] == pt_id]['Score 3'].values[0]
                scores.append([score1, score2, score3])
            elif self.output == 'one':
                score = self.scores_df[self.scores_df['Patient'] == pt_id]['Score'].values[0]
                scores.append([score])

        return np.array(scores)

    def __getitem__(self, i):
        image = self.lesion_matrix[i]
        pt_id = int(self.included_pts[i])

        if self.output == 'multi':
            return (
                torch.unsqueeze(torch.tensor(image, dtype=torch.float32), dim=0),
                *map(lambda x: torch.tensor(x, dtype=torch.float32), self.scores[i]), # [0]
                torch.tensor(pt_id, dtype=torch.int32)
            )
        elif self.output == 'one':
            return (
                torch.unsqueeze(torch.tensor(image, dtype=torch.float32), dim=0),
                torch.tensor(self.scores[i][0], dtype=torch.float32),
                torch.tensor(pt_id, dtype=torch.int32)
            )
        else:
            raise ValueError('Invalid output argument: {}'.format(self.output))

    def __len__(self):
        return len(self.lesion_matrix)
    
    
""" Cross validation Dataset """

def get_data_cv(dir_scores, dir_lesion, output, noise=False, weight='0.1', dependency=False):
    # Get ROI, lesion matrix and patients ids
    patient_ids = prepare_data.get_ids(dir_lesion)
    raw_lesion_matrix = prepare_data.lesion_matrix(patient_ids, dir_lesion)
    
    # Preprocessing
    lesion_matrix, mask_indices, prevalence_mask = prepare_data.pt_threshold(raw_lesion_matrix, threshold=10)

    # load data without noise
    if noise:
        noise_scores = dir_scores + "/noise/gaussian/" + weight
        print("Getting scores from path ", noise_scores)
        data_df = np.load(os.path.join(noise_scores, "ds_simulation_scores" + "_norm" + ".pkl"), allow_pickle=True)
    if dependency:
        dependency_scores = dir_scores + "/scores/" + weight
        print("Getting scores from path ", dependency_scores)
        data_df = np.load(os.path.join(dependency_scores, "ds_simulation_scores.pkl"), allow_pickle=True)
    else:
        data_df = np.load(os.path.join(dir_scores, "ds_simulation_scores.pkl"), allow_pickle=True)

    dataset = TRACE_VCISimulationDataset(data_df, output, lesion_matrix, patient_ids)

    return dataset


""" Load Data for Hyperparameter tunning or without cross validation"""

# Create Dataloader 
def create_dataloader(batch_size, dataset, train_ratio=0.6, val_ratio=0.20, test_ratio=0.20):
    train_set, val_set, test_set = data_split(dataset, train_ratio, val_ratio, test_ratio)
    print('Length dataset: ', len(dataset))
    print('Train length: ', len(train_set), '\t Validation length: ', len(val_set), '\t Test lenght: ', len(test_set))

    datasets = {
        'train': train_set,
        'val': val_set,
        'test': test_set
    }

    dataloaders = {
        'train': DataLoader(train_set, batch_size=batch_size, shuffle=True),
        'val': DataLoader(val_set, batch_size=batch_size, shuffle=False),  # True
        'test': DataLoader(test_set, batch_size=batch_size, shuffle=False)}  # True

    data_lengths = {'train': len(train_set), 'val': len(val_set), 'test': len(test_set)}

    return dataloaders, data_lengths, datasets


def data_split(dataset, train_ratio=0.7, val_ratio=0.2, test_ratio=0.10):
    print(f"Ratio train: {train_ratio}, val: {val_ratio}, test: {test_ratio}")
    print(train_ratio + val_ratio + test_ratio)

    if abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-10:  # if train_ratio + val_ratio + test_ratio == 1.0:
        # Calculate the sizes of the train, validation, and test sets
        train_size = int(train_ratio * len(dataset))
        if test_ratio == 0:
            val_size = len(dataset) - train_size 
            test_size = 0
            train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size])
        else: 
            val_size = int(val_ratio * len(dataset))
            test_size = len(dataset) - train_size - val_size
            # Split the dataset into train, validation, and test sets
            train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size])
        return train_set, val_set, test_set
    else:
        print('Train, validation and test proportions should sum 1')
        return 0
    
def load_data(dir_scores, dir_lesion, output, batch_size, noise=False, train_ratio=0.7, val_ratio=0.2, test_ratio=0.10):
    # Get ROI, lesion matrix and patients ids
    patient_ids = prepare_data.get_ids(dir_lesion)
    raw_lesion_matrix = prepare_data.lesion_matrix(patient_ids, dir_lesion)
    # Preprocessing
    lesion_matrix, mask_indices, prevalence_mask = prepare_data.pt_threshold(raw_lesion_matrix, threshold=10)
    # norm_lesion_matrix, lesion_volumes = prepare_data.volume_correction(lesion_matrix, mode='sqrt')

    # load data without noise
    if noise:
        # path = dir_scores + "/noise/" + simulation_type + "/" + noise_weight
        data_df = np.load(os.path.join(dir_scores, "ds_simulation_scores" + "_norm" + ".pkl"), allow_pickle=True)
    else:
        data_df = np.load(os.path.join(dir_scores, "ds_simulation_scores.pkl"), allow_pickle=True)

    # %%
    dataset = TRACE_VCISimulationDataset(data_df, output, lesion_matrix, patient_ids)

    dataloaders, data_lengths, split_datasets = create_dataloader(batch_size, dataset, train_ratio, val_ratio,
                                                                  test_ratio)
    # %%
    if output == "one":
        print("Visualize the artificial score's ground truth...")
        data = list(data_df['Score'])
        # fixed bin size
        bins = 50  # fixed bin size
        plt.hist(data, bins=bins, alpha=0.5)
        plt.title('Distribution of GT')
        plt.xlabel('Score')
        plt.ylabel('count')
        plt.show()

        # count how many have full empty 0,0,0 scores
        count = (data_df['Score'] == 0).sum()
        print(count, 'patients have all 0 scores')

        print("Visualize the distribution of the partition set...")

        train_targets = []
        validation_targets = []
        test_targets = []

        for _, targets, _ in dataloaders['train']:
            train_targets.extend(targets.tolist())

        for _, targets, _ in dataloaders['val']:
            validation_targets.extend(targets.tolist())

        for _, targets, _ in dataloaders['test']:
            test_targets.extend(targets.tolist())

        # %%
        # Create a figure with three subplots
        fig, axs = plt.subplots(1, 3, figsize=(12, 4))

        # Plot the histograms for each dataset
        axs[0].hist(train_targets, bins=20, alpha=0.5, label='Train')
        axs[1].hist(validation_targets, bins=20, alpha=0.5, label='Validation')
        axs[2].hist(test_targets, bins=20, alpha=0.5, label='Test')

        # Set titles and labels
        axs[0].set_title('Train Set')
        axs[1].set_title('Validation Set')
        axs[2].set_title('Test Set')

        for ax in axs:
            ax.set_xlabel('Target Value')
            ax.set_ylabel('Count')
            ax.legend()

        plt.tight_layout()
        plt.show()

        # Assuming you have a DataLoader called 'dataloader'
        data_iter = iter(dataloaders['test'])
        image, pt_score, pt_id = next(data_iter)

        #         print('Example Batch Patient ', pt_id[0], 'with score:', pt_score[0])
        #         plt.imshow(image[0, 0, 105, :, :], cmap='tab20b', vmin=0, vmax=3)
        #         plt.show()

        # Determine the input size
        input_size = image.shape
        # Exclude the batch dimension (index 0)

        print("Input size:", input_size)

    if output == "multi":

        print("Visualize the artificial score's ground truth...")

        data = list(data_df['Score 1'])
        # Create subplots
        fig, axs = plt.subplots(1, 3, figsize=(10, 6))

        # Plot the first histogram
        axs[0].hist(data, alpha=0.5)
        axs[0].set_xlim([0, 1])
        axs[0].set_title('Distribution of GT')
        axs[0].set_xlabel('Score 1')
        axs[0].set_ylabel('count')

        data = list(data_df['Score 2'])

        # Plot the second histogram
        axs[1].hist(data, alpha=0.5)
        axs[1].set_xlim([0, 1])
        axs[1].set_title('Distribution of GT')
        axs[1].set_xlabel('Score 2')
        axs[1].set_ylabel('count')

        data = list(data_df['Score 3'])

        # Plot the second histogram
        axs[2].hist(data, alpha=0.5)
        axs[2].set_xlim([0, 1])
        axs[2].set_title('Distribution of GT')
        axs[2].set_xlabel('Score 3')
        axs[2].set_ylabel('count')

        # Adjust the spacing between subplots
        plt.tight_layout()

        # Display the plot
        plt.show()

#         # count how many have full empty scores
#         count = (data_df['Score 1'] == 0).sum()
#         print(count, 'patients have all 0 scores for Score 1')

#         count = (data_df['Score 2'] == 0).sum()
#         print(count, 'patients have all 0 scores for Score 2')

#         count = (data_df['Score 3'] == 0).sum()
#         print(count, 'patients have all 0 scores for Score 3')

        num_labels = 3  # Specify the number of labels

        train_label_values = [[] for _ in range(num_labels)]
        validation_label_values = [[] for _ in range(num_labels)]
        test_label_values = [[] for _ in range(num_labels)]

        # Iterate over the train set
        for _, score1, score2, score3, _ in dataloaders['train']:
            train_label_values[0].extend(score1.numpy().tolist())
            train_label_values[1].extend(score2.numpy().tolist())
            train_label_values[2].extend(score3.numpy().tolist())
            
        # Iterate over the validation set
        for _, score1, score2, score3, _ in dataloaders['val']:
            validation_label_values[0].extend(score1.numpy().tolist())
            validation_label_values[1].extend(score2.numpy().tolist())
            validation_label_values[2].extend(score3.numpy().tolist())

        # Iterate over the test set
        for _, score1, score2, score3, _ in dataloaders['test']:
            test_label_values[0].extend(score1.numpy().tolist())
            test_label_values[1].extend(score2.numpy().tolist())
            test_label_values[2].extend(score3.numpy().tolist())
        
        print('Score 1 count')
        print('Train: ', len(train_label_values[0]),'validation: ', len(validation_label_values[0]), 'test:', len(test_label_values[0]))
        
        print('Score 2 count')
        print('Train: ', len(train_label_values[1]), 'validation: ', len(validation_label_values[1]), 'test:', len(test_label_values[1]))
        
        print('Score 3 count')
        print('Train: ', len(train_label_values[2]), 'validation: ', len(validation_label_values[2]), 'test:', len(test_label_values[2]))
        
        
        # %%
        print("Visualize the distribution of the partition set...")
        num_labels = 3  # Specify the number of labels

        label_names = ['Score 1', 'Score 2', 'Score 3']
        datasets = ['train', 'val', 'test']
        title = ['Train Set', 'Validation Set', 'Test Set']

        fig, axs = plt.subplots(3, 3, figsize=(10, 8))

        for i, label_name in enumerate(label_names):
            for j, dataset_name in enumerate(datasets):
                label_values = [[] for _ in range(num_labels)]

                for _, targets_label1, targets_label2, targets_label3, _ in dataloaders[dataset_name]:
                    label_values[0].extend(targets_label1.numpy().tolist())
                    label_values[1].extend(targets_label2.numpy().tolist())
                    label_values[2].extend(targets_label3.numpy().tolist())

                axs[i, j].hist(label_values[i], bins=20, alpha=0.5, label=f'{dataset_name} - {label_name}')
                axs[i, j].set_title(f'{title[j]} ')
                axs[i, j].set_xlabel(f'{label_name} Values')
                axs[i, j].set_ylabel('Count')
                axs[i, j].legend()

        plt.tight_layout()
        plt.show()
        # Assuming you have a DataLoader called 'dataloader'
        data_iter = iter(dataloaders['train'])
        images, _, _, _, _ = next(data_iter)

        # Determine the input size
        input_size = images.shape
        # Exclude the batch dimension (index 0)

        print("Input size:", input_size)

    return dataloaders, dataset



""" Visualizers  """


def show_slices(image, step=1):
    x = np.shape(image)[0]
    for i in range(0, x, step):
        print('Slice ', i)
        plt.imshow(image[i, :, :], cmap='tab20b', vmin=0, vmax=3)
        plt.show()


def show_slices_roi(image, roi_array, step=20):
    a, b, c = np.shape(roi_array)
    overlap = np.zeros(np.shape(roi_array))
    for x in range(a):
        for y in range(b):
            for z in range(c):
                if int(roi_array[x, y, z]) == 1:
                    overlap[x, y, z] += 1
                if int(image[x, y, z]) == 1:
                    overlap[x, y, z] += 2

    show_slices(overlap, step)


def slice_overlap_xai_roi(xai, roi_array, s):
    a, b, c = np.shape(roi_array)
    overlap = np.zeros((b, c))
    for y in range(b):
        for z in range(c):
            x = s
            if int(roi_array[x, y, z]) == 1:
                overlap[y, z] += 1
            if (xai[0, x, y, z] != 0).any():
                overlap[y, z] += xai[0, x, y, z].sum().item()

    return overlap
