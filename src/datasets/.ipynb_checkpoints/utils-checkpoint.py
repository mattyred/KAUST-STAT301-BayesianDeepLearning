import numpy as np
from torch.utils.data import Subset
import torch

def get_subset(data_loader, perc_samples):

    K = int((perc_samples / 100) * len(data_loader.train_data))
    samples_per_class = K // data_loader.output_dim
        
    selected_indices = []
    targets = torch.Tensor(data_loader.train_data.targets)
    for class_idx in range(data_loader.output_dim):
        class_indices = np.where(targets == class_idx)[0]
        np.random.shuffle(class_indices)
        selected_indices.extend(class_indices[:samples_per_class]) 
        
    np.random.shuffle(selected_indices)
    train_data_subset = Subset(data_loader.train_data, selected_indices)
    subset_loader = torch.utils.data.DataLoader(train_data_subset, batch_size=data_loader.batch_size, shuffle=True)
    
    return subset_loader, selected_indices