import logging
import torch
import numpy as np
import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

class CrackLoader():

    def __init__(self, data_dir, batch_size=64, normalize=True, static_split=0.8):
        logger.info(f'Loading crack dataset')
        
        # Read data
        transform = transforms.Compose([
            transforms.Resize((120, 120)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        train_image_dataset = datasets.ImageFolder(root=os.path.join(data_dir, 'train'), transform=transform)
        self.train_size = int(static_split * len(train_image_dataset))
        self.val_size = len(train_image_dataset) - self.train_size 
        indices = list(range(len(train_image_dataset)))
        self.train_indices, self.val_indices = train_test_split(indices, test_size=self.val_size, stratify=train_image_dataset.targets)

        # Create a dataloader for TRAIN and VAL
        self.train_dataset = torch.utils.data.Subset(train_image_dataset, self.train_indices)
        self.val_dataset = torch.utils.data.Subset(train_image_dataset, self.val_indices)
        self.train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        self.val_loader = DataLoader(self.val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

        # Create a dataloader for test
        # test_image_dataset = datasets.ImageFolder(root=os.path.join(data_dir, 'test'), transform=transform)
        # self.test_loader = DataLoader(test_image_dataset, batch_size=32, shuffle=False, num_workers=4)

    def __getitem__(self, index):
        return self.val_dataset[index]