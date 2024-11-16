import logging
import torch
import numpy as np
import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

class CrackLoader():

    def __init__(self, data_dir, batch_size=250, normalize=True, static_split=0.8):
        logger.info(f'Loading crack dataset')
        
        # Read data
        transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),         
        ])

        train_image_dataset = datasets.ImageFolder(root=os.path.join(data_dir, 'train'), transform=transform)
        self.train_size = int(static_split * len(train_image_dataset))
        self.val_size = len(train_image_dataset) - self.train_size 
        train_dataset, val_dataset = random_split(train_image_dataset, [self.train_size, self.val_size])

        # Create a dataloader for TRAIN and VAL
        self.train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
        self.val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

        # Create a dataloader for test
        # test_image_dataset = datasets.ImageFolder(root=os.path.join(data_dir, 'test'), transform=transform)
        # self.test_loader = DataLoader(test_image_dataset, batch_size=32, shuffle=False, num_workers=4)