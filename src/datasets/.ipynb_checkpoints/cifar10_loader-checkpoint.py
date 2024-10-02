import logging
import torch
import numpy as np
import os
from torchvision import datasets, transforms
from torch.utils.data import Subset, DataLoader, TensorDataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

class CIFAR10loader():

    def __init__(self, data_dir, batch_size=250, normalize=True, pca_latents=-1):
        #dataset_path = ('./data/' + dataset + '.pth')
        logger.info(f'Loading CIFAR-10 dataset')

        # Architectural details
        self.in_channels = 3
        self.padding = 1
        self.input_dim = 3 * 32 * 32
        self.output_dim = 10
        self.batch_size = batch_size

        RC = transforms.RandomCrop(32, padding=4)
        RHF = transforms.RandomHorizontalFlip()
        RVF = transforms.RandomVerticalFlip()
        NRM = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        TT = transforms.ToTensor()
        ALLTRANSFORMS = transforms.Compose([RC, RHF, TT, NRM])

        self.train_data = datasets.CIFAR10(data_dir, download=True, train=True, transform=ALLTRANSFORMS)
        self.test_loader = torch.utils.data.DataLoader(datasets.CIFAR10(data_dir, download=True, train=False,
                                                                 transform=transforms.Compose([TT, NRM])),
                                                                 batch_size=self.batch_size, shuffle=False)
        self.train_loader = torch.utils.data.DataLoader(self.train_data, batch_size=self.batch_size, shuffle=False)