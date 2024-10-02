import logging
import torch
import numpy as np
import os
from torchvision import datasets, transforms
from torch.utils.data import Subset, DataLoader, TensorDataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

class MNISTloader():

    def __init__(self, data_dir, batch_size=50, normalize=True, pca_latents=-1):
        #dataset_path = ('./data/' + dataset + '.pth')
        logger.info(f'Loading MNIST dataset')
        
        # Architectural details
        self.in_channels = 1
        self.padding = 2
        self.input_dim = 28 * 28
        self.output_dim = 10
        self.batch_size = batch_size

        self.train_data = datasets.MNIST(data_dir, download=True, train=False, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]))
        self.test_loader = torch.utils.data.DataLoader(datasets.MNIST(data_dir, download=True, train=False,
                                                                transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])), 
                                                batch_size=self.batch_size, shuffle=False)
        self.train_loader = torch.utils.data.DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True)

class INFMNISTloader():

    def __init__(self, data_dir, batch_size=250, train_fold='TRAIN_FOLD_1', normalize=True, pca_latents=-1):
        #dataset_path = ('./data/' + dataset + '.pth')
        logger.info(f'Loading INFINITE-MNIST dataset')

        # Architectural details
        self.in_channels = 1
        self.padding = 2
        self.input_dim = 28 * 28
        self.output_dim = 10
        self.batch_size = batch_size

        train_data = np.load(os.path.join(data_dir, f"{train_fold}.npz"))
        X, y = train_data['X_train'], train_data['Y_train']
        # Normalize data
        X = (X-X.mean())/(X.std() + 1e-7)
        X = np.expand_dims(X, axis=1) # such that we can get [NUM_BATCHES, 1, 28, 28]
        # Test data remains the same as for MNIST
        self.test_loader = torch.utils.data.DataLoader(datasets.MNIST(data_dir, download=False, train=False,
                                                                transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])), 
                                                batch_size=self.batch_size, shuffle=False)
        self.train_data = TensorDataset(torch.tensor(X, dtype=torch.float32), torch.from_numpy(y))
        self.train_loader = DataLoader(self.train_data, batch_size=self.batch_size, shuffle=False)
