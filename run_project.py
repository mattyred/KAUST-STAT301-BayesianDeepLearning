import argparse
import os
import torch
import numpy as np
from utils import set_seed
import torch.optim as optim
from src.datasets.crack_loader import CrackLoader
import torchvision.transforms as T
import matplotlib.pyplot as plt
from src.nets.lenet import LeNet5
from src.nets.swag import SWAG
from src.model_utils import train_step, validation_step

DATA_DIR = './data/crack'


# Select device
if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available():
    device = 'mps'
else:
    device = 'cpu'

# Set random seed
set_seed(0)
        
def main(args):

    dataset = CrackLoader(data_dir=DATA_DIR, batch_size=64, normalize=True)
    print(f'Train [80%]: {dataset.train_size}x3x120x120')
    print(f'Validation [20%]: {dataset.val_size}x3x120x120')
    swag = False

    if args.model == 'lenet5-optim':
        model = LeNet5(in_channels=3, output_dim=2, padding=0)

    if args.model == 'lenet5-swag':
        swag = True
        model = SWAG(model=LeNet5(in_channels=3, output_dim=2, padding=0))
    model.to(device)
    print(model)

    # Define optimizer
    optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.90, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    
    # Train lenet5 on crack dataset
    train_accuracy_progress = []
    train_loss_progress = []
    valid_accuracy_progress = []
    valid_loss_progress =[]

    for epoch in range(0, args.epochs+1):
        train_loss, correct, total = train_step(model, dataset.train_loader, optimizer, swag=swag, device=device)
        scheduler.step()
        train_accuracy = 100.*correct/total
        train_accuracy_progress.append(train_accuracy)
        train_loss_progress.append(train_loss)

        if (epoch % 5) == 0:
            val_loss, correct, total = validation_step(model, dataset.val_loader, device=device)
            val_accuracy = 100.*correct/total
            valid_accuracy_progress.append(val_accuracy)
            valid_loss_progress.append(val_loss)

    torch.save(model, f'./experiments/{args.model}' + '.pt')

    


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='bdl-project')
    parser.add_argument('--model', type=str, default='lenet5-optim')
    parser.add_argument('--approach', type=str, default='swag')
    parser.add_argument('--epochs', type=int, default=50)
    args = parser.parse_args()
    main(args)
