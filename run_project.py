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

    if args.model == 'lenet5':
        model = LeNet5(in_channels=3, output_dim=2, padding=0)
    #if args.approach == 'swag':
    #    model = SWAG(model=architecture)
    model.to(device)
    print(model)

    # Define optimizer
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.90, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Train lenet5 on crack dataset
    for epoch in range(0, args.epochs+1):
        train_loss, correct, total = train_step(model, dataset.train_loader, optimizer, device=device)
        scheduler.step()
        train_accuracy = 100.*correct/total
        print(f"TRAIN | EPOCH \t {epoch} | LOSS =\t {train_loss:3f} | ACC =\t {train_accuracy:>0.1f}%")
        if (epoch % 1) == 0:
            val_loss, correct, total = validation_step(model, dataset.val_loader, device=device)
            val_accuracy = 100.*correct/total
            print(f"VALID | EPOCH \t {epoch} | LOSS =\t {val_loss:3f} | VALID-ACC =\t {val_accuracy:>0.1f}%")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='bdl-project')
    parser.add_argument('--model', type=str, default='lenet5')
    parser.add_argument('--approach', type=str, default='swag')
    parser.add_argument('--epochs', type=int, default=100)
    args = parser.parse_args()
    main(args)
