import argparse
import os
import torch
import numpy as np
from utils import set_seed, ensure_dir
import torch.optim as optim
from src.datasets.crack_loader import CrackLoader
import torchvision.transforms as T
import matplotlib.pyplot as plt
from torchvision import models
from torch import nn
from src.nets.resnet import ResNet18, BaeResNet18
from torchvision.models import resnet18
from src.model_utils import train_step, validation_step, swag_predictions, mcdropout_predictions, predict_step, laplace_predictions, vi_train_step, vi_predictions

DATA_DIR = './data/crack'


# Select device
if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available():
    device = 'mps'
else:
    device = 'cpu'

MODELS_DIR = './experiments/models'
RESULTS_DIR = './experiments/results'
ensure_dir(MODELS_DIR)
ensure_dir(RESULTS_DIR)

# Set random seed
set_seed(0)
        
def main(args):

    dataset = CrackLoader(data_dir=DATA_DIR, batch_size=32, normalize=True)
    print(f'Train [80%]: {dataset.train_size}x3x120x120')
    print(f'Validation [20%]: {dataset.val_size}x3x120x120')
    swag = False
    laplace = False
    mcdropout = False
    optim_only = False
    vi = False
        
    if args.approach == 'swag':
        model = SWAG(model=model)
        swag = True
    elif args.approach == 'mcdropout':
        mcdropout = True
    elif args.approach == 'laplace':
        laplace = True
    elif args.approach == 'vi':
        vi = True
    elif args.approach == 'optim':
        optim_only = True

    if args.model == 'lenet5':
        model = LeNet5(in_channels=3, output_dim=2, padding=0)
    elif args.model == 'resnet18':
        if vi is False:
            pretrained_model = resnet18(pretrained=True)
            dropout_rate = 0.2 if mcdropout else 0
            model = ResNet18(pretrained_model, dropout_rate=dropout_rate, output_dim=2)
        else:
            model = BaeResNet18(num_classes=2)
    
    model.to(device)
    print(model)

    # Define optimizer
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.90, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Train
    train_accuracy_progress = []
    train_loss_progress = []
    valid_accuracy_progress = []
    valid_loss_progress =[]

    for epoch in range(0, args.epochs+1):
        print(f'Epoch : {epoch}')
        if vi is False:
            train_loss, correct, total = train_step(model, dataset.train_loader, optimizer, swag=swag, epoch=epoch, epochs=args.epochs, map=laplace, device=device)
        else:
            train_loss, correct, total = vi_train_step(model, dataset.train_loader, optimizer, device=device, kl_weight=1)
        scheduler.step()
        train_accuracy = 100.*correct/total
        if optim_only:
            train_accuracy_progress.append(train_accuracy)
            train_loss_progress.append(train_loss)

        if (epoch % 10) == 0:
            val_loss, correct, total = validation_step(model, dataset.val_loader, device=device)
            val_accuracy = 100.*correct/total
            if optim_only:
                valid_accuracy_progress.append(val_accuracy)
                valid_loss_progress.append(val_loss)

    # Predict
    if swag:
        predictions, true_labels = swag_predictions(model, dataset.val_loader, num_models=10, device=device)
    elif mcdropout:
        predictions, true_labels = mcdropout_predictions(model, dataset.val_loader, num_models=50, device=device)
    elif laplace:
        predictions, true_labels = laplace_predictions(model, dataset.val_loader, num_samples=10, device=device)
    elif vi:
        predictions, true_labels = vi_predictions(model, dataset.val_loader, num_samples=50, device=device)
    elif optim_only:
        predictions, true_labels = predict_step(model, dataset.val_loader, device=device)
  
    np.savez(os.path.join(RESULTS_DIR, f'{args.model}_{args.approach}.npz'),
                                                             train_accuracy_progress=np.array(train_accuracy_progress), 
                                                             train_loss_progress=np.array(train_loss_progress),
                                                             valid_accuracy_progress=np.array(valid_accuracy_progress),
                                                             valid_loss_progress=np.array(valid_loss_progress),
                                                             predictions=predictions, 
                                                             true_labels=true_labels,
                                                             train_indices=dataset.train_indices,
                                                             val_indices=dataset.val_indices)
    if args.save_model == 1:
        torch.save(model, (os.path.join(MODELS_DIR, f'{args.model}_{args.approach}.pt')))

    


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='bdl-project')
    parser.add_argument('--model', type=str, default='resnet18')
    parser.add_argument('--approach', type=str, default='optim')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--save_model', type=int, default=0)
    args = parser.parse_args()
    main(args)
