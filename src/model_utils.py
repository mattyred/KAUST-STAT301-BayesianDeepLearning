import torch
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F

def train_step(net, data_loader, optimizer, device='cpu'):

    net.train()
    train_loss = 0
    total = 0
    correct = 0
    for batch_idx, (Xbatch, Ybatch) in enumerate(data_loader):
        Xbatch, Ybatch = Xbatch.to(device), Ybatch.to(device)
        optimizer.zero_grad()
        output = net(Xbatch)
        loss = F.cross_entropy(output, Ybatch)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        _, predicted = output.max(1)
        total += Ybatch.size(0)
        correct += predicted.eq(Ybatch).sum().item()
    
    return train_loss/(batch_idx+1), correct, total

def validation_step(net, val_loader, device='cpu'):
    
    net.eval()
    val_loss = 0
    total = 0
    correct = 0
    with torch.no_grad():
        for batch_idx, (Xbatch, Ybatch) in enumerate(val_loader):
            Xbatch, Ybatch = Xbatch.to(device), Ybatch.to(device)
            output = net(Xbatch)
            loss = F.cross_entropy(output, Ybatch)

            val_loss += loss.item()
            _, predicted = output.max(1)
            correct += predicted.eq(Ybatch).sum().item()
            total += Ybatch.size(0)
    
    return val_loss/(batch_idx+1), correct, total

def enable_dropout(model):
    """ Function to enable the dropout layers during test-time """
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()

def get_monte_carlo_predictions(model, data_loader, forward_passes, device='cpu'):
    dropout_predictions = [] 
    softmax = torch.nn.Softmax(dim=1)

    for _ in tqdm(range(forward_passes), desc="Test MC forward passes"):
        predictions = [] 
        model.eval()
        enable_dropout(model)  # Enable dropout layers during test time

        for x, y in data_loader:
            with torch.no_grad():
                x, y = x.to(device), y.to(device)
                output = model(x)
                output = softmax(output)
                predictions.append(output.cpu().numpy())

        predictions = np.vstack(predictions)
        
        dropout_predictions.append(predictions)

    dropout_predictions = np.array(dropout_predictions)  # shape: (forward_passes, n_samples, n_classes)
    
    return dropout_predictions
