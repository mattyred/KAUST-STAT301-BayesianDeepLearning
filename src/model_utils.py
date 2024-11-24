import torch
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F
from utils import progress_bar
from torch import nn
        
def train_step(net, data_loader, optimizer, swag=False, map=False, lambda_prior = 1e-2, device='cpu'):

    net.train()
    train_loss = 0
    total = 0
    correct = 0
    for batch_idx, (Xbatch, Ybatch) in enumerate(data_loader):
        Xbatch, Ybatch = Xbatch.to(device), Ybatch.to(device)
        optimizer.zero_grad()
        output = net(Xbatch)
        loss = F.cross_entropy(output, Ybatch)
        
        train_loss += loss.item()
        if map:
            prior_loss = 0.0
            for param in net.parameters():
                prior_loss += torch.sum(param**2)
            prior_loss *= (lambda_prior / 2)
            train_loss += prior_loss

        loss.backward()
        optimizer.step()
        
        _, predicted = output.max(1)
        total += Ybatch.size(0)
        correct += predicted.eq(Ybatch).sum().item()

        # Update moments
        if swag:
            net.update_moments()

        progress_bar(batch_idx, len(data_loader), 'Train Loss: %.3f | Train Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    
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

            progress_bar(batch_idx, len(val_loader), 'Valid Loss: %.3f | Valid Acc: %.3f%% (%d/%d)'
                     % (val_loss/(batch_idx+1), 100.*correct/total, correct, total))
    
    return val_loss/(batch_idx+1), correct, total

def predict_step(model, val_loader, device='cpu'):
    true_labels = []
    val_preds = []
    with torch.no_grad():
        for data, target in val_loader:
            data = data.to(device)
            output = model(data)

            softmax_output = torch.softmax(output, dim=1)
            val_preds.append(softmax_output.cpu().numpy())
            true_labels.extend(target.cpu().numpy().flatten())

    predictions = np.concatenate(val_preds, axis=0) # (n_samples, n_classes)
    true_labels = np.array(true_labels)

    return predictions, true_labels
    
def swag_predictions(model, val_loader, num_models=10, device='cpu'):
    all_preds = []
    true_labels = []

    for i in range(num_models):
        model.sample()

        model_preds = [] 
        with torch.no_grad():
            for _, (data, target) in enumerate(tqdm(val_loader)):
                data = data.to(device)
                output = model(data)

                softmax_output = torch.softmax(output, dim=1)
                model_preds.append(softmax_output.cpu().numpy())
                if i == 0:
                    true_labels.extend(target.cpu().numpy().flatten())
        all_preds.append(np.concatenate(model_preds, axis=0))

    predictions = np.array(all_preds) # (num_models, n_samples, n_classes)
    true_labels = np.array(true_labels)

    return predictions, true_labels

def train_map_step(net, data_loader, optimizer, device='cpu'):
    lambda_prior = 1e-2
    criterion = nn.BCEWithLogitsLoss()

    train_loss = 0
    total = 0
    correct = 0
    for batch_idx, (Xbatch, Ybatch) in enumerate(data_loader):
        Xbatch, Ybatch = Xbatch.to(device), Ybatch.to(device)
        optimizer.zero_grad()
        
        logits = net(Xbatch)
        
        # Compute BCE loss (negative log-likelihood)
        Ybatch = Ybatch.float().unsqueeze(1)
        likelihood_loss = criterion(logits, Ybatch)
        
        # Compute prior loss (L2 regularization)
        prior_loss = 0.0
        for param in net.parameters():
            prior_loss += torch.sum(param**2)
        prior_loss *= (lambda_prior / 2)
        
        # Total MAP loss
        loss = likelihood_loss + prior_loss
        
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        predicted = (torch.sigmoid(logits) > 0.5).float()
        total += Ybatch.size(0)
        correct += (predicted == Ybatch).sum().item()

        progress_bar(batch_idx, len(data_loader), 'Train Loss: %.3f | Train Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
        
    return train_loss, correct, total
        
def validation_map_step(net, val_loader, device='cpu'):
    
    net.eval()
    val_loss = 0
    total = 0
    correct = 0
    criterion = nn.BCEWithLogitsLoss()
    with torch.no_grad():
        for batch_idx, (Xbatch, Ybatch) in enumerate(val_loader):
            Xbatch, Ybatch = Xbatch.to(device), Ybatch.to(device)
            logits = net(Xbatch)
            Ybatch = Ybatch.float().unsqueeze(1)
            likelihood_loss = criterion(logits, Ybatch)
            val_loss += likelihood_loss.item()  # without prior term (only likelihood)
            predicted = (torch.sigmoid(logits) > 0.5).float()
            correct += (predicted == Ybatch).sum().item()
            total += Ybatch.size(0)

            progress_bar(batch_idx, len(val_loader), 'Valid Loss: %.3f | Valid Acc: %.3f%% (%d/%d)'
                        % (val_loss/(batch_idx+1), 100.*correct/total, correct, total))
    
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
