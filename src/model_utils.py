import torch
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F
from utils import progress_bar
from torch.nn.utils import clip_grad_norm_
from torch import nn
from torch.autograd import grad
        
def train_step(net, data_loader, optimizer, swag=False, map=False, epoch=0, epochs=30, lambda_prior = 1e-2, device='cpu'):

    net.train()
    train_loss = 0
    total = 0
    correct = 0
    for batch_idx, (Xbatch, Ybatch) in enumerate(data_loader):
        Xbatch, Ybatch = Xbatch.to(device), Ybatch.to(device)
        optimizer.zero_grad()
        output = net(Xbatch)
        loss = F.cross_entropy(output, Ybatch)

        if map:
            prior_loss = 0.0
            for param in net.parameters():
                prior_loss += torch.sum(param**2)
            prior_loss *= (lambda_prior / 2)
            loss += prior_loss
        train_loss += loss.item()

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

def vi_train_step(model, data_loader, optimizer, kl_weight=1, num_epochs=10, device='cuda'):
    
    model.train()
    train_loss = 0
    kl_loss = 0
    total = 0
    correct = 0
    for batch_idx, (Xbatch, Ybatch) in enumerate(data_loader):
        Xbatch, Ybatch = Xbatch.to(device), Ybatch.to(device)
            
        outputs = model(Xbatch)

        likelihood_loss = F.cross_entropy(outputs, Ybatch)
        kl = model.kl_divergence()
        loss = likelihood_loss + kl_weight * kl / len(data_loader)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        kl_loss += kl.item()
        _, predicted = outputs.max(1)
        total += Ybatch.size(0)
        correct += predicted.eq(Ybatch).sum().item()
            
        progress_bar(batch_idx, len(data_loader), 'Train Loss: %.3f | Train Acc: %.3f%% (%d/%d) | Lik-loss: %.3f | KL-loss: %.3f'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total, likelihood_loss.item(), kl.item()))
        
    return train_loss/(batch_idx+1), correct, total

def vi_predictions(model, val_loader, num_samples=10, device='cpu'):
    model.eval()
    
    all_predictions = []
    all_targets = []
    with torch.no_grad():
        for _, (inputs, targets) in enumerate(tqdm(val_loader)):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Monte Carlo sampling
            mc_outputs = []
            for _ in range(num_samples):
                outputs = model(inputs)
                softmax_outputs = torch.softmax(outputs, dim=1)
                mc_outputs.append(softmax_outputs)
            mc_outputs = torch.stack(mc_outputs)
            
            all_predictions.append(mc_outputs.cpu())
            all_targets.append(targets.cpu())
    
    all_predictions = torch.cat(all_predictions, dim=1)  # [M, N, C]
    all_targets = torch.cat(all_targets)
    
    all_predictions = all_predictions.numpy()
    all_targets = all_targets.numpy()
    
    return all_predictions, all_targets



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

    predictions = np.vstack(val_preds)  # (n_samples, n_classes)
    true_labels = np.array(true_labels)

    return predictions, true_labels

    
def swag_predictions(model, val_loader, num_models=10, device='cpu'):
    all_preds = []
    true_labels = []
    model.eval()
    for i in range(num_models):
        model.load_sample()
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

def enable_dropout(model):
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.train()
                
def mcdropout_predictions(model, val_loader, num_models=10, device='cpu'):
    all_preds = []
    true_labels = []
    enable_dropout(model)
    for i in range(num_models):
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

def laplace_predictions(model, val_loader, num_samples=10, device='cpu'):
    all_preds = []
    true_labels = []
    model.eval()
    params = list(model.parameters())
    theta_map = [p.clone().detach().to(device) for p in params]
    hessian_diag = [torch.zeros_like(p).to(device) for p in params]
    true_labels = []

    for _, (inputs, targets) in enumerate(tqdm(val_loader)):
        inputs, targets = inputs.to(device), targets.to(device)
        true_labels.append(targets.cpu().numpy().flatten())

        # Compute loss
        outputs = torch.clamp(model(inputs), min=-10, max=10)
        loss = torch.nn.functional.cross_entropy(outputs, targets)
        
        # Compute gradients
        grads = grad(loss, params, create_graph=True)
        
        # Compute diagonal of the Hessian (second derivatives)
        for i, g in enumerate(grads):
            hessian_diag[i] += grad(g.sum(), params[i], retain_graph=True)[0].detach()**2 + 1e-6
    
    true_labels = np.concatenate(true_labels, axis=0)
    hessian_diag = [torch.clamp(diag, min=1e-6) for diag in hessian_diag]
    
    predictions = []
    for _ in range(num_samples):
        sampled_params = [
            p + torch.randn_like(p) * (1.0 / h.sqrt()) 
            for p, h in zip(theta_map, hessian_diag)
        ]
        
        for param, sampled_param in zip(model.parameters(), sampled_params):
            param.data.copy_(sampled_param.detach())

        batch_preds = []
        for _, (inputs, targets) in enumerate(tqdm(val_loader)):
            inputs = inputs.to(device)
            with torch.no_grad():
                preds = model(inputs).cpu().numpy()
                if np.any(np.isnan(preds)):
                    raise ValueError("NaN detected in predictions!")
                batch_preds.append(preds)
        
        predictions.append(np.concatenate(batch_preds, axis=0))

    return predictions, true_labels