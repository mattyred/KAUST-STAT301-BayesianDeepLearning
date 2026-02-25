## **********
## General imports
import numpy as np
import matplotlib.pyplot as plt

import torch

import torch.optim as optim

from uci_loader_classification import UCIDatasets
from torch.utils.data import DataLoader

from torchvision import datasets, transforms

import models
## **********

WANDB_REPORT = False


## **********
## Initialize wandb for logging of any metrics of interest
if WANDB_REPORT:
    import wandb
    wandb.login(key="YOUR WANDB KEY GOES HERE")
## **********


## **********
## Determine on what device we will run computations: GPU if available, or CPU
print('Using PyTorch version:', torch.__version__)
if torch.cuda.is_available():
    print('Using GPU, device name:', torch.cuda.get_device_name(0))
    device = torch.device('cuda')
else:
    print('No GPU found, using CPU instead.') 
    device = torch.device('cpu')
## **********


## **********
## Select dataset and define some model/optimization hyper-parameters
dataset_name = "banana"
# dataset_name = "MNIST"

learning_rate = 1e-3
weight_decay = 1e-3
Nhidden = 2
Nneurons = 200
Nepochs = 500
batch_size = 200
## **********   

## **********
if dataset_name == "banana":
    NFOLDS = 5
    fold = 1

    ## Load dataset
    dataset = UCIDatasets(dataset_name, "data/", n_splits=NFOLDS)

    ## Create training set and test set
    training_data = dataset.get_split(fold, train = True)
    test_data = dataset.get_split(fold, train = False)

    ## Create data loader
    train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=10000, shuffle=False)

    Din = dataset.in_dim
    Dout = dataset.out_dim


if dataset_name == "MNIST":
    ## Load dataset and create data loader
    train_dataloader = torch.utils.data.DataLoader(datasets.MNIST('data', train=True, download=True, transform=transforms.ToTensor()), batch_size=batch_size, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(datasets.MNIST('data', train=False, transform=transforms.ToTensor()), batch_size=batch_size)
    
    Din = 28 * 28
    Dout = 10
## **********
    

## **********
## This loss definition is for y NOT one-hot encoded
def CrossEntropyLoss(f, y):
    return torch.nn.functional.cross_entropy(f, y)
## **********

## **********
## Training loop function for one epoch
def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)

    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary here but added for possible future use
    model.train()
    for batch, (Xbatch, Ybatch) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(Xbatch)
        loss = loss_fn(pred, Ybatch)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return loss.item()
## **********

## **********
## Test loop to evaluate various metrics on held-out data
def test_loop(dataloader, model, loss_fn):
    # Set the model to testing mode - important for batch normalization and dropout layers
    # Unnecessary here but added for possible future use
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    loss, correct = 0, 0

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for Xbatch, Ybatch in dataloader:
            pred = model(Xbatch)

            correct += (pred.argmax(1) == Ybatch).type(torch.float).sum().item()

            loss += loss_fn(pred, Ybatch) * Xbatch.shape[0]
            
    correct /= size
    loss /= size

    print(f"Test metrics: Accuracy: {(100*correct):>0.1f}%, Avg loss: {loss:>8f}")

    return (100*correct), loss
## **********

## **********
## Create the model and print the architecture
model = models.MLP_shallow(Din, Nneurons, Dout)
print(model)
## **********


## **********
## Loss definition and choice of optimizer 
loss_fn = CrossEntropyLoss

optimizer = optim.Adam(model.parameters(), lr = learning_rate, weight_decay = weight_decay)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = Nepochs)
## **********



## **********
## Create the name of the project in wandb to log all metrics
if WANDB_REPORT:
    array_for_project = ["MLP_shallow", dataset_name]
    array_for_name = ["Nneurons", Nneurons]
    wandb.init(
        # Set the name of the project where this run will be logged
        project = "_".join(str(i) for i in array_for_project),
        # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
        name = "_".join(str(i) for i in array_for_name),
        # Track hyperparameters and run metadata
        config={
            "dataset_name": dataset_name,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "batch_size": batch_size
        }
    )
## **********



## **********
## Training loop over multiple epochs
for epoch in range(Nepochs):
    print(f"** Epoch {epoch+1} ", end="\r"),
    training_loss = train_loop(train_dataloader, model, loss_fn, optimizer)
    lr_scheduler.step()

    ## Keep track of training loss
    if WANDB_REPORT:
        wandb.log({"training/epoch": epoch, "training/loss": training_loss})
    
    if ((epoch+1) % 10) == 0:
        print(f"\ntraining loss: {training_loss:>7f}")

        test_accuracy, test_loss = test_loop(test_dataloader, model, loss_fn)

        ## Keep track of test performance metrics
        if WANDB_REPORT:
            wandb.log({"test/epoch": epoch, "test/accuracy": test_accuracy, "test/loss": test_loss})

if WANDB_REPORT:
    wandb.finish()

print("Done!")
## **********


if dataset_name == "banana":
    # Generating data
    axis_grid_x = np.float32(np.linspace(-4, 4, 100))
    axis_grid_y = np.float32(np.linspace(-4, 4, 100))
    grid_x, grid_y = np.meshgrid(axis_grid_x, axis_grid_y)

    grid = np.column_stack([grid_x.flatten(), grid_y.flatten()])
    with torch.no_grad():
        prediction_grid = model(torch.tensor(grid))
    contour = torch.reshape(prediction_grid[:,0], [100,100])

    fig1, ax2 = plt.subplots(layout='constrained')
    cmap = plt.colormaps["Spectral"]
    PLOT = ax2.contourf(grid_x, grid_y, contour, cmap=cmap)
    PLOT.cmap.set_under('yellow')
    PLOT.cmap.set_over('cyan')

    for dataX, dataY in train_dataloader:
        ax2.scatter(dataX[:,0], dataX[:,1], color=np.array(['blue', 'red'])[dataY.numpy()], alpha=0.1)

    fig1.colorbar(PLOT)
    
    # Displaying the plot
    plt.show()
