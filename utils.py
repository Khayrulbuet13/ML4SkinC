import torch
import matplotlib.pyplot as plt
import numpy as np
from torchvision.utils import make_grid

device = 'cuda' if torch.cuda.is_available() else 'cpu'
plt.interactive(False)


def show(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)), interpolation='nearest')
    plt.show()
    

def show_dataset(dataset, n=6):
    imgs = [dataset[i][0] for i in range(n)]
    grid = make_grid(imgs)
    plt.imshow(grid.numpy().transpose((1, 2, 0)))
    plt.tight_layout()
    plt.show()


def show_dl_combined(dl, n=3):
    batch = None
    for batch in dl:
        break

    imgs = batch[0][0][:n*n]
   
    grid = make_grid(imgs, nrow =n, padding=20)
    plt.imshow(grid.numpy().transpose((1, 2, 0)))
    plt.tight_layout()
    plt.show()
    
    
def show_dl(dl, n=3):
    batch = None
    for batch in dl:
        break
    imgs = batch[0][:n*n]
    grid = make_grid(imgs, nrow =n, padding=20)
    plt.imshow(grid.numpy().transpose((1, 2, 0)))
    plt.tight_layout()
    plt.show()

import matplotlib.pyplot as plt
import torch

def plot_dl(dl,  n=6, labels_map=None,):
    figure = plt.figure(figsize=(8, 8))  # Adjust the figure size if needed
    cols, rows = n, n

    # Fetch a single batch from the DataLoader
    images, labels = next(iter(dl))
    
    for i in range(1, cols * rows + 1):
        img, label = images[i-1], labels[i-1].item()  # Adjust indexing based on batch
        figure.add_subplot(rows, cols, i)
        # Check if label map is provided, else just use the numeric label
        label_title = labels_map[label] if labels_map else str(label)
        plt.title(label_title)
        plt.axis("off")
        # Ensure the image tensor is in the correct shape for display
        if img.shape[0] == 3:
            plt.imshow(img.permute(1, 2, 0))  # For RGB images
        else:
            plt.imshow(img.squeeze(), cmap='gray')  # For grayscale images, squeeze is used to drop extra dims
    plt.show()



import torch
from sklearn.metrics import roc_curve, auc

def calculate_auc(model, data_loader, device):
    model.network.eval()  # Ensure the model is in evaluation mode
    true_labels = []
    predictions = []

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model.network(inputs)
            probabilities = torch.softmax(outputs, dim=1)[:, 1]  # Assuming binary classification
            predictions.extend(probabilities.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    fpr, tpr, thresholds = roc_curve(true_labels, predictions)
    auc_score = auc(fpr, tpr)
    model.network.train()  # Reset to training mode
    return auc_score



import subprocess
import os

def get_least_used_gpu():
    # Command to get GPU usage (memory and compute)
    smi_output = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used,utilization.gpu', '--format=csv,noheader,nounits']).decode()

    # Parse the output to get a list of (memory used, gpu utilization) tuples
    gpu_stats = [tuple(map(int, line.split(', '))) for line in smi_output.strip().split('\n')]
    
    # Calculate a simple score by adding memory usage and GPU utilization (you can customize this)
    usage_scores = [memory + utilization for memory, utilization in gpu_stats]

    # Get the index of the GPU with the lowest score
    least_used_gpu = usage_scores.index(min(usage_scores))
    return least_used_gpu