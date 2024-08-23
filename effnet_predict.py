import os
import argparse
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
import h5py
import numpy as np
import io
import logging
import torch.nn as nn
import torch.nn.functional as F

# Define a custom dataset class to handle HDF5 files
class HDF5Dataset(Dataset):
    def __init__(self, hdf5_file, transform=None):
        self.hdf5_file = hdf5_file
        self.transform = transform
        self.file = h5py.File(hdf5_file, 'r')
        self.keys = list(self.file.keys())

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        image_name = self.keys[idx]
        image_data = self.file[image_name][()]
        image = Image.open(io.BytesIO(image_data)).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, image_name

    def close(self):
        if self.file:
            self.file.close()

def main(model_path, kaggle=True):
    # Initialize device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Define transformations
    transform = transforms.Compose([
        transforms.Resize((128, 128)),  # Adjust the size to 224x224 for ResNet50
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.6984, 0.5219, 0.4197], std=[0.1396, 0.1318, 0.1236]),  # Use ImageNet norms
    ])

    # Create the dataset
    if kaggle:
        hdf5_file = '/kaggle/input/isic-2024-challenge/test-image.hdf5'
    else:
        hdf5_file = 'dataset/dump/train-image.hdf5'
    dataset = HDF5Dataset(hdf5_file=hdf5_file, transform=transform)

    # Create the DataLoader
    test_loader = DataLoader(dataset, batch_size=64, shuffle=False)

    # Model setup
    model = models.efficientnet_b0(weights='IMAGENET1K_V1')
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, 2)  
    model.to(device)

    # Load existing model from the provided path
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f'Model loaded from {model_path}')
        logging.info(f'Model loaded from {model_path}')
    else: 
        print(f'Model not found at {model_path}')
        return

    model.eval()

    # Perform inference
    predictions = []
    image_ids = []

    with torch.no_grad():
        for images, ids in test_loader:
            images = images.to(device)
            outputs = model(images)
            probabilities = F.softmax(outputs, dim=1)
            class_one_prob = probabilities[:, 1].cpu().numpy()
            predictions.extend(class_one_prob)
            image_ids.extend(ids)

    # Cleanup dataset
    dataset.close()

    # Save predictions to CSV
    df = pd.DataFrame({
        'isic_id': image_ids,
        'target': predictions
    })
    df.to_csv('submission.csv', index=False) 
    print("Predictions saved to submission.csv")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run ResNet18 inference for skin cancer detection')
    parser.add_argument('model_path', type=str, help='Path to the pre-trained model file')
    parser.add_argument('kaggle', type=str, help='where to run the code')
    
    args = parser.parse_args()

    main(args.model_path, args.kaggle)
