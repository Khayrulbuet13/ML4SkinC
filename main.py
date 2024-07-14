import torch
import os
import json
import datetime
from comet_ml import Experiment
import torch.optim as optim
from torchsummary import summary
from Project import Project
from data import get_dataloader
from data.transformation import train_transform, val_transform
from models.cnn import cnn
from utils import device, calculate_auc, get_least_used_gpu
from poutyne.framework import Model
from poutyne.framework.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
from callbacks import CometCallback, PartialAUCMonitor
from logger import logging

from torchvision import models
import torch.nn as nn
from losses import VSLoss


from poutyne.framework.callbacks import Callback
import numpy as np
from sklearn.metrics import roc_curve, auc

from poutyne.framework.callbacks import Callback
import numpy as np
from sklearn.metrics import roc_curve, auc

from poutyne.framework.callbacks import Callback
import torch
import numpy as np
from sklearn.metrics import roc_curve, auc


# Set the least used GPU as visible
least_used_gpu = get_least_used_gpu()
os.environ['CUDA_VISIBLE_DEVICES'] = str(least_used_gpu)
print("Using GPU:", least_used_gpu)


def main():
    project = Project()
    params = {
        'lr': 5e-5,
        'batch_size': 64,
        'epochs': 1000,
        'model': 'resnet18-VS_loss-unbalanced_datasetv2',
        'train_resnet': True,  # Allows controlling trainability of ResNet from params
        'omega': 0.9,  # Example value for omega
        'gamma': 0.4,  # Example value for gamma
        'tau': 1.0    # Example value for tau
    }

    # Log device usage
    logging.info(f'Using device={device} 🚀')

    class_mapping = {
        'benign': 0,
        'malignant': 1
        }
    columns_to_use = None
    # Data loading
    train_dl, val_dl, test_dl = get_dataloader( train_dir=os.path.join(project.data_dir, "train"),
                                                val_dir=os.path.join(project.data_dir, "val"),
                                                train_csv=os.path.join(project.data_dir, "train_age_fixed.csv"),
                                                val_csv=os.path.join(project.data_dir, "val_age_fixed.csv"),
                                                columns_to_use=columns_to_use,
                                                class_mapping = class_mapping,
                                                train_transform=train_transform,
                                                val_transform=val_transform,
                                                split=(0.6, 0.4),
                                                batch_size=params['batch_size'],
                                                pin_memory=True,
                                                num_workers=8)
    


    # Initialize Comet Experiment
    with open('secrets.json') as f:
        secrets = json.load(f)
    experiment = Experiment(api_key=secrets['COMET_API_KEY'], 
                    project_name="skin", 
                    workspace="khayrulbuet13")
    experiment.log_parameters(params)
    experiment.set_name(params['model'])



    
    # Model setup
    model = cnn.to(device)


    # Load existing model if available
    model_saved_path = os.path.join(project.checkpoint_dir, "24 March 17:29-old-normalized.pt")
    if os.path.exists(model_saved_path):
        model.load_state_dict(torch.load(model_saved_path))
        logging.info(f'Model loaded from {model_saved_path}')

    # Model summaries
    logging.info(summary(model, input_size=(3, 64, 64)))


    # Optimizer and training configuration
    optimizer = optim.Adam(model.parameters(), lr=params['lr'])

    # Initialize VSLoss
    # Calculate class distribution
    benign_count = 0
    malignant_count = 0
    for _, target in train_dl:
        benign_count += (target == 0).sum().item()
        malignant_count += (target == 1).sum().item()
    class_dist = [benign_count, malignant_count]
    # class_dist = [len(train_dl.dataset) - sum(y for _, y in train_dl), sum(y for _, y in train_dl)]  # Example class distribution
    criterion = VSLoss(class_dist=class_dist, device=device, omega=params['omega'], gamma=params['gamma'], tau=params['tau'])

    poutyne_model = Model(model, optimizer, criterion, batch_metrics=["accuracy"]).to(device)

    # Callbacks
    current_time = datetime.datetime.now().strftime('%d %B %H:%M')
    checkpoint_path = os.path.join(project.checkpoint_dir, f"{current_time}-{params['model']}.pt")


    callbacks = [
        PartialAUCMonitor(val_dl, min_tpr=0.8, device=device),
        # ReduceLROnPlateau(monitor="val_auc", patience=20, verbose=True),
        ReduceLROnPlateau(monitor="val_auc", mode='max', patience=20, verbose=True),
        # ModelCheckpoint(checkpoint_path, monitor="val_auc", save_best_only=True, verbose=True),
        ModelCheckpoint(checkpoint_path, monitor="val_auc", mode='max', save_best_only=True, verbose=True),
        EarlyStopping(monitor="val_auc", patience=20, mode='max'),
        CometCallback(experiment)
    ]



    # Training
    poutyne_model.fit_generator(train_dl, val_dl, epochs=params['epochs'], callbacks=callbacks)

    # Evaluation
    loss, test_acc = poutyne_model.evaluate_generator(test_dl)
    logging.info(f'Test Accuracy={test_acc}')
    experiment.log_metric('test_acc', test_acc)

    # Calculate AUC for test data
    test_auc = calculate_auc(poutyne_model, test_dl, device)
    logging.info(f'Test AUC={test_auc}')
    experiment.log_metric('test_auc', test_auc)

if __name__ == '__main__':
    main()

