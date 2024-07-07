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
from models.MIML import MLP, CombinedModel
from utils import device, calculate_auc
from poutyne.framework import Model
from poutyne.framework.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
from callbacks import CometCallback, PartialAUCMonitor
from logger import logging

from torchvision import models
import torch.nn as nn


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


def main():
    project = Project()
    params = {
        'lr': 5e-5,
        'batch_size': 64,
        'epochs': 1000,
        'model': 'partial_auc_resnet101',
        'train_resnet': True  # Allows controlling trainability of ResNet from params
    }

    # Log device usage
    logging.info(f'Using device={device} 🚀')

    class_mapping = {
        'benign': 0,
        'malignant': 1
        }

    # Data loading
    train_dl, val_dl, test_dl = get_dataloader( train_dir=os.path.join(project.data_dir, "train"),
                                                val_dir=os.path.join(project.data_dir, "val"),
                                                train_csv=os.path.join(project.data_dir, "train.csv"),
                                                val_csv=os.path.join(project.data_dir, "val.csv"),
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
    model = models.resnet101(weights='IMAGENET1K_V1')
    num_ftrs = model.fc.in_features


    model.fc = nn.Linear(num_ftrs, 2)  # Assuming 2 classes (benign and malignant)
    model = model.to(device)


    # mlp = MLP(input_size=3, hidden_size=32, output_size=16)
    # model = CombinedModel(mlp=mlp, n_classes=2, train_resnet=params['train_resnet']).to(device)


    # Load existing model if available
    model_saved_path = os.path.join(project.checkpoint_dir, "24 March 17:29-old-normalized.pt")
    if os.path.exists(model_saved_path):
        model.load_state_dict(torch.load(model_saved_path))
        logging.info(f'Model loaded from {model_saved_path}')

    # Model summaries
    # logging.info(summary(model.resnet18, input_size=(3, 64, 64)))
    # logging.info(summary(mlp, input_size=(3,)))
    logging.info(summary(model, input_size=(3, 64, 64)))

    # Optimizer and training configuration
    optimizer = optim.Adam(model.parameters(), lr=params['lr'])
    poutyne_model = Model(model, optimizer, "cross_entropy", batch_metrics=["accuracy"]).to(device)

    # Callbacks
    current_time = datetime.datetime.now().strftime('%d %B %H:%M')
    checkpoint_path = os.path.join(project.checkpoint_dir, f"{current_time}-{params['model']}.pt")


    # callbacks = [
    #     PartialAUCMonitor(val_dl, min_tpr=0.8, device=device),
    #     ReduceLROnPlateau(monitor="val_auc", patience=20, verbose=True),
    #     ModelCheckpoint(checkpoint_path, save_best_only=True, verbose=True),
    #     CometCallback(experiment)
    # ]
    callbacks = [
        PartialAUCMonitor(val_dl, min_tpr=0.8, device=device),
        ReduceLROnPlateau(monitor="val_auc", patience=20, verbose=True),
        ModelCheckpoint(checkpoint_path, monitor="val_auc", save_best_only=True, verbose=True),
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
