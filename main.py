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
from utils import device, calculate_auc, get_least_used_gpu
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
        'model': 'miml-with-csv-updated_auc',
        'train_resnet': True  # Allows controlling trainability of ResNet from params
    }

    # Log device usage
    logging.info(f'Using device={device} 🚀')

    class_mapping = {
        'benign': 0,
        'malignant': 1
        }
    columns_to_use = ['age_approx', 'clin_size_long_diam_mm', 'tbp_lv_A',
       'tbp_lv_Aext', 'tbp_lv_B', 'tbp_lv_Bext', 'tbp_lv_C', 'tbp_lv_Cext',
       'tbp_lv_H', 'tbp_lv_Hext', 'tbp_lv_L', 'tbp_lv_Lext', 'tbp_lv_areaMM2',
       'tbp_lv_area_perim_ratio', 'tbp_lv_color_std_mean', 'tbp_lv_deltaA',
       'tbp_lv_deltaB', 'tbp_lv_deltaL', 'tbp_lv_deltaLB',
       'tbp_lv_deltaLBnorm', 'tbp_lv_eccentricity', 'tbp_lv_minorAxisMM',
       'tbp_lv_nevi_confidence', 'tbp_lv_norm_border', 'tbp_lv_norm_color',
       'tbp_lv_perimeterMM', 'tbp_lv_radial_color_std_max', 'tbp_lv_stdL',
       'tbp_lv_stdLExt', 'tbp_lv_symm_2axis', 'tbp_lv_symm_2axis_angle',
       'tbp_lv_x', 'tbp_lv_y', 'tbp_lv_z']
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
    # model = models.resnet18(weights='IMAGENET1K_V1')
    # num_ftrs = model.fc.in_features


    # model.fc = nn.Linear(num_ftrs, 2)  # Assuming 2 classes (benign and malignant)
    # model = model.to(device)


    mlp = MLP(input_size=34, hidden_size=128, output_size=16)
    model = CombinedModel(mlp=mlp, n_classes=2, train_resnet=params['train_resnet']).to(device)


    # Load existing model if available
    model_saved_path = os.path.join(project.checkpoint_dir, "24 March 17:29-old-normalized.pt")
    if os.path.exists(model_saved_path):
        model.load_state_dict(torch.load(model_saved_path))
        logging.info(f'Model loaded from {model_saved_path}')

    # Model summaries
    logging.info(summary(model.resnet18, input_size=(3, 64, 64)))
    logging.info(summary(mlp, input_size=(34,)))
    # logging.info(summary(model, input_size=(3, 64, 64)))

    # Optimizer and training configuration
    optimizer = optim.Adam(model.parameters(), lr=params['lr'])
    # poutyne_model = Model(model, optimizer, "cross_entropy").to(device)
    poutyne_model = Model(model, optimizer, "cross_entropy", batch_metrics=["accuracy"]).to(device)

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
