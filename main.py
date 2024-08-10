import os
# from utils import get_least_used_gpu
# os.environ['CUDA_VISIBLE_DEVICES'] = str(get_least_used_gpu())
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

from comet_ml import Experiment
import torch
import json
import datetime
import torch.optim as optim
from torchsummary import summary
from Project import Project
from data import get_dataloader
from data.transformation import train_transform, val_transform
from models.MIML import MLP, CombinedModel
from utils import device, calculate_auc, get_least_used_gpu
from poutyne.framework import Model
from poutyne.framework.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
from callbacks import CometCallback, EarlyStopping, ModelCheckpoint, pAUCMonitor
from logger import logging
from losses import VSLoss

from torchvision import models
import torch.nn as nn
from trainer import SimpleTrainer
import numpy as np




def main():
    project = Project()
    params = {
        'lr': 0.016,
        'weight_decay': .001, # best 0.001
        'batch_size': 1024,
        'epochs': 1000,
        'model': 'MIML-VS_loss-large',
        'train_resnet': True,  # Allows controlling trainability of ResNet from params
        'omega': 0.95,  # Example value for omega
        'gamma': 0.9,  # Example value for gamma
        'tau': 1.0    # Example value for tau
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


    # Initialize model
    number_of_csv_columns = train_dl.dataset.number_of_csv_columns
    mlp = MLP(input_size=number_of_csv_columns, hidden_size=512, output_size=128)
    model = CombinedModel(mlp=mlp, n_classes=2, train_resnet=params['train_resnet']).to(device)



    # # Initialize VSLoss
    # # Calculate class distribution
    # benign_count = 0
    # malignant_count = 0
    # for _, target in train_dl:
    #     benign_count += (target == 0).sum().item()
    #     malignant_count += (target == 1).sum().item()
    # class_dist = [benign_count, malignant_count]
    class_dist =  [240412, 223]
    print(f'Class distribution: {class_dist}')

    
    optimizer = optim.SGD(model.parameters(), lr=params['lr']/10, momentum=0.9)
    from torch.optim.lr_scheduler import ReduceLROnPlateau
    scheduler = ReduceLROnPlateau(optimizer, mode='max', patience=5 )
    criterion = VSLoss(class_dist=class_dist, device=device, omega=params['omega'], gamma=params['gamma'], tau=params['tau'])
    


    # Load existing model if available
    # model_saved_path = os.path.join(project.checkpoint_dir, "24 March 17:29-old-normalized.pt")
    # if os.path.exists(model_saved_path):
    #     model.load_state_dict(torch.load(model_saved_path))
    #     logging.info(f'Model loaded from {model_saved_path}')

    # Model summaries
    # logging.info(summary(model.resnet18, input_size=(3, 64, 64)))
    # logging.info(summary(mlp, input_size=(34,)))
    # logging.info(summary(model, input_size=(3, 64, 64)))


    # Callbacks
    current_time = datetime.datetime.now().strftime('%d %B %H:%M')
    checkpoint_path = os.path.join(project.checkpoint_dir, f"{current_time}-{params['model']}.pt")


    callbacks = [
        pAUCMonitor(val_loader=val_dl),
        CometCallback(experiment),
        EarlyStopping(monitor='val_pAUC', patience=40, mode='max'),
        ModelCheckpoint(filepath=checkpoint_path, monitor='val_pAUC', mode='max', save_best_only=True),
    ]


    # Initialize trainer
    trainer = SimpleTrainer(model, optimizer, criterion, device, callbacks=callbacks, scheduler=scheduler, scheduler_monitor='val_acc')
    

    trainer.train(train_dl, val_dl, epochs=params['epochs'])

    # Evaluate the model
    test_metrics = trainer.evaluate(test_dl)
    print(test_metrics)
    experiment.log_metric('test_loss', test_metrics['test_loss'])
    experiment.log_metric('test_acc', test_metrics['test_acc'])
    experiment.log_metric('test_pAUC', test_metrics['test_pAUC'])


if __name__ == '__main__':
    main()
