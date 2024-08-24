import os
# from utils import get_least_used_gpu
# os.environ['CUDA_VISIBLE_DEVICES'] = str(get_least_used_gpu())
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

from comet_ml import Experiment
import torch
import os
import json
import datetime
import torch.optim as optim
from torchsummary import summary
from Project import Project
from data import get_dataloader
from data.transformation import train_transform, val_transform
from models.cnn import cnn
from utils import device, show_dl, setup_seed
from poutyne.framework import Model
from callbacks import CometCallback, EarlyStopping, ModelCheckpoint, pAUCMonitor
from callbacks import MixUp, CutMix
from logger import logging
from losses import VSKLloss

# from timm.scheduler.cosine_lr import CosineLRScheduler
from trainer import SimpleTrainer
import numpy as np


def main():
    setup_seed()
    project = Project()

    params = {
        'lr': 0.016,
        'weight_decay': .001, # best 0.001
        'batch_size': 128,
        'epochs': 1000,
        'cnn': 'edgenext_small',
        'model': 'edgenext_small-x128-KL0.4',
        'train_resnet': True,  # Allows controlling trainability of ResNet from params
        'omega': 0.9,  # Example value for omega
        'gamma': 0.1,  # Example value for gamma
        'tau': 0,    # Example value for tau
        'kl_lambda': 0.4

    }


    # Log device usage
    logging.info(f'Using device={device}: 🚀')

    class_mapping = {
        'benign': 0,
        'malignant': 1
        }
    columns_to_use = None
    # Data loading
    train_dl, val_dl, test_dl = get_dataloader( data_dir=os.path.join(project.data_dir, "image"),
                                                csv_path=os.path.join(project.data_dir, "train-metadata.csv"),
                                                columns_to_use=columns_to_use,
                                                class_mapping = class_mapping,
                                                total_images=None,
                                                train_transform=train_transform,
                                                val_transform=val_transform,
                                                split=(0.8, 0.1, 0.1), 
                                                batch_size=params['batch_size'],
                                                pin_memory=True,
                                                num_workers=8)



    # show images
    show_dl(train_dl, 'Train DL', n=3)
    show_dl(test_dl, 'Test DL', n=3)


    # Initialize Comet Experiment
    with open('secrets.json') as f:
        secrets = json.load(f)
    experiment = Experiment(api_key=secrets['COMET_API_KEY'], 
                    project_name="skin", 
                    workspace="khayrulbuet13")
    experiment.log_parameters(params)
    experiment.set_name(params['model'])



    
    # Model setup
    model = cnn(model_name=params['cnn'], pretrained=True, num_classes=2)
    model = model.to(device)


    # Load existing model if available
    # model_saved_path = os.path.join(project.checkpoint_dir, "24 March 17:29-old-normalized.pt")
    # if os.path.exists(model_saved_path):
    #     model.load_state_dict(torch.load(model_saved_path))
    #     logging.info(f'Model loaded from {model_saved_path}')

    # Model summaries
    # logging.info(summary(model, input_size=(3, 64, 64)))


    # Optimizer and training configuration
    optimizer = optim.SGD(model.parameters(), lr=params['lr'], momentum=0.9)

    from torch.optim.lr_scheduler import ReduceLROnPlateau
    scheduler = ReduceLROnPlateau(optimizer, mode='max', patience=5 )
        


    # # Initialize VSLoss
    # # Calculate class distribution
    # benign_count = 0
    # malignant_count = 0
    # for _, target in train_dl:
    #     benign_count += (target == 0).sum().item()
    #     malignant_count += (target == 1).sum().item()
    # class_dist = [benign_count, malignant_count]
    class_dist =  [320531, 316]                                                                                                                                             
    

    print(f'Class distribution: {class_dist}')
    # class_dist = [len(train_dl.dataset) - sum(y for _, y in train_dl), sum(y for _, y in train_dl)]  # Example class distribution
    mixup = MixUp(alpha=1.0)
    cutmix = CutMix(beta=1.0, prob=0.5)
    criterion = VSKLloss(class_dist=class_dist, device=device, omega=params['omega'], gamma=params['gamma'], tau=params['tau'], kl_lambda=params['kl_lambda'])
    # criterion = FocalKLLoss(alpha=1, gamma=params['gamma'],  kl_lambda=params['kl_lambda']) 

    

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
    trainer = SimpleTrainer(model, optimizer, criterion, device, callbacks=callbacks, scheduler=scheduler, scheduler_monitor='val_acc', gradient_clip_val=1.0)
    

    trainer.train(train_dl, val_dl, epochs=params['epochs'])

    # Evaluate the model
    test_metrics = trainer.evaluate(test_dl)
    print(test_metrics)
    experiment.log_metric('test_loss', test_metrics['test_loss'])
    experiment.log_metric('test_acc', test_metrics['test_acc'])
    experiment.log_metric('test_pAUC', test_metrics['test_pAUC'])


if __name__ == '__main__':
    main()




