import os
import json
import datetime
from comet_ml import Experiment
import torch
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.models import resnet50  # Using a pre-built model which is commonly used for CIFAR-10

from Project import Project
from utils import device
from callbacks import CometCallback, EarlyStopping, ModelCheckpoint, pAUCMonitor
from trainer import SimpleTrainer
from losses import VSLoss

def main():
    project = Project()

    params = {
        'lr': 0.001,
        'weight_decay': .001,
        'batch_size': 4096,
        'epochs': 200,  # Adjusted for quicker training on a smaller dataset
        'model': 'resnet50',
        'train_resnet': True
    }

    # CIFAR-10 Data Transforms
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    val_transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        transforms.Normalize((0.9, 0.0, 0.0), (0.2, 0.9, 0.9))
    ])

    # Data loading
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
    train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True, num_workers=2)

    val_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=val_transform)
    val_loader = DataLoader(val_dataset, batch_size=params['batch_size'], shuffle=False, num_workers=2)

    # Initialize Comet Experiment
    with open('secrets.json') as f:
        secrets = json.load(f)
    experiment = Experiment(api_key=secrets['COMET_API_KEY'], project_name="cifar10", workspace="khayrulbuet13")
    experiment.log_parameters(params)
    experiment.set_name(params['model'])

    # Model setup
    model = resnet50(weights='IMAGENET1K_V1')
    if params['train_resnet']:
        for param in model.parameters():
            param.requires_grad = True
    model.fc = torch.nn.Linear(model.fc.in_features, 10)  # Adjust for CIFAR-10 classes
    model.to(device)

    # Optimizer and training configuration
    optimizer = optim.SGD(model.parameters(), lr=params['lr'], momentum=0.9, weight_decay=params['weight_decay'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)

    # Loss function setup for balanced dataset
    criterion = torch.nn.CrossEntropyLoss()

    # Callbacks
    current_time = datetime.datetime.now().strftime('%d %B %H:%M')
    checkpoint_path = os.path.join(project.checkpoint_dir, f"{current_time}-{params['model']}.pt")
    callbacks = [
        CometCallback(experiment),
        EarlyStopping(monitor='val_loss', patience=10, mode='min'),
        ModelCheckpoint(filepath=checkpoint_path, monitor='val_loss', mode='min', save_best_only=True),
    ]

    # Initialize trainer
    trainer = SimpleTrainer(model, optimizer, criterion, device, callbacks=callbacks, scheduler=scheduler, scheduler_monitor='val_loss')
    trainer.train(train_loader, val_loader, epochs=params['epochs'])

    # Evaluate the model
    test_metrics = trainer.evaluate(val_loader)
    print(test_metrics)

if __name__ == '__main__':
    main()
