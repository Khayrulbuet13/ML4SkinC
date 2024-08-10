from comet_ml import Experiment
import torch
import logging
from sklearn.metrics import roc_auc_score
import torch


class Callback:
    def set_model(self, model, device):
        self.model = model
        self.device = device


class EarlyStopping(Callback):
    def __init__(self, monitor='val_loss', patience=10, mode='min'):
        super().__init__()
        self.monitor = monitor
        self.patience = patience
        self.mode = mode
        self.best_value = float('inf') if mode == 'min' else -float('inf')
        self.counter = 0
        self.stop_training = False

    def on_epoch_end(self, epoch, logs):
        current_value = logs[self.monitor]
        if (self.mode == 'min' and current_value < self.best_value) or (self.mode == 'max' and current_value > self.best_value):
            self.best_value = current_value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                print("Early stopping")
                self.stop_training = True

class ModelCheckpoint(Callback):
    def __init__(self, filepath, monitor='val_loss', mode='min', save_best_only=True):
        super().__init__()
        self.filepath = filepath
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.best_value = float('inf') if mode == 'min' else -float('inf')

    def on_epoch_end(self, epoch, logs):
        current_value = logs[self.monitor]
        if (self.mode == 'min' and current_value < self.best_value) or (self.mode == 'max' and current_value > self.best_value):
            prev_best_value = self.best_value
            self.best_value = current_value
            if self.save_best_only:
                torch.save(self.model.state_dict(), self.filepath)
                print(f"{self.monitor} improved from {prev_best_value:.5f} to {current_value:.5f}, saving file to {self.filepath}")



class CometCallback(Callback):
    def __init__(self, experiment):
        super().__init__()
        self.experiment = experiment

    def on_epoch_end(self, epoch, logs):
        self.experiment.log_metrics(logs, step=epoch)





class pAUCMonitor(Callback):
    def __init__(self, val_loader, min_tpr=0.8):
        super().__init__()
        self.val_loader = val_loader
        self.max_fpr = 1 - min_tpr  # Calculate max FPR from min TPR

    def on_epoch_end(self, epoch, logs):
        self.model.eval()
        true_labels = []
        predictions = []

        with torch.no_grad():
            for inputs, labels in self.val_loader:
                image, csv = inputs
                image, csv = image.to(self.device), csv.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(image, csv)
                probabilities = torch.softmax(outputs, dim=1)[:, 1]  # Assuming binary classification
                predictions.extend(probabilities.cpu().numpy())
                true_labels.extend(labels.cpu().numpy())

        # Calculate pAUC using sklearn's roc_auc_score with max_fpr
        partial_auc_scaled = roc_auc_score(true_labels, predictions, max_fpr=self.max_fpr)

        # Scale from [0.5, 1.0] to [0.0, 0.2]
        partial_auc = (partial_auc_scaled - 0.5) * 0.4

        logs['val_pAUC'] = partial_auc


import numpy as np

class MixUp:
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def __call__(self, x, y):
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1
        batch_size = x.size(0)
        index = torch.randperm(batch_size).to(x.device)

        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]
        
        return mixed_x, y_a, y_b, torch.tensor(lam, dtype=torch.float32, device=x.device) 

    


class CutMix:
    def __init__(self, beta=1.0, prob=0.5):
        self.beta = beta
        self.prob = prob

    def __call__(self, x, y):
        if np.random.rand() > self.prob:
            return x, y, y, 1  # No mix
        batch_size = x.size(0)
        index = torch.randperm(batch_size).to(x.device)

        lam = np.random.beta(self.beta, self.beta)
        bbx1, bby1, bbx2, bby2 = self.rand_bbox(x.size(), lam)

        x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size(-1) * x.size(-2)))
        y_a, y_b = y, y[index]

        return x, y_a, y_b, lam

    def rand_bbox(self, size, lam):
        W = size[2]
        H = size[3]
        cut_rat = np.sqrt(1. - lam)
        cut_w = int(W * cut_rat)  # Use Python's built-in int function
        cut_h = int(H * cut_rat)  # Use Python's built-in int function

        cx = np.random.randint(W)
        cy = np.random.randint(H)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        return bbx1, bby1, bbx2, bby2