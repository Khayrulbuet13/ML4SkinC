from poutyne.framework.callbacks import Callback
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
import logging

# class CometCallback(Callback):
#     def __init__(self, experiment):
#         super().__init__()
#         self.experiment = experiment

#     def on_epoch_end(self, epoch, logs):
#         self.experiment.log_metrics(logs, step=epoch)



class CometCallback(Callback):
    def __init__(self, experiment, optimizer):
        super().__init__()
        self.experiment = experiment
        self.optimizer = optimizer

    def on_epoch_end(self, epoch, logs):
        # Log existing metrics
        self.experiment.log_metrics(logs, step=epoch)

        # Log the current learning rate
        current_lr = [group['lr'] for group in self.optimizer.param_groups][0]  # Assuming single parameter group
        self.experiment.log_metric('learning_rate', current_lr, step=epoch)




class SchedulerCallback(Callback):
    def __init__(self, scheduler, optimizer):
        super().__init__()
        self.scheduler = scheduler
        self.optimizer = optimizer

    def on_epoch_end(self, epoch, logs=None):
        # Update the scheduler with the current epoch
        self.scheduler.step(epoch)

        # Log the updated learning rate to ensure it's changing as expected
        current_lr = [group['lr'] for group in self.optimizer.param_groups][0]  # Assuming single parameter group
        logging.info(f"Updated learning rate to: {current_lr}")






class PartialAUCMonitor(Callback):
    def __init__(self, validation_data, min_tpr=0.8, device='cuda'):
        super().__init__()
        self.validation_data = validation_data
        self.min_tpr = min_tpr
        self.max_fpr = 1 - min_tpr  # Calculate max FPR from min TPR
        self.device = device

    def on_epoch_end(self, epoch, logs):
        self.model.network.eval()  # Set the model to evaluation mode

        val_y_preds = []
        val_y_trues = []

        with torch.no_grad():
            for images, labels in self.validation_data:
                images = images.to(self.device)  # Move images to the specified device
                outputs = self.model.network(images)   
                probabilities = torch.softmax(outputs, dim=1)[:, 1]  # Assuming binary classification
                val_y_preds.extend(probabilities.cpu().numpy())
                val_y_trues.extend(labels.cpu().numpy())

        # Calculate pAUC using sklearn's roc_auc_score with max_fpr
        partial_auc_scaled = roc_auc_score(val_y_trues, val_y_preds, max_fpr=self.max_fpr)

        # Scale from [0.5, 1.0] to [0.0, 0.2]
        partial_auc = (partial_auc_scaled - 0.5) * 0.4

        logs['val_auc'] = partial_auc  # Add the scaled pAUC to logs
        self.model.network.train()  # Reset the model to training mode
