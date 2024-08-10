import torch
from tqdm import tqdm
import time
from sklearn.metrics import roc_auc_score
import numpy as np


# Import callbacks from the separate script
from callbacks import CometCallback, EarlyStopping, ModelCheckpoint

# from torch.cuda.amp import GradScaler, autocast
# scaler = GradScaler()


class SimpleTrainer:
    def __init__(self, model, optimizer, loss_fn, device, callbacks=None, metrics=None, 
                 scheduler=None, scheduler_monitor = 'val_loss',gradient_clip_val=None):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        self.metrics = metrics or {}
        self.scheduler = scheduler
        self.gradient_clip_val = gradient_clip_val
        self.callbacks = callbacks or []
        self.stop_training = False
        self.scheduler_monitor = scheduler_monitor

        # Initialize default metrics
        self.metrics['loss'] = self.compute_loss
        self.metrics['acc'] = self.compute_accuracy
        self.metrics['lr'] = self.compute_learning_rate

        # Set model reference in callbacks
        for callback in self.callbacks:
            callback.set_model(self.model, self.device)

    def register_metric(self, name, metric_fn):
        self.metrics[name] = metric_fn

    def compute_loss(self, outputs, targets):
        loss = self.loss_fn(outputs, targets)
        return loss.item()

    def compute_accuracy(self, outputs, targets):
        _, preds = torch.max(outputs, 1)
        correct = (preds == targets).sum().item()
        return correct / len(targets)* 100

    def compute_learning_rate(self, outputs, targets=None):
        return self.optimizer.param_groups[0]['lr']

    def train(self, train_loader, val_loader, epochs, process_batch_fn=None, verbose=True):
        self.model.to(self.device)

        initial_lr = 0.016  # Set this to your initial learning rate if not set in the optimizer
        warmup_epochs = 10
        base_lr = initial_lr / 10  # Starting LR, 10 times less than initial_lr or your preference
        max_lr = initial_lr 
        # Compute incremental LR step
        lr_increment = (max_lr - base_lr) / warmup_epochs
        
        for epoch in range(epochs):
            epoch_start_time = time.time()
            self.model.train()
            train_metrics = {name: 0.0 for name in self.metrics}



            with tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{epochs}", unit="batch", disable=not verbose) as pbar:
                for inputs, targets in train_loader:
                    # (image, csv), target = inputs.to(self.device), csv.to(self.device), targets.to(self.device)
                    image, csv = inputs  # Unpack inputs tuple
                    image, csv, targets = image.to(self.device), csv.to(self.device), targets.to(self.device)  # Move all to the same device

                   
                    self.optimizer.zero_grad()

                    # Process batch
                    if process_batch_fn:
                        outputs, loss = process_batch_fn(self.model, inputs, targets, self.loss_fn)
                    else:
                        # Default processing
                        outputs = self.model(image, csv)
                        loss = self.loss_fn(outputs, targets)
                    
                    
                    loss.backward()


                    if self.gradient_clip_val:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_val)


                    self.optimizer.step()
                    
                    train_metrics['loss'] += loss.item()

                    # Calculate metrics
                    for name, metric_fn in self.metrics.items():
                        if name != 'loss':
                            train_metrics[name] += metric_fn(outputs, targets)

                    pbar.update(1)
                    torch.cuda.empty_cache()

            # Calculate average training metrics
            avg_train_metrics = {name: val / len(train_loader) for name, val in train_metrics.items()}
            # print(f"Debug - avg_train_metrics: {avg_train_metrics}")  # Debug statement

            # Validation phase
            val_loss, avg_val_metrics = self._validate(val_loader)
            # print(f"Debug - avg_val_metrics: {avg_val_metrics}")  # Debug statement


            
            if self.scheduler:

                if epoch < warmup_epochs:
                    lr = base_lr + epoch * lr_increment
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = lr
                else:
                    # After warmup, use ReduceLROnPlateau
                    metric_to_monitor = avg_val_metrics.get(self.scheduler_monitor, val_loss)
                    self.scheduler.step(metric_to_monitor)

                # metric_to_monitor = avg_val_metrics.get(self.scheduler_monitor, val_loss)
                # self.scheduler.step(metric_to_monitor)

            # Time logging
            epoch_duration = time.time() - epoch_start_time 

            # Construct logs
            logs = {
                'epoch': epoch,
                'epoch_duration': epoch_duration,
                'loss': avg_train_metrics['loss'],
                'acc': avg_train_metrics['acc'],  # Ensure this pulls from training metrics
                'val_loss': avg_val_metrics['val_loss'],  # Ensure this pulls from validation metrics
                'val_acc': avg_val_metrics['acc'],    # Ensure this pulls from validation metrics
                'lr': self.optimizer.param_groups[0]['lr']
            }


            for callback in self.callbacks:
                callback.on_epoch_end(epoch, logs)

                # Check for early stopping
                if isinstance(callback, EarlyStopping) and callback.stop_training:
                    print("Early stopping triggered")
                    return
        


            print(f"Epoch {epoch+1}/{epochs}, train_loss: {float(avg_train_metrics['loss']):.4f}," 
                  f"train_acc: {float(avg_train_metrics['acc']):.4f}, val_loss: {float(val_loss):.4f},"
                  f" val_acc: {float(avg_val_metrics['acc']):.4f}"
)


    def _validate(self, val_loader):
        self.model.eval()
        val_loss = 0.0
        val_metrics = {name: 0.0 for name in self.metrics}

        with torch.no_grad():
            for inputs, targets in val_loader:
                image, csv = inputs  # Unpack inputs tuple
                image, csv, targets = image.to(self.device), csv.to(self.device), targets.to(self.device)
                outputs = self.model(image, csv)
    
                loss = self.loss_fn(outputs, targets)
                val_loss += loss.item()

                # Calculate metrics
                for name, metric_fn in self.metrics.items():
                    if name != 'loss':
                        val_metrics[name] += metric_fn(outputs, targets)

        avg_val_loss = val_loss / len(val_loader)
        avg_val_metrics = {name: val / len(val_loader) for name, val in val_metrics.items()}
        # print(f'avg_val_metrics: {avg_val_metrics}')

        avg_val_metrics['val_loss'] = avg_val_loss
        return avg_val_loss, avg_val_metrics

    def evaluate(self, test_loader):
        self.model.eval()
        test_loss = 0.0
        correct = 0
        total = 0
        test_metrics = {name: 0.0 for name in self.metrics}
        true_labels = []
        predictions = []

        with torch.no_grad():
            for inputs, targets in test_loader:
                image, csv = inputs  # Unpack inputs tuple
                image, csv, targets = image.to(self.device), csv.to(self.device), targets.to(self.device)
                outputs = self.model(image, csv)
    

                loss = self.loss_fn(outputs, targets)
                test_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()

                 # Collect true labels and predictions for pAUC calculation
                probabilities = torch.softmax(outputs, dim=1)[:, 1]  # Assuming binary classification
                predictions.extend(probabilities.cpu().numpy())
                true_labels.extend(targets.cpu().numpy())

                # Calculate metrics
                for name, metric_fn in self.metrics.items():
                    if name != 'loss':
                        test_metrics[name] += metric_fn(outputs, targets)

        accuracy = correct / total
        avg_test_metrics = {name: val / len(test_loader) for name, val in test_metrics.items()}
        avg_test_metrics['test_loss'] = test_loss / len(test_loader)
        avg_test_metrics['test_acc'] = accuracy*100

        # Calculate test pAUC
        max_fpr = 1 - 0.8
        partial_auc_scaled = roc_auc_score(true_labels, predictions, max_fpr=max_fpr)
        partial_auc = (partial_auc_scaled - 0.5) * 0.4
        avg_test_metrics['test_pAUC'] = partial_auc


        return avg_test_metrics