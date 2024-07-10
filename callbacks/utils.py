from poutyne.framework.callbacks import Callback
# import torch
# import numpy as np
# from sklearn.metrics import roc_curve, auc

# # class PartialAUCMonitor(Callback):
# #     def __init__(self, validation_data, min_tpr=0.8, device='cuda'):
# #         super().__init__()
# #         self.validation_data = validation_data
# #         self.min_tpr = min_tpr
# #         self.device = device

# #     def on_epoch_end(self, epoch, logs):
# #         # Set the model to evaluation mode
# #         self.model.network.eval()
        
# #         val_y_preds = []
# #         val_y_trues = []

# #         with torch.no_grad():
# #             for inputs, labels in self.validation_data:
# #                 inputs = inputs.to(self.device)
# #                 labels = labels.to(self.device)
# #                 # Use the network directly for inference
# #                 outputs = self.model.network(inputs)
# #                 probabilities = torch.softmax(outputs, dim=1)[:, 1]  # Assuming binary classification
# #                 val_y_preds.extend(probabilities.cpu().numpy())
# #                 val_y_trues.extend(labels.cpu().numpy())

# #         fpr, tpr, thresholds = roc_curve(val_y_trues, val_y_preds)
# #         relevant_indices = np.where(tpr >= self.min_tpr)[0]
# #         if relevant_indices.size > 0:
# #             min_fpr = fpr[relevant_indices[0]]
# #             pAUC = auc(fpr[fpr >= min_fpr], tpr[fpr >= min_fpr]) / (1 - min_fpr)
# #         else:
# #             pAUC = 0

# #         # Add the pAUC to logs
# #         logs['val_auc'] = pAUC  
# #         # Reset the model to training mode
# #         self.model.network.train()
        


# class PartialAUCMonitor(Callback):
#     def __init__(self, validation_data, min_tpr=0.8, device='cuda'):
#         super().__init__()
#         self.validation_data = validation_data
#         self.min_tpr = min_tpr
#         self.device = device

#     def on_epoch_end(self, epoch, logs):
#         self.model.network.eval()  # Set the model to evaluation mode

#         val_y_preds = []
#         val_y_trues = []

#         with torch.no_grad():
#             for (images, csv_data), labels in self.validation_data:
#                 images = images.to(self.device)  # Move images to the specified device
#                 csv_data = csv_data.to(self.device)  # Move CSV data to the specified device
#                 outputs = self.model.network(images, csv_data)   # Pass both images and CSV data
#                 probabilities = torch.softmax(outputs, dim=1)[:, 1]  # Assuming binary classification
#                 val_y_preds.extend(probabilities.cpu().numpy())
#                 val_y_trues.extend(labels.cpu().numpy())

#         fpr, tpr, thresholds = roc_curve(val_y_trues, val_y_preds)
#         relevant_indices = np.where(tpr >= self.min_tpr)[0]
#         if relevant_indices.size > 0:
#             min_fpr = fpr[relevant_indices[0]]
#             pAUC = auc(fpr[fpr >= min_fpr], tpr[fpr >= min_fpr]) / (1 - min_fpr)
#         else:
#             pAUC = 0

#         logs['val_auc'] = pAUC  # Add the pAUC to logs
#         self.model.network.train()  # Reset the model to training mode





import numpy as np
import torch
from sklearn.metrics import roc_auc_score

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
            for (images, csv_data), labels in self.validation_data:
                images = images.to(self.device)  # Move images to the specified device
                csv_data = csv_data.to(self.device)  # Move CSV data to the specified device
                outputs = self.model.network(images, csv_data)   # Pass both images and CSV data
                probabilities = torch.softmax(outputs, dim=1)[:, 1]  # Assuming binary classification
                val_y_preds.extend(probabilities.cpu().numpy())
                val_y_trues.extend(labels.cpu().numpy())

        # Calculate pAUC using sklearn's roc_auc_score with max_fpr
        partial_auc_scaled = roc_auc_score(val_y_trues, val_y_preds, max_fpr=self.max_fpr)

        # Scale from [0.5, 1.0] to [0.0, 0.2]
        partial_auc = (partial_auc_scaled - 0.5) * 0.4

        logs['val_auc'] = partial_auc  # Add the scaled pAUC to logs
        self.model.network.train()  # Reset the model to training mode
