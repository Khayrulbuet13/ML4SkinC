import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        """
        Initializes Focal Loss.
        :param alpha: Balancing factor for class imbalance.
        :param gamma: Focusing parameter.
        :param reduction: Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'.
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)  # prevents nans when probability is 0
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        
        if self.reduction == 'mean':
            return F_loss.mean()
        elif self.reduction == 'sum':
            return F_loss.sum()
        else:
            return F_loss



import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def get_omega_list(class_dist, device, k=1):
    per_cls_weights = 1.0 / (np.array(class_dist) ** k)
    per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(class_dist)
    return torch.FloatTensor(per_cls_weights).to(device)


def get_binary_omega_list(omega, device, minority_class=1):
    if minority_class == 1:
         weights = [1 - omega, omega]
    else:
         weights = [omega, 1 - omega]
    return torch.FloatTensor(weights).to(device)


def get_delta_list(class_dist, gamma, device):
    temp = (1.0 / np.array(class_dist)) ** gamma
    delta_list = temp / np.min(temp)
    return torch.FloatTensor(delta_list).to(device)


def get_iota_list(class_dist, tau, device):
    cls_probs = [cls_num / sum(class_dist) for cls_num in class_dist]
    iota_list = tau * np.log(cls_probs)
    return torch.FloatTensor(iota_list).to(device)


class VSLoss(nn.Module):
    '''
    Cite: https://github.com/orparask/VS-Loss/blob/main/class_imbalance/losses.py
    '''

    def __init__(self, class_dist, device, omega=0.5, gamma=0.3, tau=1.0):
        super().__init__()
        if len(class_dist) == 2:
            self.omega_list = get_binary_omega_list(omega, device)
        else:
            print(f'Warning: Hyperparameter Omega is not being used since this is a"\
                  " multi-class dataset.')
            self.omega_list = get_omega_list(class_dist, device, k=1)
        self.delta_list = get_delta_list(class_dist, gamma, device)
        self.iota_list = get_iota_list(class_dist, tau, device)

    def forward(self, x, target):
        if self.iota_list is not None and self.delta_list is not None:
            output = x / self.delta_list + self.iota_list

        return F.cross_entropy(output, target, weight=self.omega_list)
    


import torch
import torch.nn.functional as F
import torch.nn as nn

import numpy as np
    
class VSLossLCT(nn.Module):    
    def __init__(self, class_dist, device, omega, gamma, tau):
        super().__init__()
        self.device = device
        self.class_dist = class_dist

        if omega is None:
            self.omega_list = None
        elif len(class_dist) == 2:
            self.omega_list = get_binary_omega_list(omega, device)
        else:
            print(f'Warning: Hyperparameter Omega is not being used since this is a"\
                  " multi-class dataset.')
            self.omega_list = get_omega_list(class_dist, device, k=1)
        self.delta_list = get_delta_list(self.class_dist, gamma, self.device) if gamma is not None else None
        self.iota_list = get_iota_list(self.class_dist, tau, self.device) if tau is not None else None
        

    def forward(self, x, target, hypers):
        i = 0
        if self.omega_list is not None:
            weight = self.omega_list
        else:
            if len(self.class_dist) == 2:
                weight = get_binary_omega_list(hypers[i].item(), self.device)
            else:
                print(f'Warning: Hyperparameter Omega is not being used since this is a"\
                    " multi-class dataset.')
                weight = get_omega_list(self.class_dist, self.device, k=1)
            i += 1
        if self.delta_list is not None:
            delta_list = self.delta_list
        else:
            delta_list = get_delta_list(self.class_dist, hypers[i].item(), self.device)
            i += 1
        if self.iota_list is not None:
            iota_list = self.iota_list
        else:
            iota_list = get_iota_list(self.class_dist, hypers[i].item(), self.device)
            i += 1

        output = x / delta_list + iota_list

        return F.cross_entropy(output, target, weight=weight)