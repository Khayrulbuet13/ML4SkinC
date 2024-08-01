from torchvision import models
from torch import nn


cnn = models.resnet50(weights='IMAGENET1K_V1')
num_ftrs = cnn.fc.in_features
cnn.fc = nn.Linear(num_ftrs, 2)  # Assuming 2 classes (benign and malignant)

cnn = cnn