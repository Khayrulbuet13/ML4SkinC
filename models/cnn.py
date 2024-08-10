from torchvision import models
from torch import nn


cnn = models.resnet18(weights='IMAGENET1K_V1')
num_ftrs = cnn.fc.in_features
# cnn.fc = nn.Linear(num_ftrs, 2)  # Assuming 2 classes (benign and malignant)

# Adding a dropout layer before the final fully connected layer
cnn.fc = nn.Sequential(
    # nn.Dropout(0.3),  # Adding a dropout layer with 50% dropout rate
    nn.Linear(num_ftrs, 2)  # Assuming 2 classes (benign and malignant)
)

cnn = cnn