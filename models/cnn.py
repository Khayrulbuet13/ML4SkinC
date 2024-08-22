from torchvision import models
from torch import nn


# cnn = models.resnet50(weights='IMAGENET1K_V1')
# num_ftrs = cnn.fc.in_features
# cnn.fc = nn.Linear(num_ftrs, 2)  # Assuming 2 classes (benign and malignant)




# Load a pre-trained EfficientNet model
cnn = models.efficientnet_b0(weights='IMAGENET1K_V1')

# Get the number of input features of the last fully connected layer
num_ftrs = cnn.classifier[1].in_features

# Replace the last fully connected layer with a new one that matches your number of classes
cnn.classifier[1] = nn.Linear(num_ftrs, 2)  # Assuming 2 classes (benign and malignant)



cnn = cnn