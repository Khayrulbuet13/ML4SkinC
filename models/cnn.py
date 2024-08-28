from torchvision import models
from torch import nn
import timm

# original resnet50 model
cnn = models.resnet18(weights='IMAGENET1K_V1')
num_ftrs = cnn.fc.in_features
cnn.fc = nn.Linear(num_ftrs, 2)  # Assuming 2 classes (benign and malignant)




# # Load a pre-trained EfficientNet model
# cnn = models.efficientnet_b0(weights='IMAGENET1K_V1')

# # Get the number of input features of the last fully connected layer
# num_ftrs = cnn.classifier[1].in_features

# # Replace the last fully connected layer with a new one that matches your number of classes
# cnn.classifier[1] = nn.Linear(num_ftrs, 2)  # Assuming 2 classes (benign and malignant)



# # Load a pre-trained EdgeNeXt model
# cnn = timm.create_model('edgenext_small', pretrained=True)

# # Modify the last fully connected layer
# if hasattr(cnn, 'head'):
#     num_features = cnn.head.fc.in_features
#     cnn.head.fc = nn.Linear(num_features, 2)  # Set to 2 for binary classification


# # Load a pre-trained EfficientNetV2 model
# cnn = timm.create_model('efficientnetv2_rw_m', pretrained=True)  # Example: 'efficientnetv2_rw_m' for medium variant




# # Modify the last fully connected layer
# if hasattr(cnn, 'classifier'):
#     num_features = cnn.classifier.in_features
#     cnn.classifier = nn.Linear(num_features, 2) 

# Load a pre-trained resnext model
# cnn = timm.create_model('resnext101_32x16d.fb_swsl_ig1b_ft_in1k', pretrained=True)  # Example: 'resnext101_32x16d' for medium variant

# # Modify the last fully connected layer
# if hasattr(cnn, 'classifier'):
#     num_features = cnn.classifier.in_features
#     cnn.classifier = nn.Linear(num_features, 2) 
# elif hasattr(cnn, 'fc'):
#     num_features = cnn.fc.in_features
#     cnn.fc = nn.Linear(num_features, 2)  # Set to 2 for binary classification


cnn = cnn


# def cnn(model_name, pretrained=True, num_classes=2):
#     # Create the model using the specified model name and pretrained option
#     cnn = timm.create_model(model_name, pretrained=pretrained)

#     # Modify the last fully connected layer according to the model's specific last layer identifier
#     if hasattr(cnn, 'classifier'):
#         num_features = cnn.classifier.in_features
#         cnn.classifier = nn.Linear(num_features, num_classes)
#     elif hasattr(cnn, 'fc'):
#         num_features = cnn.fc.in_features
#         cnn.fc = nn.Linear(num_features, num_classes)
#     elif hasattr(cnn, 'head'):
#         num_features = cnn.head.fc.in_features
#         cnn.head.fc = nn.Linear(num_features, num_classes)

#     return cnn