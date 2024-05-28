# model.py
#
# ==========================================
# generate the model.
# ==========================================


import torch.nn as nn  
from torchvision.models.resnet import resnet18, ResNet18_Weights
from torchvision.models.alexnet import alexnet, AlexNet_Weights

model_dict = {
    'resnet18': (resnet18, ResNet18_Weights.DEFAULT),
    'alexnet': (alexnet, AlexNet_Weights.DEFAULT)
}

bird_resnet18 = resnet18()
bird_resnet18.fc = nn.Linear(512, 200)

def init_model(pre_model, num_classes, pretrained=True):
    model, weights = model_dict[pre_model]
    if pretrained:
        model = model(weights=weights)
    else:
        model = model()
    
    if pre_model == "resnet18":
        num_ftrs = model.fc.in_features  
        model.fc = nn.Linear(num_ftrs, num_classes)
    elif pre_model == "alexnet":
        num_ftrs = model.classifier[6].in_features  
        model.classifier[6] = nn.Linear(num_ftrs, num_classes)
  
    return model