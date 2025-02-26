import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models

class CNNModel(nn.Module):
    def __init__(self, num_classes=10):
        super(CNNModel, self).__init__()
        self.model = models.resnet18(pretrained=True)
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)

def get_model(num_classes=10):
    model = CNNModel(num_classes)
    return model