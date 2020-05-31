import torch
from torch import nn
from torchvision import models
import config

def load_model():
    model = models.vgg16(pretrained=True, progress=True)

    # Freezing other layers of the model
    for p in model.parameters():
        p.requires_grad = False

    model.classifier = nn.Sequential()
    model.classifier = nn.Sequential(*list(model.classifier) + 
                                    [nn.Linear(25088, 1000)] + 
                                    [nn.ReLU(inplace=True)] + 
                                    [nn.Linear(1000, config.NUM_CLASSES)] +
                                    [nn.LogSoftmax(dim=1)])

    return model

