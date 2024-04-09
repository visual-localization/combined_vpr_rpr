import torchvision.models as models
import torch
from torch import nn
import torch.nn.functional as F

from .netvlad_layer import NetVLAD


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class L2Norm(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, input):
        return F.normalize(input, p=2, dim=self.dim)

def get_arch(arch:str = "vgg16"):
    if arch.lower() == "alexnet":
        encoder = models.alexnet(weights=models.AlexNet_Weights.DEFAULT)
        # capture only features and remove last relu and maxpool
        layers = list(encoder.features.children())[:-2]

        # if using pretrained only train conv5
        for l in layers[:-1]:
            for p in l.parameters():
                p.requires_grad = False

    elif arch.lower() == "vgg16":
        encoder = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        # capture only feature part and remove last relu and maxpool
        layers = list(encoder.features.children())[:-2]

        # if using pretrained then only train conv5_1, conv5_2, and conv5_3
        for l in layers[:-5]:
            for p in l.parameters():
                p.requires_grad = False
    
    return torch.nn.Sequential(*layers)

def get_pooling_layer(pooling:str,encoder_dim:int):
    if pooling.lower() == "netvlad":
        return NetVLAD(
            num_clusters=64, dim=encoder_dim, vladv2=False
        )
    elif pooling.lower() == "max":
        global_pool = torch.nn.AdaptiveMaxPool2d((1, 1))
        return torch.nn.Sequential(*[global_pool, Flatten(), L2Norm()])
    
    elif pooling.lower() == "avg":
        global_pool = torch.nn.AdaptiveAvgPool2d((1, 1))
        return torch.nn.Sequential(*[global_pool, Flatten(), L2Norm()])
    else:
        raise ValueError("Unknown pooling type: " + pooling)