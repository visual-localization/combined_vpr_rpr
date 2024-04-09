import torch
from torch import nn

from .util import get_arch,get_pooling_layer

ENCODER_DIM = {
    "alexnet": 256,
    "vgg16": 512
}

class NetVLADModel(nn.Module):
    def __init__(
        self,
        arch:str = "vgg16",
        pooling:str = "netvlad",
        pararell:bool = True
    ):
        super(NetVLADModel, self).__init__()
        if pararell and (torch.cuda.device_count() > 1):
            print("Pararellizing ... ")
            self.encoder = torch.nn.DataParallel(get_arch(arch=arch)) 
        else:
            self.encoder = get_arch(arch=arch)
        self.pool = get_pooling_layer(pooling=pooling, encoder_dim=ENCODER_DIM[arch])
    
    def forward(self, img):
        x = self.encoder(img)
        x = self.pool(x)
        return x