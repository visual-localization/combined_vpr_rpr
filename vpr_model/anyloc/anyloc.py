import numpy as np
import cv2 as cv
import torch
from torch import nn
from torch.nn import functional as F
from torchvision import transforms as tvf
from torchvision.transforms import functional as T
from PIL import Image
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from typing import Literal, List
import os
from copy import deepcopy
# DINOv2 imports
from .utilities import DinoV2ExtractFeatures
from .utilities import VLAD

FEATURE_DIM = {
    8: 12288,
    16: 24576,
    32: 49152
}

# Program parameters
class AnyLocVPR(torch.nn.Module):
    def __init__(
        self,desc_layer:int = 31, num_c:Literal[8,16,32] = 8,
        desc_facet: Literal["query", "key", "value", "token"] = "value",
        domain:Literal["aerial", "indoor", "urban"]="urban",
        device:str ='cuda' if torch.cuda.is_available() else 'cpu',
        max_img_size: int = 1024
    ):
        super(AnyLocVPR, self).__init__()
        self.max_img_size = max_img_size
        self.extractor = DinoV2ExtractFeatures("dinov2_vitg14", desc_layer, desc_facet, device=device)

        # Ensure that data is present
        ext_specifier = f"dinov2_vitg14/l{desc_layer}_{desc_facet}_c{num_c}"
        c_centers_file = os.path.join("./vpr_model","anyloc","vocabulary", ext_specifier,
                                    domain, "c_centers.pt")
        assert os.path.isfile(c_centers_file), "Cluster centers not cached!"
        c_centers = torch.load(c_centers_file)
        assert c_centers.shape[0] == num_c, "Wrong number of clusters!"

        # VLAD object
        self.vlad = VLAD(num_c, desc_dim=None,
                cache_dir=os.path.dirname(c_centers_file))
        # Fit (load) the cluster centers (this'll also load the desc_dim)
        self.vlad.fit(None)
        
        
    def forward(self, img_pt): 
        bs, c, h, w = img_pt.shape
        h_new, w_new = (h // 14) * 14, (w // 14) * 14
        img_pt = tvf.CenterCrop((h_new, w_new))(img_pt)
        
        base = []
        
        for i in range(bs):
            img_part = img_pt[i][None, ...]
            ret = self.extractor(img_part)
            gd = self.vlad.generate(ret.cpu().squeeze()) # VLAD: shape [agg_dim]
            base += [gd]
        
        return torch.stack(base)