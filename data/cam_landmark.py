import os
from pathlib import Path
from typing import Optional,Dict,Tuple,Any, Union,List
from tqdm import tqdm

import torch
from torch.utils.data import Dataset
import numpy as np
import cv2

from .utils import correct_intrinsic_scale,read_depth_image
from const import MAPFREE_RESIZE
from .scene import Scene,SceneDataset
from utils import convert_world2cam_to_cam2world

class CamLandmarkDataset(SceneDataset):
    data_path: Path
    img_path: Path
    depth_path: Path
    
    resize: Optional[Tuple[(int,int)]]
    transforms:Any
    poses: Dict[str,Tuple[np.ndarray,np.ndarray]]
    estimated_depth:str
    img_path_list: List[str] # name of the images in the "image" folder sorted
    
    def __init__(
        self,
        data_path:Path,
        resize:Optional[Tuple[int,int]]=None,
        transforms=None,
        estimated_depth:str='dptkitti'
    ):
        
        # Setup all required path
        self.data_path = data_path
        self.img_path = Path(os.path.join(data_path,'image'))
        self.depth_path = Path(os.path.join(data_path,'depth'))
        
        # Additional Args
        self.resize = resize if resize is not None else MAPFREE_RESIZE
        self.transforms=transforms
        self.estimated_depth=estimated_depth
        
        # load absolute poses
        self.poses = self.read_poses(self.data_path)
        img_path_list = list(os.listdir(self.img_path))
        self.img_path_list = sorted(img_path_list)