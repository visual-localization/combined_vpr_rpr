import os
from pathlib import Path
from typing import Optional,Dict,Tuple,Any, Union,List
from tqdm import tqdm

import torch
import numpy as np

from .utils import correct_intrinsic_scale,read_depth_image
from const import CAM_RESIZE
from .scene import Scene,SceneDataset,transform
from depth_dpt import DPT_DepthModel

class CamLandmarkDatasetPartial(SceneDataset):
    data_path: Path
    mode:str
    depth_solver: DPT_DepthModel
    
    resize: Optional[Tuple[(int,int)]]
    transforms:Any
    poses: Dict[str,Tuple[np.ndarray,np.ndarray]]
    estimated_depth:str
    img_path_list: List[str] # name of the images in the "image" folder sorted