## A datasets of images as for both DB and query should be containing:
# ##### path/to/db
#         => image: containing image files with name: cat_fx_fy_cx_cy_width_height
#         => depth: cat.dptkitti
#         => poses.txt: containing pose of the image if it is a db of reference images: cat qw qx qy qz tx ty tz

# ##### path/to/query
#         => image: containing image files with name: cat_frameid_fx_fy_cx_cy_width_height
#         => depth: cat_frameid.dptkitti
#         => poses.txt: containing pose of the image if it is a db of reference images: cat_frameid qw qx qy qz tx ty tz

import os
from pathlib import Path
from typing import Optional,Dict,Tuple,Any, Union
from tqdm import tqdm

import torch
from torch.utils.data import Dataset
import numpy as np
import cv2

from .utils import read_color_image,correct_intrinsic_scale,read_depth_image
from const import MAPFREE_RESIZE

def transform(img_path, resize):
  cv_type = cv2.IMREAD_COLOR
  image = cv2.imread(img_path, cv_type)
  image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  image = cv2.resize(image, resize)
  image = torch.from_numpy(image).float().permute(2, 0, 1) / 255
  return image



class Scene:
    name: str
    image: torch.tensor
    depth: torch.tensor
    intrinsics_matrix: np.ndarray #(3,3)
    rotation: np.ndarray #[4]: q1,q2,q3,q4
    translation: np.ndarray # [3]: x,y,z: should be the absolute pose of the image
    def create_dict(
        name:str,
        image:torch.tensor, # (h, w, 3) in cpu
        depth:torch.tensor,
        intrinsics_matrix: np.ndarray,
        rotation: np.ndarray,
        translation: np.ndarray,
        width:float,
        height:float
    ):
        return{
            "name": name,
            "image": image,
            "depth": depth,
            "intrinsics_matrix":intrinsics_matrix,
            "rotation": rotation,
            "translation":translation,
            "width": width,
            "height": height
        }

class SceneDataset(Dataset):
    data_path: Path
    img_path: Path
    depth_path: Path
    
    resize: Optional[Tuple[(int,int)]]
    transforms:Any
    poses: Dict[str,Tuple[np.ndarray,np.ndarray]]
    estimated_depth:str
    
    def __init__(
        self,
        data_path:Path = Path("/content/drive/MyDrive/Dataset/map-free-vpr/db"),
        resize:Optional[Tuple[int,int]]=None,
        transforms=None,
        estimated_depth:str='dptkitti'
    ):
        raise NotImplementedError("init function has not been implemented")
        
    @staticmethod
    def read_intrinsics(img_name: str, resize=None):
        raise NotImplementedError("read_intrinsics function has not been implemented")

    @staticmethod
    def read_poses(root_path: Path) -> Dict[str,Tuple[np.ndarray,np.ndarray]]:
        """
        Returns a dictionary that maps: img_path -> (q, t) where
        np.array q = (qw, qx qy qz) quaternion encoding rotation matrix;
        np.array t = (tx ty tz) translation vector;
        (q, t) encodes absolute pose (world-to-camera), i.e. X_c = R(q) X_W + t
        """
        raise NotImplementedError("read_poses function has not been implemented")

    def __len__(self):
        raise NotImplementedError("len function has not been implemented")
    
    def __getitem__(self,name_idx:Union[str,int])->Tuple[Scene,int]:
        """
        Args:
            name (str): has name and intrinsics value in it. Ex: s00516_588.6688_588.6688_271.2803_348.6664_540_720.jpg
            or
            idx (int): the index of the name in the img_path_list as some function used the index
        Returns:
            scene_obj (Scene): lorem
        """
        raise NotImplementedError("get item function has not been implemented")
        
        