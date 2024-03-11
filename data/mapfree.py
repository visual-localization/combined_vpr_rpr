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

def transform(img_path, resize):
    cv_type = cv2.IMREAD_COLOR
    image = cv2.imread(img_path, cv_type)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, resize)
    image = torch.from_numpy(image).float().permute(2, 0, 1) / 255
    return image

class MapfreeDataset(SceneDataset):
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
        
    @staticmethod
    def read_intrinsics(img_name: str, resize=None):
        """
        Read the intrinsics of a specific image, according to its name
        """
        full_name = img_name[:-4]
        parts = full_name.split('_')
        start_idx = 2 if len(parts)==8 else 1
        fx, fy, cx, cy, W, H = tuple(map(float, parts[start_idx:]))

        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
        if resize is not None:
            K = correct_intrinsic_scale(K, resize[0] / W, resize[1] / H)
        return K,W,H

    @staticmethod
    def read_poses(root_path: Path) -> Dict[str,Tuple[np.ndarray,np.ndarray]]:
        """
        Returns a dictionary that maps: img_path -> (q, t) where
        np.array q = (qw, qx qy qz) quaternion encoding rotation matrix;
        np.array t = (tx ty tz) translation vector;
        (q, t) encodes absolute pose (world-to-camera), i.e. X_c = R(q) X_W + t
        """
        poses = {}
        with(root_path / 'poses.txt').open('r') as f:
            for line in tqdm(f.readlines()):
                if '#' in line:
                    continue

                line = line.strip().split(' ')
                img_name = line[0]
                qt = np.array(list(map(float, line[1:])))
                poses[img_name] = (qt[:4], qt[4:])
        return poses

    def __len__(self):
        return len(self.img_path_list)
    
    def __getitem__(self,name_idx:Union[str,int])->Tuple[Scene,int]:
        """
        Args:
            name (str): has name and intrinsics value in it. Ex: s00516_588.6688_588.6688_271.2803_348.6664_540_720.jpg
        Returns:
            scene_obj (Scene): return Scene object
            index (int): return the index of the scene_object in the img_path_list
        """
        if(isinstance(name_idx,int)):
            name = self.img_path_list[name_idx]
            index= name_idx
        else:
            name = name_idx
            index= self.img_path_list.index(name_idx)
        
        #Load image into torch.tensor
        image = transform(str(self.img_path/name),self.resize)

        #Load depth map into torch.tensor
        if self.estimated_depth is not None:
            depth_path = str(self.depth_path / name).replace('.jpg','.png')
            depth = read_depth_image(depth_path)
        else:
            depth = torch.tensor([])
            
        #Load intrinsics matrix
        intrinsics_matrix,width,height = self.read_intrinsics(name,self.resize)
        
        #Load rotation and translation
        q,t = self.poses[name]
        q,t = convert_world2cam_to_cam2world(q,t)
        return Scene.create_dict(
            name,
            image,depth,
            intrinsics_matrix,
            q,t,
            width,height
        ), index