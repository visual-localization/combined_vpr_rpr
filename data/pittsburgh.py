import os
from pathlib import Path
from typing import Optional,Dict,Tuple,Any, Union,List
from tqdm import tqdm

import torch
import numpy as np

from .utils import correct_intrinsic_scale,read_depth_image
from const import PITTS_RESIZE
from .scene import Scene,SceneDataset,transform
from depth_dpt import DPT_DepthModel

def generate_depth_path(img_path:str)->str:
    if 'queries' in str(img_path):
        depth_path=img_path.replace("queries","queries_depth")[:-4]
    else:
        depth_path=img_path.replace("images","depths")[:-4]
    return depth_path

def join_db_img(root_dir,dbIm):
    #root_dir: /pitts250k
    [folder,name] = dbIm.split("/") #003/0003313
    
    img_parts = name[:-4].split("_")
    split_info = int(img_parts[0])-int(folder)*1000
    
    new_index = int(split_info/250)
    new_folder = f"00{new_index}" #000/001/002/003
    new_path = os.path.join(f"{root_dir}_images_{str(folder)}",new_folder,name)
    return new_path


class Pittsburgh250kSceneDataset(SceneDataset):
    set_name:str
    data_path: Path
    mode:str
    depth_solver: DPT_DepthModel

    resize: Optional[Tuple[(int,int)]]
    transforms:Any
    poses: Dict[str,Tuple[np.ndarray,np.ndarray]]
    estimated_depth:str
    img_path_list: List[str] # name of the images in the "image" folder sorted

    def __init__(
        self,
        set_name:str,
        data_path:Path,
        resize:Optional[Tuple[int,int]]=None,
        transforms=None,
        estimated_depth:str='dptkitti',
        mode:str = "db",
        depth_solver: DPT_DepthModel = None
    ):
        # Setup 
        self.mode = mode
        self.data_path = data_path # /pitts250k
        self.set_name = set_name # pitts250k_test

        # Additional Args
        self.resize = resize if resize is not None else PITTS_RESIZE
        self.transforms=transforms
        self.estimated_depth=estimated_depth

        # load absolute poses
        self.poses = self.read_poses(self.data_path,self.mode)
        img_path_list = self.poses.keys()
        self.img_path_list = sorted(img_path_list)

        # create depth images
        for img_path in tqdm(self.img_path_list):
            input_path = join_db_img(str(self.data_path),img_path)
            output_path = generate_depth_path(input_path)
            # print(f"{img_path} {input_path} {output_path}")
            depth_solver.solo_generate_monodepth(input_path,output_path,self.resize)

    @staticmethod
    def read_intrinsics(img_name: str, resize=None):
        """
        Read the intrinsics of a specific image, according to its name
        """
        fx, fy, cx, cy, W, H = 768.000, 768.000, 320, 240, 648, 480
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
        if resize is not None:
            K = correct_intrinsic_scale(K, resize[0] / W, resize[1] / H)
        return K,W,H

    def read_poses(self, root_path: Path, mode:str) -> Dict[str,Tuple[np.ndarray,np.ndarray]]:
        """
        Returns a dictionary that maps: img_path -> (q, t) where
        np.array q = (qw, qx qy qz) quaternion encoding rotation matrix;
        np.array t = (tx ty tz) translation vector;
        (q, t) encodes absolute pose (world-to-camera), i.e. X_c = R(q) X_W + t
        """
        filename = os.path.join(root_path,"poses",f"{self.set_name}_{mode}.txt")
        poses = {}
        with (filename).open('r') as f:
            for line in tqdm(f.readlines()):
                if(".jpg" not in line):
                    continue
                line = line.strip().split(" ")
                img_name = line[0] # img_name = seq5/frame00587.png
                qt = np.array(list(map(float, line[1:])))
                poses[img_name] = (qt[3:],qt[:3])
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
        image_path = join_db_img(str(self.data_path),name)
        image = transform(image_path,self.resize)

        #Load depth map into torch.tensor
        if self.estimated_depth is not None:
            depth_path = generate_depth_path(image_path)
            depth = read_depth_image(str(depth_path)+".png")
        else:
            depth = torch.tensor([])

        #Load intrinsics matrix
        intrinsics_matrix,width,height = self.read_intrinsics(name,self.resize)

        #Load rotation and translation
        q,t = self.poses[name]
        return Scene.create_dict(
            name,
            image,depth,
            intrinsics_matrix,
            q,t,
            width,height
        ), index