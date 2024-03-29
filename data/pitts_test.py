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

def generate_depth_path(root_path:Path,img_path:Path)->Path:
    # name = str(img_path)
    # return Path(name.replace("input", "working")[:-4])

    name = str(img_path)
    root_name = str(root_path)
    tail_name = name[len(str(root_path)) + 1:]
    head_name_split = root_name.split("/")
    scene_bundle_depth = head_name_split[-1] + "_depth"
    depth_path = os.path.join(
        *head_name_split[:-1],
        scene_bundle_depth,
        tail_name
    )
    return Path("/" + depth_path[:-4])

class Pittsburgh250kSceneDataset(SceneDataset):
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
        data_path:Path,
        resize:Optional[Tuple[int,int]]=None,
        transforms=None,
        estimated_depth:str='dptkitti',
        mode:str = "db",
        depth_solver: DPT_DepthModel = None
    ):
        # Setup 
        self.mode = mode
        self.data_path = data_path # /content/drive/MyDrive/Dataset/pittsburgh250k/data/images


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
            input_path = (self.data_path/img_path)
            output_path = generate_depth_path(self.data_path,Path(input_path))
            # print(f"{img_path} {input_path} {output_path}")
            depth_solver.solo_generate_monodepth(input_path,output_path,self.resize)

    @staticmethod
    def alt_convert_zxy(point_xyz):
        point_xyz[0], point_xyz[1], point_xyz[2] = point_xyz[0], -point_xyz[2], point_xyz[1]
        return point_xyz

    @staticmethod
    def read_intrinsics(img_name: str, resize=None):
        """
        Read the intrinsics of a specific image, according to its name
        """
        fx, fy, cx, cy, W, H = 768.000, 768.000, 320, 240, 600, 400
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
        if resize is not None:
            K = correct_intrinsic_scale(K, resize[0] / W, resize[1] / H)
        return K,W,H

    @staticmethod
    def read_poses(root_path: Path, mode:str) -> Dict[str,Tuple[np.ndarray,np.ndarray]]:
        """
        Returns a dictionary that maps: img_path -> (q, t) where
        np.array q = (qw, qx qy qz) quaternion encoding rotation matrix;
        np.array t = (tx ty tz) translation vector;
        (q, t) encodes absolute pose (world-to-camera), i.e. X_c = R(q) X_W + t
        """

        filename = "pitts30k_test_query.txt"  if mode == "query" \
            else "pitts30k_test_db.txt"
        poses = {}
        with (root_path/filename).open('r') as f:
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
        image = transform(str(self.data_path/name),self.resize)

        #Load depth map into torch.tensor
        if self.estimated_depth is not None:
            depth_path = generate_depth_path(self.data_path,(self.data_path/name))
            depth = read_depth_image(str(depth_path)+".png")
        else:
            depth = torch.tensor([])

        #Load intrinsics matrix
        intrinsics_matrix,width,height = self.read_intrinsics(name,self.resize)

        #Load rotation and translation
        q,t = self.poses[name]
        t = alt_to_zxy(t)
        return Scene.create_dict(
            name,
            image,depth,
            intrinsics_matrix,
            q,t,
            width,height
        ), index