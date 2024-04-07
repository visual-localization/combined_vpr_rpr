import os
from pathlib import Path
from typing import Optional,Dict,Tuple,Any, Union,List
from tqdm import tqdm

import torch
import numpy as np
import pandas as pd
import cv2

from .utils import correct_intrinsic_scale,read_depth_image
from const import GSV_RESIZE
from .scene import Scene,SceneDataset,transform
from depth_dpt import DPT_DepthModel

def get_img_name(row):
    # given a row from the dataframe
    # return the corresponding image name

    city = row['city_id']
    
    # now remove the two digit we added to the id
    # they are superficially added to make ids different
    # for different cities
    pl_id = row.name
    pl_id = str(pl_id).zfill(7)
    
    panoid = row['panoid']
    year = str(row['year']).zfill(4)
    month = str(row['month']).zfill(2)
    northdeg = str(row['northdeg']).zfill(3)
    lat, lon = str(row['lat']), str(row['lon'])
    sub_dir = str(row["sub_dir"]) + "/" if "sub_dir" in row.index else ""
        
    name = sub_dir+city+'_'+pl_id+'_'+year+'_'+month+'_' + \
        northdeg+'_'+lat+'_'+lon+'_'+panoid+'.jpg'
    return name

def generate_depth_path(img_path:Path)->Path:
    name = str(img_path)
    return Path(name.replace("Images","Depths")[:-4])

class GsvDatasetPartial(SceneDataset):
    data_path: Path
    resize: Optional[Tuple[(int,int)]]
    transforms:Any
    estimated_depth:str
    mode:str
    depth_solver: DPT_DepthModel
    random_state: float  # seeding the pandas sample to get consistent result 
    sample_percent:float # getting the percentage of samples
    
    poses: Dict[str,Tuple[np.ndarray,np.ndarray]]
    img_path_list: List[str] # name of the images in the "image" folder sorted
    
    def __init__(
        self,
        data_path:Path,
        resize:Optional[Tuple[int,int]]=None,
        transforms=None,
        estimated_depth:str='dptkitti',
        mode:str = "db",
        depth_solver: DPT_DepthModel = None,
        random_state:int = 46,
        sample_percent:float = 0.25
    ):
        # Setup 
        self.mode = mode
        self.data_path = data_path # /content/drive/MyDrive/Dataset/gsv-cities/Images/Bangkok
        self.random_state = random_state
        self.sample_percent = sample_percent
        
        # Additional Args
        self.resize = resize if resize is not None else GSV_RESIZE
        self.transforms=transforms
        self.estimated_depth=estimated_depth
        
        # load absolute poses
        self.poses = self.read_poses(self.data_path,self.mode)
        img_path_list = self.poses.keys()
        self.img_path_list = sorted(img_path_list)
        
        # create depth images
        for img_path in tqdm(self.img_path_list):
            input_path = (self.data_path/img_path)
            output_path = generate_depth_path(Path(input_path))
            depth_solver.solo_generate_monodepth(input_path,output_path,self.resize)
        
    
    
    @staticmethod
    def read_intrinsics(img_name: str, resize=None):
        """
        Read the intrinsics of a specific image, according to its name
        """
        img_shape = cv2.imread(img_name).shape
        W,H = img_shape[1],img_shape[0]
        fx, fy, cx, cy, W, H = 744.375,744.375,W/2,H/2
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
        poses = {}
        csv_path = root_path.replace("Images","Dataframes")+".csv"
        df = pd.read_csv(csv_path).set_index("place_id")
        loc_index = pd.unique(df.index)
        for idx in loc_index:
            place = df.loc[idx]
            threshold = int(self.sample_percent*len(place))

            place = place.sample(frac=1,random_state=22)
            place = place.iloc[:threshold] if mode == "db" else place.iloc[threshold:]
            for _, row in place.iterrows():
                img_name = get_img_name(row)
                #TODO: Gotta find out how to turn lat/long and north deg to xyz and quaternion
                qt = np.array([0,0,0,1,0,0,0])
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
        intrinsics_matrix,width,height = self.read_intrinsics(str(self.data_path/name),self.resize)
        
        #Load rotation and translation
        q,t = self.poses[name]
        return Scene.create_dict(
            name,
            image,depth,
            intrinsics_matrix,
            q,t,
            width,height
        ), index