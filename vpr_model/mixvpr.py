import glob
import os
from typing import Tuple, Dict, List,Union

import torch
from PIL import Image
from torch.utils import data
import numpy as np
import torchvision.transforms as tvf
from tqdm import tqdm
import cv2

from .main import VPRModel
from .anyloc import AnyLocVPR,FEATURE_DIM
from .netvlad import NetVLADModel
from data import SceneDataset,Scene
from const import MIXVPR_RESIZE

class MatchingPipeline:
    def __init__(
        self,
        ckpt_path:str = None,
        device:str = "cpu",
        vpr_type:str = "MixVPR"
    ):
        self.device= device
        self.vpr_type = vpr_type
        if(vpr_type == "MixVPR"):
            assert ckpt_path is not None, "Please provide a path for MixVPR Model"
            self.model = self.load_model_mixvpr(ckpt_path)
            self.feature_dim = 4096
        elif(vpr_type == "AnyLoc"):
            num_c = 8
            self.feature_dim = FEATURE_DIM[num_c]
            self.model = self.load_model_anyloc(num_c=num_c)
        elif(vpr_type == "NetVLAD"):
            assert ckpt_path is not None, "Please provide a path for NetVLAD Model"
            self.feature_dim = 32768
            self.model = self.load_model_netvlad(ckpt_path=ckpt_path)
        else:
            raise NotImplementedError()

    def load_model_netvlad(self,ckpt_path):
        model = NetVLADModel(
            arch="vgg16",
            pooling="netvlad",
            pararell=True
        ).eval()
        checkpoint = torch.load(ckpt_path, map_location=torch.device(self.device))
        model.load_state_dict(checkpoint["state_dict"])
        model = model.to(self.device)
        
        return model

    def load_model_mixvpr(self,ckpt_path:str):
        model = VPRModel(backbone_arch='resnet50',
                    layers_to_crop=[4],
                    agg_arch='MixVPR',
                    agg_config={'in_channels': 1024,
                                'in_h': 20,
                                'in_w': 20,
                                'out_channels': 1024,
                                'mix_depth': 4,
                                'mlp_ratio': 1,
                                'out_rows': 4},
                    )

        state_dict = torch.load(ckpt_path, map_location=torch.device(self.device))
        model.load_state_dict(state_dict)

        model.eval()
        model.to(self.device)
        
        
        # model = VPRModel.load_from_checkpoint(ckpt_path,map_location=self.device)
        # model.eval()
        print(f"Loaded model from {ckpt_path} Successfully!")
        return model
    
    def load_model_anyloc(self,num_c):
        model = AnyLocVPR(num_c=num_c)
        return model


    def calculate_top_k(self,q_matrix: np.ndarray,
                        db_matrix: np.ndarray,
                        top_k) -> np.ndarray:
        # compute similarity matrix
        similarity_matrix = np.matmul(q_matrix, db_matrix.T)  # shape: (num_query, num_db)

        # compute top-k matches
        top_k_matches = np.argsort(-similarity_matrix, axis=1)[:, :top_k]  # shape: (num_query_images, 10)

        return top_k_matches


    def record_matches(self,top_k_matches: np.ndarray,
                    query_dataset: SceneDataset,
                    database_dataset: SceneDataset,
                    out_file: str = 'record.txt') -> None:
        with open(f'{out_file}', 'a') as f:
            for query_index, db_indices in enumerate(tqdm(top_k_matches, ncols=100, desc='Recording matches')):
                pred_query_path,_ = query_dataset[query_index]
                for i in db_indices.tolist():
                    pred_db_paths,_ = database_dataset[i]
                f.write(f'{pred_query_path} {pred_db_paths}\n')


    def visualize(self,top_k_matches: np.ndarray,
                query_dataset: SceneDataset,
                database_dataset: SceneDataset,
                visual_dir: str = '/root/LOGS/visualize',
                img_resize_size: Tuple = (320, 320)) -> None:
        if not os.path.exists(visual_dir):
            os.makedirs(visual_dir)
        for q_idx, db_idx in enumerate(tqdm(top_k_matches, ncols=100, desc='Visualizing matches')):
            pred_q_path,_ = query_dataset[q_idx]
            q_array = cv2.imread(pred_q_path, cv2.IMREAD_COLOR)
            q_array = cv2.resize(q_array, img_resize_size, interpolation=cv2.INTER_CUBIC)
            gap_array = np.ones((q_array.shape[0], 10, 3)) * 255  # white gap

            for i in db_idx.tolist():
                pred_db_paths,_ = database_dataset[i]
                db_array = cv2.imread(pred_db_paths, cv2.IMREAD_COLOR)
                db_array = cv2.resize(db_array, img_resize_size, interpolation=cv2.INTER_CUBIC)

                q_array = np.concatenate((q_array, gap_array, db_array), axis=1)

            result_array = q_array.astype(np.uint8)
            # result_array = cv2.cvtColor(result_array, cv2.COLOR_RGB2BGR)

            # save result as image using cv2
            cv2.imwrite(f'{visual_dir}/{os.path.basename(pred_q_path)}', result_array)

    def get_path_matches(
        self,
        top_k_matches:np.ndarray,
        query_dataset:SceneDataset,
        database_dataset:SceneDataset
    )->Dict[str,List[str]]:
        res = {}
        for q_idx, db_idx in enumerate(top_k_matches):
            pred_q_path = query_dataset.img_path_list[q_idx]
            res[pred_q_path] = [database_dataset.img_path_list[i] for i in db_idx.tolist()]
        return res
            
    def run(self,query_dataset:SceneDataset,db_dataset:SceneDataset,top_k:int=10)->Dict[str,List[str]]:
        # set up inference pipeline
        database_pipeline = InferencePipeline(model=self.model, dataset=db_dataset, feature_dim=self.feature_dim, device=self.device,vpr_type=self.vpr_type)
        query_pipeline = InferencePipeline(model=self.model, dataset=query_dataset, feature_dim=self.feature_dim, device=self.device,vpr_type=self.vpr_type)

        # run inference
        db_global_descriptors = database_pipeline.run(split='db')  # shape: (num_db, feature_dim)
        query_global_descriptors = query_pipeline.run(split='query')  # shape: (num_query, feature_dim)
        # calculate top-k matches
        top_k_matches = self.calculate_top_k(q_matrix=query_global_descriptors, db_matrix=db_global_descriptors, top_k=top_k)

        # # record query_database_matches
        # record_matches(top_k_matches, query_dataset, database_dataset, out_file='./LOGS/record.txt')

        # # visualize top-k matches
        # visualize(top_k_matches, query_dataset, database_dataset, visual_dir='./LOGS/visualize')\
        
        return self.get_path_matches(top_k_matches,query_dataset,db_dataset)

class InferencePipeline:
    model:Union[VPRModel,AnyLocVPR]
    dataset: SceneDataset
    feature_dim: int
    batch_size: int
    num_workers: int
    device: str
    def __init__(
        self, model:Union[VPRModel,AnyLocVPR], 
        dataset:SceneDataset, 
        feature_dim:int, batch_size:int=4, num_workers:int=5, device:str='cuda',
        vpr_type:str=None
    ):
        self.model = model
        self.dataset = dataset
        self.feature_dim = feature_dim
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.device = device
        self.vpr_type = vpr_type

        self.dataloader = data.DataLoader(self.dataset,
                                          batch_size=self.batch_size,
                                          shuffle=False,
                                          num_workers=self.num_workers,
                                          pin_memory=True,
                                          drop_last=False)

    def run(self, split: str = 'db') -> np.ndarray:
        if os.path.exists(f'/root/LOGS/vpr_cache/{self.vpr_type}_global_descriptors_{split}.npy'):
            print(f"Skipping {split} features extraction of {type(self.model)}, loading from cache")
            return np.load(f'/root/LOGS/vpr_cache/{self.vpr_type}_global_descriptors_{split}.npy')

        with torch.no_grad():
            global_descriptors = np.zeros((len(self.dataset), self.feature_dim))
            for batch in tqdm(self.dataloader, ncols=100, desc=f'Extracting {split} features'):
                scenes, indices = batch
                transforms = tvf.Compose([
                    tvf.Resize(MIXVPR_RESIZE, interpolation=tvf.InterpolationMode.BICUBIC),
                    tvf.Normalize([0.485, 0.456, 0.406],
                                [0.229, 0.224, 0.225])
                ])
                imgs = transforms(scenes["image"])
                imgs = imgs.to(self.device)

                # model inference
                descriptors = self.model(imgs)
                descriptors = descriptors.detach().cpu().numpy()

                # add to global descriptors
                global_descriptors[np.array(indices), :] = descriptors

        # save global descriptors
        np.save(f'/root/LOGS/vpr_cache/{self.vpr_type}_global_descriptors_{split}.npy', global_descriptors)
        return global_descriptors

    def forward(self):
        with torch.no_grad():
            global_descriptors = np.zeros((len(self.dataset), self.feature_dim))
            for batch in tqdm(self.dataloader, ncols=100):
                scenes, indices = batch
                transforms = tvf.Compose([
                    tvf.Resize(MIXVPR_RESIZE, interpolation=tvf.InterpolationMode.BICUBIC),
                    tvf.Normalize([0.485, 0.456, 0.406],
                                [0.229, 0.224, 0.225])
                ])
                imgs = transforms(scenes["image"])
                imgs = imgs.to(self.device)

                # model inference
                descriptors = self.model(imgs)
                descriptors = descriptors.detach().cpu().numpy()

                # add to global descriptors
                global_descriptors[np.array(indices), :] = descriptors
        return global_descriptors


