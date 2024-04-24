from pathlib import Path
from typing import Tuple,Dict,List
from tqdm import tqdm

import torchvision.transforms as tvf
import torch
import numpy as np

from vpr_model import MatchingPipeline,VPRModel
from depth_dpt import DPT_DepthModel
from mapfree import FeatureDepthModel,Pose,NaivePoseModel
from data import MapfreeDataset,CamLandmarkDataset,CamLandmarkDatasetPartial,GsvDatasetPartial,Pittsburgh250kSceneDataset
from validation import validate_results
from const import MAPFREE_RESIZE,CAM_RESIZE,GSV_RESIZE,PITTS_RESIZE

class RPR_Solver:
    def __init__(self, db_path:Path, query_path:Path,dataset:str="Mapfree",set_name=None,vpr_only=False,vpr_type="MixVPR"):
        self.depth_solver = DPT_DepthModel()
        self.pose_solver = FeatureDepthModel(feature_matching="SuperGlue",pose_solver="EssentialMatrixMetric",dataset=dataset) if not vpr_only else NaivePoseModel()
        self.reranker = self.load_reranker("epoch(08).ckpt")
        self.dataset = dataset
        self.vpr_type = vpr_type
        
        if(dataset == "Mapfree"):
            #Prepare depth image
            self.prep_dataset(db_path,dataset,MAPFREE_RESIZE)
            self.prep_dataset(query_path,dataset,MAPFREE_RESIZE)
            # Load into MapfreeDataset
            self.db_dataset = MapfreeDataset(
                db_path,
                resize=MAPFREE_RESIZE
            )
            self.query_dataset = MapfreeDataset(
                query_path,
                resize=MAPFREE_RESIZE
            )
        elif(dataset == "CamLandmark"):
            self.db_dataset = CamLandmarkDataset(
                db_path,
                mode="db",
                depth_solver=self.depth_solver,
                resize=CAM_RESIZE
            )
            self.query_dataset = CamLandmarkDataset(
                query_path,
                mode="query",
                depth_solver = self.depth_solver,
                resize = CAM_RESIZE    
            )
        elif(dataset == "CamLandmark_Partial"):
            self.db_dataset = CamLandmarkDatasetPartial(
                db_path,
                mode="db",
                depth_solver=self.depth_solver,
                resize=CAM_RESIZE
            )
            self.query_dataset = CamLandmarkDatasetPartial(
                query_path,
                mode="query",
                depth_solver = self.depth_solver,
                resize = CAM_RESIZE    
            )
        elif(dataset == "GSV_Partial"):
            self.db_dataset = GsvDatasetPartial(
                data_path=db_path,
                resize = GSV_RESIZE,
                mode = "db",
                depth_solver=self.depth_solver,
                random_state=222,
                sample_percent=0.5
            )
            self.query_dataset = GsvDatasetPartial(
                data_path=query_path,
                resize = GSV_RESIZE,
                mode = "query",
                depth_solver=self.depth_solver,
                random_state=222,
                sample_percent=0.5
            )

        elif(dataset == "Pittsburgh250k"):
            assert set_name is not None, "Please specify which set"
            self.db_dataset = Pittsburgh250kSceneDataset(
                set_name = set_name,
                data_path=db_path,
                resize = PITTS_RESIZE,
                mode = "db",
                depth_solver=self.depth_solver,
            )
            self.query_dataset = Pittsburgh250kSceneDataset(
                set_name = set_name,
                data_path=query_path,
                resize = PITTS_RESIZE,
                mode = "query",
                depth_solver=self.depth_solver,
            )
        else:
            raise NotImplementedError()
        
    def run(self,top_k=5):
        top_k_matches = self.run_vpr(top_k=top_k)
        poses = self.run_rpr(
            top_k=top_k_matches
        )
        self.validation(poses)

    def run_vpr(self,top_k=10)->Dict[str,List[str]]:
        #Run VPR
        ckpt_path = ""
        if(self.vpr_type == "MixVPR"):
            ckpt_path ="/root/LOGS/init.ckpt"
        elif(self.vpr_type == "NetVLAD"):
            ckpt_path ="/root/LOGS/checkpoint.pth.tar"
        
        matcher = MatchingPipeline(
            ckpt_path=ckpt_path,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            vpr_type=self.vpr_type
        )
        top_k_matches = matcher.run(
            query_dataset=self.query_dataset,
            db_dataset=self.db_dataset,
            top_k=top_k
        )
        return top_k_matches
    
    def run_rpr(
        self,
        top_k:Dict[str,List[str]]
    )->Dict[str,Pose]:
        final_pose = self.pose_solver.process_top_k(
            db_dataset = self.db_dataset,
            query_dataset = self.query_dataset,
            top_k = top_k
        )
        return final_pose

    def validation(
        self,
        final_pose:Dict[str,Pose]
    ):
        validate_results(final_pose,self.query_dataset,self.dataset)
        
    def load_reranker(self,reranker_path:str):
        return VPRModel.load_from_checkpoint(reranker_path).eval()
    
    def rerank(self,top_k_matches,rerank_k):
        res = {}
        
        for key,vals in tqdm(top_k_matches.items()):
            images = torch.stack([self.query_dataset[key][0]["image"]] + [self.db_dataset[val][0]["image"] for val in vals])
            embeddings = self.reranker(images)
            similarity_matrix = embeddings[0].reshape(1,-1)@embeddings[1:].T  # shape: (1, top_k)

            # compute top-k matches
            rerank_top_k = np.argsort(-similarity_matrix, axis=1)[:, :rerank_k]  # shape: (num_query_images, 10)
            res[key] = [vals[db_idx] for db_idx in rerank_top_k]
        
        return res
        
    def prep_dataset(self,data_path:Path,dataset:str,resize):
        self.depth_solver.img_db_process(data_path,dataset,resize)