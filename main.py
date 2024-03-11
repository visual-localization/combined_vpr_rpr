from pathlib import Path
from typing import Tuple,Dict,List

import torchvision.transforms as tvf
import torch

from mixvpr_model import MatchingPipeline
from depth_dpt import DPT_DepthModel
from mapfree import FeatureDepthModel,Pose
from data import MapfreeDataset,CamLandmarkDataset
from validation import validate_results
from miner import CustomMultiSimilarityMiner

class RPR_Solver:
    def __init__(self, db_path:Path, query_path:Path,dataset:str="Mapfree"):
        self.depth_solver = DPT_DepthModel()
        self.pose_solver = FeatureDepthModel(feature_matching="SuperGlue",pose_solver="EssentialMatrixMetric")
        
        
        if(dataset == "Mapfree"):
            #Prepare depth image
            self.prep_dataset(db_path,dataset)
            self.prep_dataset(query_path,dataset)
            # Load into MapfreeDataset
            self.db_dataset = MapfreeDataset(
                db_path
            )
            self.query_dataset = MapfreeDataset(
                query_path
            )
        elif(dataset == "CamLandmark"):
            self.db_dataset = CamLandmarkDataset(db_path,mode="db",depth_solver = self.depth_solver)
            self.query_dataset = CamLandmarkDataset(query_path,mode="query",depth_solver = self.depth_solver)
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
        matcher = MatchingPipeline(
            ckpt_path="./LOGS/resnet50_MixVPR_4096_channels(1024)_rows(4).ckpt",
            device='cuda' if torch.cuda.is_available() else 'cpu'
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
        validate_results(final_pose,self.query_dataset)
        
        
        
    def prep_dataset(self,data_path:Path,dataset:str):
        self.depth_solver.img_db_process(data_path,dataset)