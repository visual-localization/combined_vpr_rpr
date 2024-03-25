from pathlib import Path
from typing import Tuple,Dict,List

import torchvision.transforms as tvf
import torch

from mixvpr_model import MatchingPipeline
from depth_dpt import DPT_DepthModel
from mapfree import FeatureDepthModel,Pose
from data import MapfreeDataset,CamLandmarkDataset,CamLandmarkDatasetPartial,GsvDatasetPartial,Pittsburgh250kSceneDataset
from validation import validate_results
from const import MAPFREE_RESIZE,CAM_RESIZE,GSV_RESIZE,PITTS_RESIZE

class RPR_Solver:
    def __init__(self, db_path:Path, query_path:Path,dataset:str="Mapfree"):
        self.depth_solver = DPT_DepthModel()
        self.pose_solver = FeatureDepthModel(feature_matching="SuperGlue",pose_solver="EssentialMatrixMetric",dataset=dataset)
        self.dataset = dataset
        
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
        elif (dataset == "Pittsburgh250k"):
            self.db_dataset = Pittsburgh250kSceneDataset(
                db_path,
                mode="db",
                depth_solver=self.depth_solver,
                resize=PITTS_RESIZE
            )
            self.query_dataset = Pittsburgh250kSceneDataset(
                query_path,
                mode="query",
                depth_solver=self.depth_solver,
                resize=PITTS_RESIZE
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
        validate_results(final_pose,self.query_dataset,self.dataset)
        
        
        
    def prep_dataset(self,data_path:Path,dataset:str,resize):
        self.depth_solver.img_db_process(data_path,dataset,resize)