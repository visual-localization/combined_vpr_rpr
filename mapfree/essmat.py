from typing import List,Dict
from tqdm import tqdm

import numpy as np

from data import SceneDataset,Scene
from .pose_solver import EssentialMatrixMetricSolver
from .feature_matching import SuperGlue_matcher
from .converter import convert_pose

class Pose:
    R: np.ndarray
    t: np.ndarray
    inliers: int
    anchor: str
    def __init__(
        self,
        R: np.ndarray,
        t: np.ndarray,
        inliers:int,
        anchor:str
    ):
        self.R = R
        self.t = t
        self.inliers = inliers
        self.anchor = anchor
    def __str__(self):
      print(self.R)
      print(self.t)
      print(self.inliers)
      print(self.anchor)
      return ""    

class FeatureDepthModel:
    def __init__(
        self,
        feature_matching:str,
        pose_solver:str,
    ):
        if(feature_matching=="SuperGlue"):
            self.feature_matching = SuperGlue_matcher(outdoor=True)
        else:
            assert False, "Feature Matching Model not found"
        
        if(pose_solver == "EssentialMatrixMetric"):
            self.pose_solver = EssentialMatrixMetricSolver()
        else:
            assert False, "Pose Solver Model not found"
    
    def process_top_k(
        self,
        db_dataset:SceneDataset,
        query_dataset:SceneDataset,
        top_k:Dict[str,List[str]]
    )->Dict[str,Pose]:
        final_pose = {}
        for query_key in tqdm(top_k.keys()):
            res = [self.process_pair(query_dataset[query_key][0],db_dataset[db_key][0]) for db_key in top_k[query_key]]
            max_pose = max(res, key=lambda item:item.inliers)
            final_pose[query_key] = max_pose
        return final_pose
    
    def process_pair(
        self,
        query_img: Scene,
        db_img: Scene
    )->Pose:
        db_pts, query_pts = self.feature_matching.match(query_img["image"],db_img["image"])
        R, t, inliers = self.pose_solver.estimate_pose(
            db_pts = db_pts, db_scene = db_img,
            query_pts = query_pts, query_scene = query_img
        )

        if(inliers>0):
          R_final, t_final = convert_pose(R,t,db_img)
        else:
          R_final, t_final = R,t

        return Pose(
            R=R_final,
            t=t_final,
            inliers=inliers,
            anchor=db_img["name"]
        )

        
        
        