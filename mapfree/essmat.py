from typing import List,Dict
from tqdm import tqdm

import numpy as np

from data import SceneDataset,Scene
from .pose_solver import EssentialMatrixMetricSolver
from .feature_matching import SuperGlue_matcher
from .converter import convert_pose
from utils import weightedAverageQuaternions,quat_angle_error
from const import MAPFREE_RESIZE, CAM_RESIZE, GSV_RESIZE, PITTS_RESIZE, TRANS_THRESHOLD,ROT_THRESHOLD

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
        dataset:str,
        pose_mode:str
    ):
        assert pose_mode in ["max","weighted"], "Please specify a value from max or weighted"
        self.pose_mode = pose_mode
        resize = None
        if(dataset == "Mapfree"):
            resize = MAPFREE_RESIZE
        elif("CamLandmark" in dataset):
            resize = CAM_RESIZE
        elif("GSV" in dataset):
            resize = GSV_RESIZE
        elif("Pittsburgh" in dataset):
            resize = PITTS_RESIZE
        else:
            raise NotImplementedError("No resize value for this dataset")
        if(feature_matching=="SuperGlue"):
            self.feature_matching = SuperGlue_matcher(outdoor=True,resize=resize)
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
            res = [self.process_pair(
                query_dataset[query_key][0],
                db_dataset[db_key][0]
            ) for db_key in top_k[query_key]]
            final_res = None
            if(self.pose_mode == "max"):
                final_res = max(res, key=lambda item:item.inliers)
            elif(self.pose_mode=="weighted"):
                filtered_pose = list(filter(lambda pose:pose.inliers>0,res))
                if len(filtered_pose) == 0:
                    final_res = Pose(
                        R = np.full((3, 3), np.nan),
                        t = np.full((3, 1), np.nan),
                        inliers = 0,
                        anchor=None
                    )
                else:
                    cont = True
                    while cont:
                        cont = False
                        sum_inlier = sum(pose.inliers for pose in filtered_pose)
                        t_final = sum(pose.t*(pose.inliers/sum_inlier) for pose in filtered_pose)
                        inliers_final = sum(pose.inliers*(pose.inliers/sum_inlier) for pose in filtered_pose)
                        R_final = weightedAverageQuaternions(
                            np.array((pose.R for pose in filtered_pose)),
                            list((pose.inliers/sum_inlier for pose in filtered_pose))
                        )
                        # for pose in filtered_pose:
                        #     trans_err = np.linalg.norm(pose.t - t_final)
                        #     rot_err = quat_angle_error(pose.R - R_final)[0][0]
                        #     if(trans_err<=TRANS_THRESHOLD and rot_err<ROT_THRESHOLD):
                        #         cont = True
                        #         filtered_pose.remove(pose)
                    final_res = Pose(
                        R = R_final,
                        t = t_final,
                        inliers = inliers_final,
                        anchor=None
                    )
            final_pose[query_key] = final_res
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

class NaivePoseModel:
    def __init__(
        self
    ):
        pass
    
    def process_top_k(
        self,
        db_dataset:SceneDataset,
        query_dataset:SceneDataset,
        top_k:Dict[str,List[str]]
    )->Dict[str,Pose]:
        final_pose = {}
        for query_key in tqdm(top_k.keys()):
            final_pose[query_key] = self.process_pair(
                query_dataset[query_key][0],
                db_dataset[top_k[query_key][0]][0]
            )
        return final_pose
    
    def process_pair(
        self,
        query_img: Scene,
        db_img: Scene
    )->Pose:
        R_final = db_img["rotation"]
        t_final = db_img["translation"]
        return Pose(
            R=R_final,
            t=t_final,
            inliers=100,
            anchor=db_img["name"]
        )