import numpy as np
import torch
from transforms3d.quaternions import mat2quat, qinverse, qmult, rotate_vector, quat2mat

from typing import Tuple

from data import Scene


class FrustumDifferennce:
    @staticmethod
    def get_frustum_difference(origin_img:Scene,target_img:Scene)->float:
        point_map,depth_list = FrustumDifferennce.sample_point(
            origin_height = origin_img["height"],
            origin_width = origin_img["width"],
            origin_depth = origin_img["depth"]
        )
        
        # Generate 3D points relative to the origin camera coordinate system
        origin_points = FrustumDifferennce.backproject_3d(point_map,depth_list,origin_img["intrinsics_matrix"])
        
        # Project 3D points to world coordinate system
        points_world = FrustumDifferennce.proj_to_world_coord(
            origin_points = origin_points,
            origin_rotation = origin_img["rotation"],
            origin_translation = origin_img["translation"]
        )
        
        points_target_2D = FrustumDifferennce.proj_to_target(
            world_points=points_world,
            target_translation=target_img["translation"],
            target_rotation=target_img["rotation"],
            target_intrinsics=target_img["intrinsics_matrix"]
        )
        
        filter = []
        for point in points_target_2D:
            filter.append(point[0]>0 and point[0]<target_img["width"] and point[1]>0 and point[1]<target_img["height"])
        
        return point_map[filter].shape[0]/len(filter)
        
    
    @staticmethod
    def sample_point(origin_height:float, origin_width:float, origin_depth:torch.tensor, interval:int=10)->Tuple[np.ndarray, np.ndarray]:
        '''
        Sampling from the origin image and assign depth value to each sampled pixel
        :param origin_height: float
        :param origin_width: float
        :param origin_depth: np.ndarray (H,W)
        :return: xyz: array [N,3]
        '''
        point_map = []
        depth_list = []
        for y in range(0,int(origin_height),interval):
            for x in range(0,int(origin_width),interval):
                if origin_depth[y][x]!=0 and origin_depth[y][x]!=65535:
                    point_map.append([x,y])
                    depth_list.append(origin_depth[y][x].item())
        point_map = np.array(point_map)
        depth_list = np.array(depth_list)
        return point_map,depth_list
    
    @staticmethod
    def backproject_3d(point_map:np.ndarray, depth_list:np.ndarray, K:np.ndarray)->np.ndarray:
        '''
        Backprojects 2d points given by uv coordinates into 3D using their depth values and intrinsic K
        :param point_map: array [N,2]
        :param depth_list: array [N]
        :param K: array [3,3]
        :return: xyz: array [N,3]
        '''
        point_map_1 = np.concatenate([point_map, np.ones((point_map.shape[0], 1))], axis=1)
        points3D = depth_list.reshape(-1, 1) * (np.linalg.inv(K) @ point_map_1.T).T
        return points3D
    
    @staticmethod
    def proj_to_world_coord(origin_points:np.ndarray, origin_rotation:np.ndarray, origin_translation:np.ndarray)->np.ndarray:
        '''
        Project 3D points sampled from origin image to world coordinate system
        :param origin_points: array [N,3]
        :param origin_rotation: array [4]
        :param origin_translation: array [3]
        :return: abs_point: array [N,3]
        '''
        # mat1 = quat2mat(origin_rotation)
        # abs_cam_origin = rotate_vector(-origin_translation, qinverse(origin_rotation))
        # abs_point = (mat1.T@origin_points.T).T+abs_cam_origin
        # return abs_point
        mat1 = quat2mat(origin_rotation)
        abs_point = (mat1@origin_points.T).T + origin_translation
        return abs_point
    
    @staticmethod
    def proj_to_target(
        world_points:np.ndarray,
        target_translation: np.ndarray,
        target_rotation: np.ndarray,
        target_intrinsics: np.ndarray
    )->np.ndarray:
        '''
        Project 3D points in world coordinate to target camera 2D coordinate system
        :param world_points: array [N,3]
        :param target_translation: array [4]
        :param target_rotation: array [3]
        :param target_intrinsics: array [3,3]
        :return: target_points: array [N,2]
        '''
        # mat2 = quat2mat(target_rotation) 
        # abs_cam_target = rotate_vector(-target_translation, qinverse(target_rotation))
        # point_in_query = mat2@(world_points-abs_cam_target).T
        # point_in_query = target_intrinsics@point_in_query
        # temp = point_in_query.T
        # target_points = temp/temp[:,2].reshape(-1, 1)
        # return target_points[:,:2]
        mat2 = quat2mat(target_rotation) 
        point_in_query = mat2.T@(world_points-target_translation).T
        point_in_query = target_intrinsics@point_in_query
        temp = point_in_query.T
        target_points = temp/temp[:,2].reshape(-1, 1)
        return target_points[:,:2]

class AngleDifference:
    @staticmethod
    def relative_q(q1:np.ndarray, q2:np.ndarray)->float:
        '''
        The difference in angle(degree) between two camera rotation
        :param q1: array [4]
        :param q2: array [4]
        :return: target_points: array [N,2]
        '''
        mat1 = quat2mat(q1)
        mat2 = quat2mat(q2)
        return np.arccos((np.trace(mat1.T@mat2)-1)/2)*(180/np.pi)