import numpy as np
import cv2 as cv

from const import PNP,EMAT_RANSAC
from data import Scene
def backproject_3d(uv, depth, K):
    '''
    Backprojects 2d points given by uv coordinates into 3D using their depth values and intrinsic K
    :param uv: array [N,2]
    :param depth: array [N]
    :param K: array [3,3]
    :return: xyz: array [N,3]
    '''

    uv1 = np.concatenate([uv, np.ones((uv.shape[0], 1))], axis=1)
    xyz = depth.reshape(-1, 1) * (np.linalg.inv(K) @ uv1.T).T
    return xyz


class EssentialMatrixSolver:
    '''Obtain relative pose (up to scale) given a set of 2D-2D correspondences'''

    def __init__(self):

        # EMat RANSAC parameters
        self.ransac_pix_threshold = EMAT_RANSAC["pix_threshold"]
        self.ransac_confidence = EMAT_RANSAC["confidence"]

    def estimate_pose(self, db_pts, query_pts,db_scene, query_scene):
        # 0: db
        # 1: query
        
        R = np.full((3, 3), np.nan)
        t = np.full((3, 1), np.nan)
        if len(db_pts) < 5:
            return R, t, 0
        K0 = db_scene["intrinsics_matrix"]
        K1 = query_scene["intrinsics_matrix"]
        # normalize keypoints
        db_pts = (db_pts - K0[[0, 1], [2, 2]][None]) / K0[[0, 1], [0, 1]][None]
        query_pts = (query_pts - K1[[0, 1], [2, 2]][None]) / K1[[0, 1], [0, 1]][None]
        # normalize ransac threshold
        ransac_thr = self.ransac_pix_threshold / np.mean([K0[0, 0], K1[1, 1], K0[1, 1], K1[0, 0]])
        # compute pose with OpenCV
        E, mask = cv.findEssentialMat(
            db_pts, query_pts, np.eye(3),
            threshold=ransac_thr, prob=self.ransac_confidence, method=cv.USAC_MAGSAC)
        self.mask = mask
        if E is None:
            return R, t, 0
        # recover pose from E
        best_num_inliers = 0
        ret = R, t, 0
        for _E in np.split(E, len(E) / 3):
            n, R, t, _ = cv.recoverPose(_E, db_pts, query_pts, np.eye(3), 1e9, mask=mask)
            if n > best_num_inliers:
                best_num_inliers = n
                ret = (R, t[:, 0], n)
        return ret


class EssentialMatrixMetricSolverMEAN(EssentialMatrixSolver):
    '''Obtains relative pose with scale using E-Mat decomposition and depth values at inlier correspondences'''

    def __init__(self):
        super().__init__()

    def estimate_pose(self, db_pts, query_pts,db_scene, query_scene):
        '''Estimates metric translation vector using by back-projecting E-mat inliers to 3D using depthmaps.
        The metric translation vector can be obtained by looking at the residual vector (projected to the translation vector direction).
        In this version, each 3D-3D correspondence gives an optimal scale for the translation vector. 
        We simply aggregate them by averaging them.
        '''

        # get pose up to scale
        R, t, inliers = super().estimate_pose(
            db_pts=db_pts, db_scene=db_scene,
            query_pts=query_pts, query_scene=query_scene
        )
        if inliers == 0:
            return R, t, inliers

        # backproject E-mat inliers at each camera
        K0 = db_scene["intrinsics_matrix"]
        K1 = query_scene["intrinsics_matrix"]
        mask = self.mask.ravel() == 1        # get E-mat inlier mask from super class
        inliers_db_pts = np.int32(db_pts[mask])
        inliers_query_pts = np.int32(query_pts[mask])
        depth_inliers_db = db_scene["depth"][inliers_db_pts[:, 1], inliers_db_pts[:, 0]].numpy()
        depth_inliers_query = query_scene["depth"][inliers_query_pts[:, 1], inliers_query_pts[:, 0]].numpy()
        # check for valid depth
        valid = (depth_inliers_db > 0) * (depth_inliers_query > 0)
        if valid.sum() < 1:
            R = np.full((3, 3), np.nan)
            t = np.full((3, 1), np.nan)
            inliers = 0
            return R, t, inliers
        xyz0 = backproject_3d(inliers_db_pts[valid], depth_inliers_db[valid], K0)
        xyz1 = backproject_3d(inliers_query_pts[valid], depth_inliers_query[valid], K1)

        # rotate xyz0 to xyz1 CS (so that axes are parallel)
        xyz0 = (R @ xyz0.T).T

        # get average point for each camera
        pmean_db = np.mean(xyz0, axis=0)
        pmean_query = np.mean(xyz1, axis=0)

        # find scale as the 'length' of the translation vector that minimises the 3D distance between projected points from 0 and the corresponding points in 1
        scale = np.dot(pmean_query - pmean_db, t)
        t_metric = scale * t
        t_metric = t_metric.reshape(3, 1)

        return R, t_metric, inliers


class EssentialMatrixMetricSolver(EssentialMatrixSolver):
    '''
        Obtains relative pose with scale using E-Mat decomposition and RANSAC for scale based on depth values at inlier correspondences.
        The scale of the translation vector is obtained using RANSAC over the possible scales recovered from 3D-3D correspondences.
    '''

    def __init__(self):
        super().__init__()
        self.ransac_scale_threshold = EMAT_RANSAC["scale_threshold"]

    def estimate_pose(self, db_pts, query_pts,db_scene, query_scene):
        '''Estimates metric translation vector using by back-projecting E-mat inliers to 3D using depthmaps.
        '''

        # get pose up to scale
        R, t, inliers = super().estimate_pose(
            db_pts=db_pts, db_scene=db_scene,
            query_pts=query_pts, query_scene=query_scene
        )
        if inliers == 0:
            return R, t, inliers

        # backproject E-mat inliers at each camera
        K0 = db_scene["intrinsics_matrix"]
        K1 = query_scene["intrinsics_matrix"]
        mask = self.mask.ravel() == 1        # get E-mat inlier mask from super class
        inliers_db_pts = np.int32(db_pts[mask])
        inliers_query_pts = np.int32(query_pts[mask])
        depth_inliers_db = db_scene["depth"][inliers_db_pts[:, 1], inliers_db_pts[:, 0]].numpy()
        depth_inliers_query = query_scene["depth"][inliers_query_pts[:, 1], inliers_query_pts[:, 0]].numpy()

        # check for valid depth
        valid = (depth_inliers_db > 0) * (depth_inliers_query > 0)
        if valid.sum() < 1:
            R = np.full((3, 3), np.nan)
            t = np.full((3, 1), np.nan)
            inliers = 0
            return R, t, inliers
        xyz0 = backproject_3d(inliers_db_pts[valid], depth_inliers_db[valid], K0)
        xyz1 = backproject_3d(inliers_query_pts[valid], depth_inliers_query[valid], K1)

        # rotate xyz0 to xyz1 CS (so that axes are parallel)
        xyz0 = (R @ xyz0.T).T

        # get individual scales (for each 3D-3D correspondence)
        scale = np.dot(xyz1 - xyz0, t.reshape(3, 1))  # [N, 1]

        # RANSAC loop
        best_inliers = 0
        best_scale = None
        for scale_hyp in scale:
            inliers_hyp = (np.abs(scale - scale_hyp) < self.ransac_scale_threshold).sum().item()
            if inliers_hyp > best_inliers:
                best_scale = scale_hyp
                best_inliers = inliers_hyp

        # Output results
        t_metric = best_scale * t
        t_metric = t_metric.reshape(3, 1)

        return R, t_metric, best_inliers
