from typing import TYPE_CHECKING
import numpy as np
import cv2 as cv

if TYPE_CHECKING:
    from torch import Tensor


def backproject_3d(uv: np.ndarray, depth: np.ndarray, k: np.ndarray):
    """
    Backprojects 2d points given by uv coordinates into 3D using their depth values and intrinsic K
    :param uv: array [N,2]
    :param depth: array [N]
    :param K: array [3,3]
    :return: xyz: array [N,3]
    """

    uv1 = np.concatenate([uv, np.ones((uv.shape[0], 1))], axis=1)
    xyz = depth.reshape(-1, 1) * (np.linalg.inv(k) @ uv1.T).T
    return xyz


def estimate_pose(
    db_pts,
    query_pts,
    db_intrinsics: np.ndarray,
    query_intrinsics: np.ndarray,
    ransac_pix_threshold: float = 2.0,
    ransac_confidence: float = 0.9999,
):
    r = np.full((3, 3), np.nan)
    t = np.full((3, 1), np.nan)

    if len(db_pts) < 5:
        return r, t, 0, None

    K0 = db_intrinsics
    K1 = query_intrinsics

    # normalize keypoints
    db_pts = (db_pts - K0[[0, 1], [2, 2]][None]) / K0[[0, 1], [0, 1]][None]
    query_pts = (query_pts - K1[[0, 1], [2, 2]][None]) / K1[[0, 1], [0, 1]][None]

    # normalize ransac threshold
    ransac_thr = ransac_pix_threshold / np.mean(
        [K0[0, 0], K1[1, 1], K0[1, 1], K1[0, 0]]
    )

    # compute pose with OpenCV
    e, mask = cv.findEssentialMat(
        db_pts,
        query_pts,
        np.eye(3),
        threshold=ransac_thr,
        prob=ransac_confidence,
        method=cv.USAC_MAGSAC,
    )

    if e is None:
        return r, t, 0, mask

    # recover pose from E
    best_num_inliers = 0
    ret = r, t, 0

    for e in np.split(e, len(e) / 3):
        n, r, t, _ = cv.recoverPose(e, db_pts, query_pts, np.eye(3), 1e9, mask=mask)
        if n > best_num_inliers:
            best_num_inliers = n
            ret = (r, t[:, 0], n)

    return (*ret, mask)


def estimate_pose_with_depth(
    db_pts,
    query_pts,
    db_intrinsics: np.ndarray,
    query_intrinsics: np.ndarray,
    db_depth: "Tensor",
    query_depth: "Tensor",
    ransac_pix_threshold: float = 2.0,
    ransac_confidence: float = 0.9999,
    ransac_scale_threshold: float = 0.1,
):
    r, t, inliers, mask = estimate_pose(
        db_pts,
        query_pts,
        db_intrinsics,
        query_intrinsics,
        ransac_pix_threshold,
        ransac_confidence,
    )

    if inliers == 0:
        return r, t, inliers, mask

    # backproject E-mat inliers at each camera
    K0 = db_intrinsics
    K1 = query_intrinsics

    # get E-mat inlier mask from base handler
    mask = mask.ravel() == 1

    inliers_db_pts = np.int32(db_pts[mask])
    inliers_query_pts = np.int32(query_pts[mask])

    depth_inliers_db = db_depth[inliers_db_pts[:, 1], inliers_db_pts[:, 0]].numpy()
    depth_inliers_query = query_depth[
        inliers_query_pts[:, 1], inliers_query_pts[:, 0]
    ].numpy()

    # check for valid depth
    valid = (depth_inliers_db > 0) * (depth_inliers_query > 0)

    if valid.sum() < 1:
        r = np.full((3, 3), np.nan)
        t = np.full((3, 1), np.nan)
        inliers = 0
        return r, t, inliers, mask

    xyz0 = backproject_3d(inliers_db_pts[valid], depth_inliers_db[valid], K0)
    xyz1 = backproject_3d(inliers_query_pts[valid], depth_inliers_query[valid], K1)

    # rotate xyz0 to xyz1 CS (so that axes are parallel)
    xyz0 = (r @ xyz0.T).T

    # get individual scales (for each 3D-3D correspondence)
    scale = np.dot(xyz1 - xyz0, t.reshape(3, 1))  # [N, 1]

    # RANSAC loop
    best_inliers = 0
    best_scale = None
    for scale_hyp in scale:
        inliers_hyp = (np.abs(scale - scale_hyp) < ransac_scale_threshold).sum().item()
        if inliers_hyp > best_inliers:
            best_scale = scale_hyp
            best_inliers = inliers_hyp

    # Output results
    t_metric = best_scale * t
    t_metric = t_metric.reshape(3, 1)

    return r, t_metric, best_inliers, mask
