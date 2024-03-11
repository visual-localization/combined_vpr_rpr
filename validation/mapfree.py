import argparse
from collections import defaultdict
from pathlib import Path
from zipfile import ZipFile
from io import TextIOWrapper
import json
import logging
from typing import Dict

import numpy as np


from data import SceneDataset,Scene
from validation.utils import precision_recall
from validation.metrics import MetricManager, Inputs
import validation.config as config
from mapfree import Pose
from utils import extract_scene_label,convert_world2cam_to_cam2world



def compute_scene_metrics(estimated_pose:Pose,query_scene:Scene):
    metric_manager = MetricManager()

    # failures encode how many images that match to the wrong database image
    failures = 0
    
    # Results encoded as dict
    # key: metric name; value: list of values (one per frame).
    # e.g. results['t_err'] = [1.2, 0.3, 0.5, ...]
    results = defaultdict(list)

    # compute metrics per frame
    if extract_scene_label(estimated_pose.anchor) != extract_scene_label(query_scene["name"]):
        failures += 1
    else:
        q_est, t_est = estimated_pose.R, estimated_pose.t
        q_gt, t_gt = query_scene["rotation"],query_scene["translation"]
        inputs = Inputs(
            q_gt=q_gt, t_gt=t_gt,
            q_est=q_est,t_est=t_est,
            confidence=estimated_pose.inliers,
            K = query_scene["intrinsics_matrix"],
            W = query_scene["width"], H = query_scene["height"]
        )
        metric_manager(inputs,results)

    return results, failures


def aggregate_results(all_results, all_failures):
    # aggregate metrics
    median_metrics = defaultdict(list)
    all_metrics = defaultdict(list)
    for scene_results in all_results.values():
        for metric, values in scene_results.items():
            median_metrics[metric].append(np.median(values))
            all_metrics[metric].extend(values)
    all_metrics = {k: np.array(v) for k, v in all_metrics.items()}
    assert all([v.ndim == 1 for v in all_metrics.values()]
               ), 'invalid metrics shape'
    # compute avg median metrics
    avg_median_metrics = {metric: np.mean(
        values) for metric, values in median_metrics.items()}

    # compute precision/AUC for pose error and reprojection errors
    accepted_poses = (all_metrics['trans_err'] < config.t_threshold) * \
        (all_metrics['rot_err'] < config.R_threshold)
    accepted_vcre = all_metrics['reproj_err'] < config.vcre_threshold
    total_samples = len(next(iter(all_metrics.values()))) + all_failures

    prec_pose = np.sum(accepted_poses) / total_samples
    prec_vcre = np.sum(accepted_vcre) / total_samples

    # compute AUC for pose and VCRE
    _, _, auc_pose = precision_recall(
        inliers=all_metrics['confidence'], tp=accepted_poses, failures=all_failures)
    _, _, auc_vcre = precision_recall(
        inliers=all_metrics['confidence'], tp=accepted_vcre, failures=all_failures)

    # output metrics
    output_metrics = dict()
    output_metrics['Average Median Translation Error'] = avg_median_metrics['trans_err']
    output_metrics['Average Median Rotation Error'] = avg_median_metrics['rot_err']
    output_metrics['Average Median Reprojection Error'] = avg_median_metrics['reproj_err']
    output_metrics[f'Precision @ Pose Error < ({config.t_threshold*100}cm, {config.R_threshold}deg)'] = prec_pose
    output_metrics[f'AUC @ Pose Error < ({config.t_threshold*100}cm, {config.R_threshold}deg)'] = auc_pose
    output_metrics[f'Precision @ VCRE < {config.vcre_threshold}px'] = prec_vcre
    output_metrics[f'AUC @ VCRE < {config.vcre_threshold}px'] = auc_vcre
    output_metrics[f'Estimates for % of frames'] = len(all_metrics['trans_err']) / total_samples
    return output_metrics


def count_unexpected_scenes(scenes: tuple, submission_zip: ZipFile):
    submission_scenes = [fname[5:-4]
                         for fname in submission_zip.namelist() if fname.startswith("pose_")]
    return len(set(submission_scenes) - set(scenes))


def validate_results(
    final_pose:Dict[str,Pose],
    query_dataset:SceneDataset
):
    all_results = dict()
    all_failures = 0
    
    for scene_name in final_pose.keys():
        metrics, failures = compute_scene_metrics(
            final_pose[scene_name],
            query_dataset[scene_name][0]
        )
        if(failures==1):
          all_failures += failures
        else:
          all_results[scene_name] = metrics
        
    
    print(f"VPR success rate: {1-(all_failures/len(final_pose))}")
    output_metrics = aggregate_results(all_results, all_failures)
    output_json = json.dumps(output_metrics, indent=2)
    print(output_json)
