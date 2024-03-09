import torch
import numpy as np
import cv2

from SuperGlue.models.utils import transform_image, read_image
from SuperGlue.models.matching import Matching
from const import MAPFREE_RESIZE

class SuperGlue_matcher:
    def __init__(self, resize=MAPFREE_RESIZE, outdoor=False):
        # copied default values
        nms_radius = 4
        keypoint_threshold = 0.005
        max_keypoints = 1024

        superglue_weights = 'outdoor' if outdoor else 'indoor'  # indoor trained on scannet
        sinkhorn_iterations = 20
        match_threshold = 0.2

        # Load the SuperPoint and SuperGlue models.
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print('Running inference on device \"{}\"'.format(device))
        config = {
            'superpoint': {
                'nms_radius': nms_radius,
                'keypoint_threshold': keypoint_threshold,
                'max_keypoints': max_keypoints
            },
            'superglue': {
                'weights': superglue_weights,
                'sinkhorn_iterations': sinkhorn_iterations,
                'match_threshold': match_threshold,
            }
        }
        self.matching = Matching(config).eval().to(device)
        self.device = device
        print('SuperGlue model loaded')
        self.resize = resize

    def match(self, query_img:torch.tensor, db_img:torch.tensor):
        '''return correspondences between images (w/ path pair_path)'''
        resize = self.resize
        resize_float = True
        rot0, rot1 = 0, 0

        image0, inp0, scales0 = transform_image(
            db_img, self.device, resize, rot0, resize_float)
        image1, inp1, scales1 = transform_image(
            query_img, self.device, resize, rot1, resize_float)
        pred = self.matching({'image0': inp0, 'image1': inp1})
        pred = {k: v[0].cpu().detach().numpy() for k, v in pred.items()}

        kpts0, kpts1 = pred['keypoints0'], pred['keypoints1']
        matches, conf = pred['matches0'], pred['matching_scores0']

        # Keep the matching keypoints.
        valid = matches > -1
        mkpts0 = kpts0[valid]
        mkpts1 = kpts1[matches[valid]]

        return mkpts0, mkpts1

    def debug_match(self, query_path:str, db_path:str):
        resize = self.resize
        resize_float = True
        rot0, rot1 = 0, 0

        image0, inp0, scales0 = read_image(
            query_path, self.device, resize, rot0, resize_float)
        image1, inp1, scales1 = read_image(
            db_path, self.device, resize, rot1, resize_float)
        pred = self.matching({'image0': inp0, 'image1': inp1})
        pred = {k: v[0].cpu().detach().numpy() for k, v in pred.items()}
        kpts0, kpts1 = pred['keypoints0'], pred['keypoints1']
        matches, conf = pred['matches0'], pred['matching_scores0']

        # Keep the matching keypoints.
        valid = matches > -1
        mkpts0 = kpts0[valid]
        mkpts1 = kpts1[matches[valid]]

        return mkpts0, mkpts1
