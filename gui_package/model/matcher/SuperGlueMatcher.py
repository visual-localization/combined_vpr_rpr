from typing import TYPE_CHECKING
from SuperGlue.models.utils import transform_image
from SuperGlue.models.matching import Matching

if TYPE_CHECKING:
    from torch import Tensor


class SuperGlueMatcher:
    def __init__(self, resize: tuple[int, int], outdoor=False, device="cuda"):
        # copied default values
        nms_radius = 4
        keypoint_threshold = 0.005
        max_keypoints = 1024

        # indoor trained on scannet
        superglue_weights = "outdoor" if outdoor else "indoor"
        sinkhorn_iterations = 20
        match_threshold = 0.2

        # Load the SuperPoint and SuperGlue models.
        print('Running inference on device "{}"'.format(device))

        config = {
            "superpoint": {
                "nms_radius": nms_radius,
                "keypoint_threshold": keypoint_threshold,
                "max_keypoints": max_keypoints,
            },
            "superglue": {
                "weights": superglue_weights,
                "sinkhorn_iterations": sinkhorn_iterations,
                "match_threshold": match_threshold,
            },
        }

        self.matching = Matching(config).eval().to(device)
        self.device = device

        print("SuperGlue model loaded")
        self.resize = resize

    def match(self, query_img: "Tensor", db_img: "Tensor"):
        """return correspondences between images (w/ path pair_path)"""
        resize = self.resize
        resize_float = True
        rot0, rot1 = 0, 0

        _, inp0, _ = transform_image(db_img, self.device, resize, rot0, resize_float)
        _, inp1, _ = transform_image(query_img, self.device, resize, rot1, resize_float)
        pred = self.matching({"image0": inp0, "image1": inp1})
        pred = {k: v[0].cpu().detach().numpy() for k, v in pred.items()}

        kpts0, kpts1 = pred["keypoints0"], pred["keypoints1"]
        matches, _ = pred["matches0"], pred["matching_scores0"]

        # Keep the matching keypoints.
        valid = matches > -1
        mkpts0 = kpts0[valid]
        mkpts1 = kpts1[matches[valid]]

        return mkpts0, mkpts1
