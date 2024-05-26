from typing import NamedTuple, Literal, TYPE_CHECKING
from functools import reduce
from numpy import array, zeros, outer, real, linalg
from .pose_solver import estimate_pose_with_depth
from .matcher import SuperGlueMatcher
from transforms3d.quaternions import mat2quat, qinverse, qmult, rotate_vector

if TYPE_CHECKING:
    from ..data import Scene, SceneDataset
    from numpy import ndarray
    from torch import Tensor


def convert_pose(r: "ndarray", t: "ndarray", scene: "Scene"):
    q1 = scene.rotation
    t1 = scene.translation

    q12 = mat2quat(r).reshape(-1)
    t12 = t.flatten()

    q2 = qmult(q1, qinverse(q12))
    t2 = rotate_vector(-t12, qinverse(q2)) + t1
    return q2, t2


def weighted_average_quaternions(q: "ndarray", w: list[float]):
    # Number of quaternions to average
    M = q.shape[0]
    A = zeros(shape=(4, 4))
    weight_sum = 0

    for i in range(0, M):
        qq = q[i, :]
        A = w[i] * outer(qq, qq) + A
        weight_sum += w[i]

    # scale
    A = (1.0 / weight_sum) * A

    # compute eigenvalues and -vectors
    eigen_values, eigen_vectors = linalg.eig(A)

    # Sort by largest eigenvalue
    eigen_vectors = eigen_vectors[:, eigen_values.argsort()[::-1]]

    # return the real part of the largest eigenvector (has only real part)
    return real(eigen_vectors[:, 0])


class Pose(NamedTuple):
    R: "ndarray"
    t: "ndarray"
    inliers: int
    anchor: str

    def __str__(self):
        return f"R: {self.R}\nt: {self.t}\ninliers: {self.inliers}\nanchor: {self.anchor}\n"


class EssMatPoseRegressionModel:
    def __init__(
        self,
        resize: tuple[int, int],
        query_dataset: "SceneDataset",
        db_dataset: "SceneDataset",
        device="cuda",
    ):
        self.feature_matching = SuperGlueMatcher(resize, True, device)
        self.query_dataset = query_dataset
        self.db_dataset = db_dataset

    def process_top_k(
        self,
        top_k: list["Scene"],
        query: "Scene",
        pose_mode: Literal["max", "weighted"] = "max",
    ) -> Pose | None:
        db_images = [
            depth
            for depth in self.db_dataset.read_images(scene.image for scene in top_k)
        ]

        db_depth_maps = [
            depth_map
            for depth_map in self.db_dataset.read_depth_maps(
                scene.depth for scene in top_k
            )
        ]

        query_image = self.query_dataset.read_image(query.image)
        query_depth_map = self.query_dataset.read_depth_map(query.depth)

        if pose_mode == "max":
            generator = (
                self.process_pair(
                    query,
                    query_image,
                    query_depth_map,
                    db_scene,
                    db_image,
                    db_depth_map,
                )
                for (db_scene, db_image, db_depth_map) in zip(
                    top_k, db_images, db_depth_maps
                )
            )

            return max(
                generator,
                key=lambda item: item.inliers,
            )

        def reduce_positive_pairs(
            acc: list[Pose], item: tuple["Scene", "Tensor", "Tensor"]
        ) -> list[Pose]:
            (db_scene, db_image, db_depth_map) = item

            pose = self.process_pair(
                query, query_image, query_depth_map, db_scene, db_image, db_depth_map
            )

            if pose.inliers <= 0:
                return acc

            acc.append(pose)
            return acc

        filtered_pose = reduce(
            reduce_positive_pairs, zip(top_k, db_images, db_depth_maps), []
        )

        if len(filtered_pose) == 0:
            return None

        sum_inlier = sum(pose.inliers for pose in filtered_pose)

        t_final = sum(pose.t * (pose.inliers / sum_inlier) for pose in filtered_pose)
        inliers_final = sum(
            pose.inliers * (pose.inliers / sum_inlier) for pose in filtered_pose
        )

        r_list = array([pose.R for pose in filtered_pose])
        r_final = weighted_average_quaternions(
            r_list,
            list((pose.inliers / sum_inlier for pose in filtered_pose)),
        )

        final_res = Pose(r_final, t_final, inliers_final, None)

        return final_res

    def process_pair(
        self,
        query_scene: "Scene",
        query_image: "Tensor",
        query_depth_map: "Tensor",
        db_scene: "Scene",
        db_image: "Tensor",
        db_depth_map: "Tensor",
    ) -> Pose:
        db_pts, query_pts = self.feature_matching.match(query_image, db_image)

        r, t, inliers, mask = estimate_pose_with_depth(
            db_pts,
            query_pts,
            db_scene.intrinsics,
            query_scene.intrinsics,
            db_depth_map,
            query_depth_map,
        )

        if inliers <= 0:
            return Pose(r, t, inliers, db_scene.name)

        r_final, t_final = convert_pose(r, t, db_scene)
        return Pose(r_final, t_final, inliers, db_scene.name)
