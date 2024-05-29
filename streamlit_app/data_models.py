from typing import TYPE_CHECKING
from dataclasses import dataclass
from pathlib import Path
from numpy import array

if TYPE_CHECKING:
    from gui_package.data.scene import Scene
    from gui_package.model import Pose
    import numpy as np


@dataclass
class SceneDTOS:
    name: str
    image: Path
    depth: Path
    intrinsics: "np.ndarray"
    rotation: "np.ndarray"
    translation: "np.ndarray"
    width: float
    height: float

    def to_dict(self) -> dict:
        return dict(
            name=self.name,
            image=self.image.as_posix(),
            depth=self.depth.as_posix(),
            intrinsics=self.intrinsics.tolist(),
            rotation=self.rotation.tolist(),
            translation=self.translation.tolist(),
            width=self.width,
            height=self.height,
        )

    @staticmethod
    def from_dict(data: dict) -> "SceneDTOS":
        return SceneDTOS(
            name=data["name"],
            image=Path(data["image"]),
            depth=Path(data["depth"]),
            intrinsics=array(data["intrinsics"]),
            rotation=array(data["rotation"]),
            translation=array(data["translation"]),
            width=data["width"],
            height=data["height"],
        )

    @staticmethod
    def from_scene(scene: "Scene") -> "SceneDTOS":
        return SceneDTOS(
            name=scene.name,
            image=scene.image,
            depth=scene.depth,
            intrinsics=scene.intrinsics,
            rotation=scene.rotation,
            translation=scene.translation,
            width=scene.width,
            height=scene.height,
        )


@dataclass
class PoseDTOS:
    r: "np.ndarray"
    t: "np.ndarray"
    inliers: int
    anchor_name: str

    def to_dict(self) -> dict:
        return dict(
            r=self.r.tolist(),
            t=self.t.tolist(),
            inliers=self.inliers,
            anchor_name=self.anchor_name,
        )

    @staticmethod
    def from_dict(data: dict) -> "PoseDTOS":
        return PoseDTOS(
            r=array(data["r"]),
            t=array(data["t"]),
            inliers=data["inliers"],
            anchor_name=data["anchor_name"],
        )

    @staticmethod
    def from_pose(pose: "Pose") -> "PoseDTOS":
        return PoseDTOS(
            r=pose.R,
            t=pose.t,
            inliers=pose.inliers,
            anchor_name=pose.anchor,
        )


@dataclass
class ImageDTOS:
    name: str
    media_type: str
    size: int
    data: bytes

    def to_dict(self) -> dict:
        return dict(
            name=self.name, media_type=self.media_type, size=self.size, data=self.data
        )

    @staticmethod
    def from_dict(data: dict) -> "ImageDTOS":
        return ImageDTOS(
            name=data["name"],
            media_type=data["media_type"],
            size=data["size"],
            data=data["data"],
        )


@dataclass
class SampleDTOS:
    scene: SceneDTOS
    image: ImageDTOS

    def to_dict(self) -> dict:
        return dict(scene=self.scene.to_dict(), image=self.image.to_dict())

    @staticmethod
    def from_dict(data: dict) -> "SampleDTOS":
        return SampleDTOS(
            scene=SceneDTOS.from_dict(data["scene"]),
            image=ImageDTOS.from_dict(data["image"]),
        )


@dataclass
class QueryResponseDTOS:
    query: SceneDTOS
    retrieved_scenes: list[SceneDTOS]
    reranking_indices: list[int]
    pose: PoseDTOS

    def to_dict(self) -> dict:
        return dict(
            query=self.query.to_dict(),
            retrived_scenes=[scene.to_dict() for scene in self.retrieved_scenes],
            reranking_indices=self.reranking_indices,
            pose=self.pose.to_dict(),
        )

    @staticmethod
    def from_dict(data: dict) -> "QueryResponseDTOS":
        return QueryResponseDTOS(
            query=SceneDTOS.from_dict(data["query"]),
            retrieved_scenes=[
                SceneDTOS.from_dict(scene) for scene in data["retrived_scenes"]
            ],
            reranking_indices=data["reranking_indices"],
            pose=PoseDTOS.from_dict(data["pose"]),
        )
