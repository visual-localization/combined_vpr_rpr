from pathlib import Path
from typing import Tuple, Union, NamedTuple, Iterable, Generator, TYPE_CHECKING
from torch.utils.data import Dataset

if TYPE_CHECKING:
    from numpy import ndarray
    from torch import Tensor


class Scene(NamedTuple):
    """A class to represent a Scene.

    `rotation` and `translation` represent the absolute pose of the image.

    Attributes:
        name (str): The unique name of the image.
        image (Path): The path to the image.
        depth (Path): The path to the image's depth map.
        intrinsics ("ndarray"): The intrinsic matrix of the camera. E.g. (3, 3)
        rotation ("ndarray"): The rotation matrix of the camera. E.g. [4]: q1, q2, q3, q4
        translation ("ndarray"): The translation matrix of the camera. E.g. [3]: x, y, z
        width (float): The width of the image and depth map.
        height (float): The height of the image and depth map.
    """

    name: str
    image: Path
    depth: Path
    intrinsics: "ndarray"
    rotation: "ndarray"
    translation: "ndarray"
    width: float
    height: float


ABSTRACT_METHOD_ERROR = "Subclasses must implement this method."


class SceneDataset(Dataset):
    def __init__(self):
        raise NotImplementedError(
            "Abstract class can not be instantiated. Implement the subclass instead."
        )

    def __getitem__(self, name_or_index: Union[str, int]) -> Tuple[Scene, int]:
        """
        Args:
            name_or_index (Union[str, int]): The index of the scene in the dataset or a unique name of the scene e.g. s00516_588.6688_588.6688_271.2803_348.6664_540_720.jpg.
        Returns:
            Tuple[Scene, int]: The scene and its index in the dataset.
        """
        raise NotImplementedError(ABSTRACT_METHOD_ERROR)

    def read_image(self, image_path: Path) -> "Tensor":
        """Reads an image from the given path.

        Args:
            image_path (Path): The path to the image. The image is expected to be an image within the current dataset.

        Returns:
            Tensor: The image as a tensor.
        """
        raise NotImplementedError(ABSTRACT_METHOD_ERROR)

    def read_images(
        self, image_paths: Iterable[Path]
    ) -> Generator["Tensor", None, None]:
        """Reads multiple images from the given paths.

        Args:
            image_paths (Iterable[Path]): The paths to the images. The images are expected to be images within the current dataset.

        Returns:
            Generator[Tensor, None, None]: The images as tensors.
        """
        for image_path in image_paths:
            yield self.read_image(image_path)

    def read_depth_map(self, depth_path: Path) -> "Tensor":
        """Reads a depth map from the given path.

        Args:
            depth_path (Path): The path to the depth map. The depth map is expected to be a depth map within the current dataset.

        Returns:
            Tensor: The depth map as a tensor.
        """
        raise NotImplementedError(ABSTRACT_METHOD_ERROR)

    def read_depth_maps(
        self, depth_paths: Iterable[Path]
    ) -> Generator["Tensor", None, None]:
        """Reads multiple depth maps from the given paths.

        Args:
            depth_paths (Iterable[Path]): The paths to the depth maps. The depth maps are expected to be depth maps within the current dataset.

        Returns:
            Generator[Tensor, None, None]: The depth maps as tensors.
        """
        for depth_path in depth_paths:
            yield self.read_depth_map(depth_path)
