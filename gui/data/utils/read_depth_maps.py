from pathlib import Path
from typing import Generator, Iterable, TYPE_CHECKING
from cv2 import imread, IMREAD_UNCHANGED
from torch import from_numpy

if TYPE_CHECKING:
    from torch import Tensor


def read_depth_maps(paths: Iterable[Path]) -> Generator["Tensor", None, None]:
    """
    Args:
        paths (Iterable[Path]): The paths to the depth maps
    Yields:
        Generator["Tensor", None, None]: The depth maps as a torch tensor, (h, w).

    """

    for path in paths:
        # read and resize image
        depth = imread(path.as_posix(), IMREAD_UNCHANGED)
        depth = depth / 1000
        depth = from_numpy(depth).float()
        yield depth
