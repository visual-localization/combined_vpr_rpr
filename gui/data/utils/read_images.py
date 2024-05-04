from pathlib import Path
from typing import Generator, Iterable, Tuple, TYPE_CHECKING
from cv2 import imread, resize, cvtColor, COLOR_BGR2RGB, IMREAD_COLOR
from torch import from_numpy

if TYPE_CHECKING:
    from torch import Tensor


def read_images(
    paths: Iterable[Path], size: Tuple[float, float] = None
) -> Generator["Tensor", None, None]:
    """Reads an image from the given path.

    Args:
        paths (Iterable[Path]): The paths to the images
        size (Tuple[float, float]): The size to resize the images to

    Yields:
        Generator["Tensor", None, None]: The image as a torch tensor, (h, w, 3) -> (3, h, w) and normalized
    """

    for path in paths:
        image = imread(path.as_posix(), IMREAD_COLOR)
        image = cvtColor(image, COLOR_BGR2RGB)

        if size is not None:
            image = resize(image, size)

        image = from_numpy(image).float().permute(2, 0, 1) / 255
        yield image
