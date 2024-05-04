from typing import Iterable, Generator, TYPE_CHECKING
from numpy import eye

if TYPE_CHECKING:
    from numpy import ndarray


def rescale_intrinsics(
    ks: Iterable["ndarray"], scale_x: float, scale_y: float
) -> Generator["ndarray", None, None]:
    """Given an intrinsic matrix (3x3) and two scale factors, returns the new intrinsic matrix corresponding to
    the new coordinates x' = scale_x * x; y' = scale_y * y
    Source: https://dsp.stackexchange.com/questions/6055/how-does-resizing-an-image-affect-the-intrinsic-camera-matrix
    Yields:
        Generator["ndarray", None, None]: The new intrinsic matrix.
    """

    for k in ks:
        transform = eye(3)
        transform[0, 0] = scale_x
        transform[0, 2] = scale_x / 2 - 0.5
        transform[1, 1] = scale_y
        transform[1, 2] = scale_y / 2 - 0.5
        kprime = transform @ k
        yield kprime
