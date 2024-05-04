from pathlib import Path
from typing import (
    Dict,
    Tuple,
    Callable,
    TYPE_CHECKING,
    Iterable,
    Generator,
)

from tqdm import tqdm
from numpy import array, float32

if TYPE_CHECKING:
    from torch import Tensor
    from numpy import ndarray

from const import PITTS_RESIZE
from .scene import Scene, SceneDataset
from .utils import read_depth_maps, read_images, rescale_intrinsics


def reorder_coordinates(xyz: tuple[float, float, float]):
    x, y, z = xyz
    return x, -z, y


def to_depth_path(image_path: Path) -> Path:
    image_path_string = image_path.as_posix()

    if "queries" in image_path_string:
        new_path = image_path_string.replace("queries_real", "queries_depths")[:-4]
        return Path(f"{new_path}.png")

    new_path = image_path_string.replace("images", "depths")[:-4]
    return Path(f"{new_path}.png")


def remap_image_path(root_dir: str, old_image_name: str) -> Path:
    # root_dir: /pitts250k
    # 003/0003313
    [dirname, name] = old_image_name.split("/")

    image_parts = name[:-4].split("_")
    splitted = int(image_parts[0]) - int(dirname) * 1000

    new_index = int(splitted / 250)

    # 000/001/002/003
    new_dirname = f"00{new_index}"
    return Path(f"{root_dir}_images_{str(dirname)}", new_dirname, name)


def read_intrinsics(resize=None):
    """
    Read the intrinsics of a specific image, according to its name
    """
    fx, fy, cx, cy, W, H = 768.000, 768.000, 320, 240, 640, 480
    K = array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=float32)
    if resize is not None:
        K = [rk for rk in rescale_intrinsics([K], resize[0] / W, resize[1] / H)][0]
    return K, W, H


def read_poses(
    filename: Path,
) -> Dict[str, Tuple["ndarray", "ndarray"]]:
    """
    Returns a dictionary that maps: img_path -> (q, t) where
    array q = (qw, qx qy qz) quaternion encoding rotation matrix;
    array t = (tx ty tz) translation vector;
    (q, t) encodes absolute pose (world-to-camera), i.e. X_c = R(q) X_W + t
    """
    poses = {}
    with filename.open("r") as f:
        for line in tqdm(
            f.readlines(), desc="🕺 Pittsburgh250k: Loading poses", ncols=100
        ):
            if ".jpg" not in line:
                continue
            line = line.strip().split(" ")

            # img name is seq5/frame00587.png
            img_name = line[0]

            qt = array(list(map(float, line[1:])))
            qt[:3] = reorder_coordinates(qt[:3])
            poses[img_name] = (qt[3:], qt[:3])
    return poses


def bulk_generate_depth_maps(
    paths: Iterable[Path],
    handler: Callable[[Path, Path, Tuple[float, float] | None], None],
    resize: Tuple[float, float] = None,
    skip_existing_depth_maps=True,
):
    for input_path in tqdm(
        paths, desc="📸 Pittsburgh250k: Generating depth maps", ncols=100
    ):
        output_path = to_depth_path(input_path)

        if output_path.exists() and skip_existing_depth_maps:
            continue

        handler(input_path, output_path, resize)


class Pittsburgh250k(SceneDataset):
    def __init__(
        self,
        base_dir: Path,
        poses_file: Path,
        generate_depth_map: Callable[[Path, Path, Tuple[float, float] | None], None],
        resize: Tuple[float, float] = PITTS_RESIZE,
        remap_image_paths=True,
        skip_existing_depth_maps=True,
    ):
        """Create a Pittsburgh250k instance of a Dataset

        Args:
            base_dir (Path): The base directory where the images are located. E.g. Path("/pitts250k").
            metadata_file (Path): Where the metadata file for this set is located. E.g. Path("/pitts250k", "poses", "pitts250k_test_db.txt").
            generate_depth_map (Callable[[Path, Path, Tuple[float, float]  |  None], None]): _description_
            resize (Tuple[float, float], optional): Whether to resize the images or not. Defaults to `PITTS_RESIZE`
        """
        print(
            f"🗺️ Pittsburgh250k. Initiating dataset at {base_dir} with poses file at {poses_file}"
        )

        self.base_dir = base_dir
        self.resize = resize
        self.intrinsics = read_intrinsics(resize)
        self.remap_image_paths = remap_image_paths
        self.poses = read_poses(poses_file)
        self.image_paths = sorted(self.poses.keys())

        input_paths = [
            (
                remap_image_path(base_dir.as_posix(), image_path)
                if remap_image_paths
                else base_dir / image_path
            )
            for image_path in self.image_paths
        ]

        bulk_generate_depth_maps(
            input_paths,
            generate_depth_map,
            resize,
            skip_existing_depth_maps,
        )

        print(f"🗺️ Pittsburgh250k. Dataset initiated with {len(self)} scenes")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, name_or_index: str | int) -> Tuple[Scene, int]:
        """
        Args:
            name (str): has name and intrinsics value in it. Ex: s00516_588.6688_588.6688_271.2803_348.6664_540_720.jpg
        Returns:
            scene_obj (Scene): return Scene object
            index (int): return the index of the scene_object in the image_paths
        """
        if isinstance(name_or_index, int):
            name = self.image_paths[name_or_index]
            index = name_or_index
        else:
            name = name_or_index
            index = self.image_paths.index(name_or_index)

        if self.remap_image_paths:
            image_path = remap_image_path(self.base_dir.as_posix(), name)
        else:
            image_path = self.base_dir / name

        depth_path = to_depth_path(image_path)

        intrinsics_matrix, width, height = self.intrinsics
        q, t = self.poses[name]

        return (
            Scene(
                name=name,
                image=image_path,
                depth=depth_path,
                intrinsics=intrinsics_matrix,
                rotation=q,
                translation=t,
                width=width,
                height=height,
            ),
            index,
        )

    def read_image(self, image_path: Path) -> "Tensor":
        return [image for image in read_images([image_path], self.resize)][0]

    def read_images(
        self, image_paths: Iterable[Path]
    ) -> Generator["Tensor", None, None]:
        return read_images(image_paths, self.resize)

    def read_depth_map(self, depth_path: Path) -> "Tensor":
        return [image for image in read_depth_maps([depth_path], self.resize)][0]

    def read_depth_maps(
        self, depth_paths: Iterable[Path]
    ) -> Generator["Tensor", None, None]:
        return read_depth_maps(depth_paths, self.resize)
