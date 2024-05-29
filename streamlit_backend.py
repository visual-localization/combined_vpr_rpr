from modal import App, Volume, Image, Mount, gpu, enter, method
from typing import Literal, Callable, Literal, TYPE_CHECKING
from pathlib import Path
from dataclasses import dataclass
from streamlit_app.data_models import (
    SceneDTOS,
    PoseDTOS,
    QueryResponseDTOS,
    SampleDTOS,
    ImageDTOS,
)

from const import PITTS, MIXVPR_RESIZE

if TYPE_CHECKING:
    from gui_package.model import MixVPRModel
    from gui_package.data import Pittsburgh250k
    from torch import Tensor
    import numpy as np

image = (
    Image.debian_slim(python_version="3.11")
    .apt_install(["ffmpeg", "libsm6", "libxext6"])
    .pip_install_from_requirements("./requirements.txt")
)

app = App(name="EssentialMixer", image=image)

LOG_DIR = "/LOGS"

VOLUMES_DICT = {**PITTS, LOG_DIR: "MixVprEssMatRprBackend"}

VOLUMES = {k: Volume.lookup(v) for k, v in VOLUMES_DICT.items()}

RERANKER_CHECKPOINT_FILEPATH = Path(
    LOG_DIR,
    "resnet50_epoch(00)_step(0111)_R1[0.9368]_R5[0.9823]_OverlapR1[0.0869]_OverlapR5[0.0930].ckpt",
)

RERANKER_NAME = RERANKER_CHECKPOINT_FILEPATH.stem

CHECKPOINT_FILEPATH = Path(
    LOG_DIR,
    "resnet50_MixVPR_4096_channels(1024)_rows(4).ckpt",
)

CHECKPOINT_NAME = CHECKPOINT_FILEPATH.stem

CACHE_DB_DESCRIPTORS_FILEPATH = Path(
    LOG_DIR, f"{CHECKPOINT_NAME}_database_global_descriptors.npy"
)

CACHE_RERANKER_DB_DESCRIPTORS_FILEPATH = Path(
    LOG_DIR, f"{RERANKER_NAME}_database_global_descriptors.npy"
)

CACHE_QUERY_DESCRIPTORS_FILEPATH = Path(
    LOG_DIR, f"{CHECKPOINT_NAME}_query_global_descriptors.npy"
)

CACHE_RERANKER_QUERY_DESCRIPTORS_FILEPATH = Path(
    LOG_DIR, f"{RERANKER_NAME}_query_global_descriptors.npy"
)

BASE_DATA_DIR = Path("/pitts250k")
QUERIES_DIR = Path("/pitts250k_queries_real")


@dataclass
class InferenceConfig:
    feature_dim: int
    batch_size: int = 4
    num_workers: int = 4
    device: str = "cuda"


@app.cls(
    mounts=[
        Mount.from_local_python_packages(
            "gui_package.data", "gui_package.model", "depth_dpt", "SuperGlue.models"
        )
    ],
    volumes=VOLUMES,
    container_idle_timeout=300,
    concurrency_limit=3,
    allow_concurrent_inputs=10,
    gpu=gpu.A10G(),
    timeout=86400,
    memory=4096,
)
class MixVprEssMatRprModel:
    @enter()
    def setup(self):
        from gui_package.data import Pittsburgh250k
        from gui_package.model import MixVPRModel, EssMatPoseRegressionModel
        import numpy as np
        from torch import load as torch_load

        # We presume the database and queries datasets have not been truncated in anyway
        # so we're kipping the depth map generation process during file validation
        # from depth_dpt import DPT_DepthModel
        # DEPTH_MODEL = DPT_DepthModel()
        # def generate_depth_map(i, o, r):
        #     return DEPTH_MODEL.solo_generate_monodepth(i, o, r)

        self.database = Pittsburgh250k(
            base_dir=BASE_DATA_DIR,
            poses_file=BASE_DATA_DIR / "poses" / "pitts250k_test_db.txt",
            # uncomment this line to enable depth map generation and comment the line after it to enable file validation
            # generate_depth_map=generate_depth_map,
            skip_validation=True,
        )

        self.queries = Pittsburgh250k(
            base_dir=QUERIES_DIR,
            poses_file=BASE_DATA_DIR / "poses" / "pitts250k_test_query.txt",
            remap_image_paths=False,
            # uncomment this line to enable depth map generation and comment the line after it to enable file validation
            # generate_depth_map=generate_depth_map,
            skip_validation=True,
        )

        self.matching_model = EssMatPoseRegressionModel(
            self.queries.resize,
            self.queries,
            self.database,
            "cuda",
        )

        # validate all the cache files to quickly skip inference model loading
        if (
            CACHE_DB_DESCRIPTORS_FILEPATH.exists()
            and CACHE_RERANKER_DB_DESCRIPTORS_FILEPATH.exists()
            and CACHE_QUERY_DESCRIPTORS_FILEPATH.exists()
            and CACHE_RERANKER_QUERY_DESCRIPTORS_FILEPATH.exists()
        ):
            print("Skipping model loading. All cache files detected.")
            return

        rerank_model = MixVPRModel.load_from_checkpoint(RERANKER_CHECKPOINT_FILEPATH)
        rerank_model.to("cuda")
        rerank_model.eval()

        inference_model = MixVPRModel(
            backbone_arch="resnet50",
            layers_to_crop=[4],
            agg_arch="MixVPR",
            agg_config={
                "in_channels": 1024,
                "in_h": 20,
                "in_w": 20,
                "out_channels": 1024,
                "mix_depth": 4,
                "mlp_ratio": 1,
                "out_rows": 4,
            },
        )

        inference_model.load_state_dict(
            torch_load(CHECKPOINT_FILEPATH, map_location="cuda")
        )
        inference_model.to("cuda")
        inference_model.eval()

        self._validate_descriptors_cache(
            CACHE_DB_DESCRIPTORS_FILEPATH,
            inference_model,
            self.database,
            lambda x: np.save(CACHE_DB_DESCRIPTORS_FILEPATH, x),
        )

        self._validate_descriptors_cache(
            CACHE_QUERY_DESCRIPTORS_FILEPATH,
            inference_model,
            self.queries,
            lambda x: np.save(CACHE_QUERY_DESCRIPTORS_FILEPATH, x),
        )

        self._validate_descriptors_cache(
            CACHE_RERANKER_DB_DESCRIPTORS_FILEPATH,
            rerank_model,
            self.database,
            lambda x: np.save(CACHE_RERANKER_DB_DESCRIPTORS_FILEPATH, x),
        )

        self._validate_descriptors_cache(
            CACHE_RERANKER_QUERY_DESCRIPTORS_FILEPATH,
            rerank_model,
            self.queries,
            lambda x: np.save(CACHE_RERANKER_QUERY_DESCRIPTORS_FILEPATH, x),
        )

        for volume in VOLUMES.values():
            volume.reload()

    def _validate_descriptors_cache(
        self,
        filepath: Path,
        inference_model: "MixVPRModel",
        database: "Pittsburgh250k",
        save: Callable[["np.ndarray[np.float64]"], None],
        force_update: bool = False,
    ):
        if not filepath.exists() or force_update:
            descriptors = self._bulk_inference_on_dataset(
                inference_model,
                InferenceConfig(feature_dim=4096, num_workers=4, batch_size=512),
                database,
            )

            save(descriptors)

            # commit changes to volumes
            for volume in VOLUMES.values():
                volume.commit()
        else:
            print(f"Skipping features extraction from {filepath}. Cache file detected.")

    def _run_query(
        self,
        image_name: str,
        top_k: int,
        rerank_k: int,
        pose_mode: Literal["max", "weighted"] = "max",
    ):
        import numpy as np

        query_scene, query_index = self.queries[image_name]

        database_descriptors = np.load(CACHE_DB_DESCRIPTORS_FILEPATH)
        all_query_descriptors = np.load(CACHE_QUERY_DESCRIPTORS_FILEPATH)
        reranker_db_descriptors = np.load(CACHE_RERANKER_DB_DESCRIPTORS_FILEPATH)
        reranker_all_query_descriptors = np.load(
            CACHE_RERANKER_QUERY_DESCRIPTORS_FILEPATH
        )

        query_descriptors = all_query_descriptors[query_index]
        reranker_query_descriptors = reranker_all_query_descriptors[query_index]

        similarity_vector = np.dot(database_descriptors, query_descriptors)
        top_k_matches = np.argsort(-similarity_vector, axis=0)[:top_k]

        retrieved_database_scenes = [self.database[i][0] for i in top_k_matches]

        reranker_top_k_descriptors = reranker_db_descriptors[top_k_matches]
        reranked_similarity_vector = np.dot(
            reranker_top_k_descriptors, reranker_query_descriptors
        )

        reranked_top_k = np.argsort(-reranked_similarity_vector, axis=0)[:rerank_k]

        reranked_database_scenes = [
            retrieved_database_scenes[i] for i in reranked_top_k
        ]

        pose = self.matching_model.process_top_k(
            reranked_database_scenes, query_scene, pose_mode
        )

        return query_scene, retrieved_database_scenes, reranked_top_k, pose

    def _bulk_inference_on_dataset(
        self, model: "MixVPRModel", config: InferenceConfig, dataset: "Pittsburgh250k"
    ):
        from torch import no_grad
        from torch.utils.data import DataLoader
        from numpy import zeros, array
        from tqdm import tqdm
        import torchvision.transforms as tvf
        from torch.utils.data import Dataset

        class TensorPittsburghDataset(Dataset):
            def __init__(self, dataset: "Pittsburgh250k"):
                self.dataset = dataset
                self.mixvpr_transform = tvf.Compose(
                    [
                        tvf.Resize(
                            MIXVPR_RESIZE, interpolation=tvf.InterpolationMode.BICUBIC
                        ),
                        tvf.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                    ]
                )

            def __getitem__(self, index: int) -> tuple["Tensor", int]:
                scene, index = self.dataset[index]
                image = self.dataset.read_image(scene.image)
                image = self.mixvpr_transform(image)
                return image, index

            def __len__(self) -> int:
                return len(self.dataset)

        dataloader = DataLoader(
            dataset=TensorPittsburghDataset(dataset),
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=True,
            drop_last=False,
        )

        with no_grad():
            global_descriptors = zeros((len(dataset), config.feature_dim))

            for batch in tqdm(dataloader, ncols=100, desc="Extracting features"):
                images, indices = batch
                images = images.to(config.device)

                descriptors = model(images)
                descriptors = descriptors.detach().cpu().numpy()

                global_descriptors[array(indices), :] = descriptors

        return global_descriptors

    @method()
    def sample_query_dataset(self, seed=0, count=1):
        if count < 1:
            raise ValueError("Count should be greater than 0")

        from random import Random
        import base64

        random = Random(seed)

        count = min(count, len(self.queries))
        indices = random.sample(range(len(self.queries)), k=count)

        samples: list[dict] = []
        for index in indices:
            scene, _ = self.queries[index]

            with open(scene.image, "rb") as f:
                image_data = f.read()

            samples.append(
                SampleDTOS(
                    scene=SceneDTOS.from_scene(scene),
                    image=ImageDTOS(
                        name=scene.name,
                        media_type="image/jpeg",
                        size=len(image_data),
                        data=base64.b64encode(image_data),
                    ),
                ).to_dict()
            )

        return samples

    @method()
    def query(
        self,
        image_name: str,
        top_k: int = 10,
        rerank_k: int = 10,
        pose_mode: Literal["max", "weighted"] = "max",
    ):
        query_scene, retrieved_database_scenes, reranked_top_k, pose = self._run_query(
            image_name, top_k, rerank_k, pose_mode
        )
        query_scene_dtos = SceneDTOS.from_scene(query_scene)
        retrieved_scenes_dtos = [
            SceneDTOS.from_scene(scene) for scene in retrieved_database_scenes
        ]
        reranking_indices = reranked_top_k.tolist()
        pose_dtos = PoseDTOS.from_pose(pose)

        return QueryResponseDTOS(
            query=query_scene_dtos,
            retrieved_scenes=retrieved_scenes_dtos,
            reranking_indices=reranking_indices,
            pose=pose_dtos,
        ).to_dict()

    @method()
    def get_images(
        self, image_requests: list[tuple[Literal["database", "query"], str]]
    ):
        import base64

        images_bytes: list[dict] = []
        for request_type, image_name in image_requests:
            if request_type == "database":
                scene, _ = self.database[image_name]
            elif request_type == "query":
                scene, _ = self.queries[image_name]
            else:
                raise ValueError(f"Invalid request type: {request_type}")

            with open(scene.image, "rb") as f:
                data = f.read()

            images_bytes.append(
                ImageDTOS(
                    name=scene.name,
                    media_type="image/jpeg",
                    size=len(data),
                    data=base64.b64encode(data),
                ).to_dict()
            )

        return images_bytes


@app.function(
    mounts=[
        Mount.from_local_python_packages(
            "gui_package.data", "gui_package.model", "depth_dpt", "SuperGlue.models"
        )
    ],
    volumes=VOLUMES,
    cpu=2.0,
    timeout=86400,
    memory=8192,
)
def generate_samples():
    import shutil

    sample_resp: list[dict] = MixVprEssMatRprModel.sample_query_dataset.remote(
        2011744, 30
    )
    samples = [SampleDTOS.from_dict(sample) for sample in sample_resp]

    image_paths = [sample.scene.image for sample in samples]
    scene_names = [sample.scene.name for sample in samples]

    sample_dir = Path("/LOGS/samples")
    sample_dir.mkdir(exist_ok=True)

    # save scene names to /samples/scene_names.txt
    with open(sample_dir / "scene_names.txt", "w") as f:
        f.write("\n".join(scene_names))

    # copy image to /samples, renamed to index
    for index, image_path in enumerate(image_paths):
        target_file = sample_dir / f"{index}.jpg"
        if target_file.exists():
            continue

        shutil.copy(image_path, sample_dir / f"{index}.jpg")

    for volume in VOLUMES.values():
        volume.commit()


@app.local_entrypoint()
def entry():
    generate_samples.remote()
