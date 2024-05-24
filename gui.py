from modal import App, Volume, Image, Mount, enter, method, gpu
from typing import Literal, Callable, TYPE_CHECKING
from pathlib import Path
from dataclasses import dataclass

from const import PITTS, MIXVPR_RESIZE

if TYPE_CHECKING:
    from gui_package.model import MixVPRModel
    from gui_package.data import Pittsburgh250k, Scene
    from torch import Tensor
    import numpy as np


app = App(name="GUI Model modal backend")

image = (
    Image.debian_slim(python_version="3.11")
    .apt_install(["ffmpeg", "libsm6", "libxext6"])
    .pip_install_from_requirements("./requirements.txt")
)

LOG_DIR = "/LOGS"

vol_dict = {**PITTS, LOG_DIR: "MixVprEssMatRprBackend"}

VOLUMES = {k: Volume.lookup(v) for k, v in vol_dict.items()}

RERANKER_CHECKPOINT_FILEPATH = Path(
    LOG_DIR,
    "resnet50_MixVPR_4096_channels(1024)_rows(4).ckpt",
)

RERANKER_NAME = RERANKER_CHECKPOINT_FILEPATH.stem

CHECKPOINT_FILEPATH = Path(
    LOG_DIR,
    "resnet50_epoch(00)_step(0111)_R1[0.9368]_R5[0.9823]_OverlapR1[0.0869]_OverlapR5[0.0930].ckpt",
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
    image=image,
    mounts=[
        Mount.from_local_python_packages(
            "gui_package.data", "gui_package.model", "depth_dpt", "SuperGlue.models"
        )
    ],
    volumes=VOLUMES,
    cpu=2.0,
    gpu=gpu.A10G(),
    timeout=86400,
    memory=32768,
    allow_concurrent_inputs=True,
)
class MixVprEssMatRprModel:
    @enter()
    def setup(self):
        from gui_package.data import Pittsburgh250k
        from gui_package.model import MixVPRModel, EssMatPoseRegressionModel
        from depth_dpt import DPT_DepthModel
        import numpy as np
        from torch import load as torch_load

        DEPTH_MODEL = DPT_DepthModel()

        def generate_depth_map(i, o, r):
            return DEPTH_MODEL.solo_generate_monodepth(i, o, r)

        self.database = Pittsburgh250k(
            base_dir=BASE_DATA_DIR,
            poses_file=BASE_DATA_DIR / "poses" / "pitts250k_test_db.txt",
            generate_depth_map=generate_depth_map,
        )

        self.queries = Pittsburgh250k(
            base_dir=QUERIES_DIR,
            poses_file=BASE_DATA_DIR / "poses" / "pitts250k_test_query.txt",
            remap_image_paths=False,
            generate_depth_map=generate_depth_map,
        )

        inference_model = MixVPRModel.load_from_checkpoint(CHECKPOINT_FILEPATH)
        inference_model.to("cuda")
        inference_model.eval()

        self.matching_model = EssMatPoseRegressionModel(
            self.queries.resize,
            self.queries,
            self.database,
            "cuda",
        )

        rerank_model = MixVPRModel(
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

        rerank_model.load_state_dict(
            torch_load(RERANKER_CHECKPOINT_FILEPATH, map_location="cuda")
        )
        rerank_model.to("cuda")
        rerank_model.eval()

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

    @method()
    def get_scene(
        self, dataset: Literal["database", "queries"], index_or_name: str | int
    ) -> tuple["Scene", int]:
        if dataset == "database":
            return self.database[index_or_name]
        elif dataset == "queries":
            return self.queries[index_or_name]

    @method()
    def run(self, image_name: str, top_k: int, rerank_k: int):
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
        reranked_top_k = top_k_matches[reranked_top_k]

        reranked_database_scenes = [self.database[i][0] for i in reranked_top_k]

        pose = self.matching_model.process_top_k(
            reranked_database_scenes, query_scene, "max"
        )

        return query_scene, retrieved_database_scenes, reranked_database_scenes, pose

    @method()
    def quick_test(self):
        first_scene = self.get_scene.remote("queries", 0)[0]
        query_scene, retrieved_db_scenes, pose = self.run.remote(first_scene.name, 5, 5)

        print("Query scene", query_scene)
        print("Retrieved database scenes", retrieved_db_scenes)
        print("Pose", pose)

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


@app.local_entrypoint()
def entry():
    # main.remote()
    MixVprEssMatRprModel().quick_test.remote()
