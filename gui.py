from modal import App, Volume, Image, Mount, gpu, enter, method, asgi_app
from fastapi import FastAPI
from fastapi.responses import FileResponse
from typing import Literal, Callable, NamedTuple, Literal, TYPE_CHECKING
from pathlib import Path
from dataclasses import dataclass

from const import PITTS, MIXVPR_RESIZE

if TYPE_CHECKING:
    from gui_package.model import MixVPRModel
    from gui_package.data import Pittsburgh250k, Scene
    from torch import Tensor
    import numpy as np
    from PIL import Image as PIL_Image

app = App(name="MixVprEssMatRprGui")

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


class Sample(NamedTuple):
    scene: "Scene"
    image: "PIL_Image.Image"


@app.cls(
    image=image,
    mounts=[
        Mount.from_local_python_packages(
            "gui_package.data", "gui_package.model", "depth_dpt", "SuperGlue.models"
        )
    ],
    volumes=VOLUMES,
    cpu=2.0,
    concurrency_limit=5,
    gpu=gpu.A10G(),
    timeout=86400,
    memory=32768,
    allow_concurrent_inputs=5,
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

        inference_model = MixVPRModel.load_from_checkpoint(CHECKPOINT_FILEPATH)
        inference_model.to("cuda")
        inference_model.eval()

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

    @method()
    def run(
        self,
        image_name: str,
        top_k: int,
        rerank_k: int,
        pose_mode: Literal["max", "weighted"] = "max",
    ):
        return self._run_query(image_name, top_k, rerank_k, pose_mode)

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

    @method()
    def run_and_visualize(
        self,
        sample_scene_name: str,
        top_k: int,
        rerank_k: int,
        pose_mode: Literal["max", "weighted"] = "max",
    ):
        from PIL import Image

        query_scene, retrieved_db_scenes, reranked_ordering, pose = self._run_query(
            sample_scene_name, top_k, rerank_k, pose_mode
        )

        query_image = Image.open(query_scene.image).convert("RGB")
        query_sample = Sample(query_scene, query_image)

        retrieved_db_samples = []
        for scene in retrieved_db_scenes:
            image = Image.open(scene.image).convert("RGB")
            retrieved_db_samples.append(Sample(scene, image))

        return query_sample, retrieved_db_samples, reranked_ordering, pose

    @method()
    def generate_query_sample(self, seed: int, count: int) -> list[Sample]:
        from random import Random
        from PIL import Image
        import io

        random = Random(seed)

        count = min(count, len(self.queries))
        indices = random.sample(range(len(self.queries)), k=count)

        samples = []
        for index in indices:
            scene, _ = self.queries[index]
            image = Image.open(scene.image).convert("RGB")
            samples.append(Sample(scene, image))

        return samples

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


web_app = FastAPI()
assets_path = Path(__file__).parent / "assets"


@app.function(
    image=image,
    cpu=4.0,
    memory=16384,
    concurrency_limit=3,
    mounts=[
        Mount.from_local_dir(assets_path, remote_path="/assets"),
        Mount.from_local_python_packages(
            "gui_package.data", "gui_package.model", "depth_dpt", "SuperGlue.models"
        ),
    ],
)
@asgi_app()
def fastapi_app():
    from math import radians, sin, cos, pi, tan, log
    from bokeh.plotting import figure
    from scipy.spatial.transform import Rotation
    from bokeh.models import Arrow, VeeHead, ColumnDataSource
    import xyzservices.providers as xyz
    from PIL import Image
    from gui_package.data import Scene
    from gui_package.model import Pose
    import gradio as gr
    import utm
    from gradio.routes import mount_gradio_app

    @web_app.get("/favicon.ico", include_in_schema=False)
    async def favicon():
        return FileResponse("/assets/favicon.svg")

    def to_pitch_yaw_roll(w: float, x: float, y: float, z: float):
        r = Rotation.from_quat([x, y, z, w])
        return r.as_euler("zyx", degrees=True)

    def to_start_end_arrow(x: float, y: float, length: float, angle: float):
        angle = radians(angle)
        x_end = x + length * cos(angle)
        y_end = y + length * sin(angle)
        return (x, y), (x_end, y_end)

    def pitts_to_latlon(easting: float, northing: float):
        return utm.to_latlon(easting, northing, 17, "T")

    def merc(lat, lon):
        r_major = 6378137.000
        x = r_major * radians(lon)
        scale = x / lon
        y = 180.0 / pi * log(tan(pi / 4.0 + lat * (pi / 180.0) / 2.0)) * scale
        return y, x

    def generate_map(query: Scene, retrieved: list[Scene], inferred_pose: Pose):
        gt_head = VeeHead(fill_color="#00A526", size=15)
        inferred_head = VeeHead(fill_color="#DD0018", size=15)
        reference_head = VeeHead(fill_color="#000000", size=15)

        fig = figure(
            x_range=(-2000000, 6000000),
            y_range=(-1000000, 7000000),
            x_axis_type="mercator",
            y_axis_type="mercator",
        )

        def draw_arrow(yaw, head, point):
            start, end = to_start_end_arrow(point[1], point[0], 15, yaw)
            fig.add_layout(
                Arrow(
                    end=head,
                    line_color=head.fill_color,
                    line_width=10,
                    x_start=start[0],
                    y_start=start[1],
                    x_end=end[0],
                    y_end=end[1],
                )
            )

        fig.add_tile(xyz.OpenStreetMap.Mapnik, retina=True)

        gt_latlng = pitts_to_latlon(query.translation[0], query.translation[2])
        gt_merc = merc(gt_latlng[0], gt_latlng[1])
        gt_point = ColumnDataSource(
            data=dict(
                lat=[gt_merc[0]],
                lon=[gt_merc[1]],
            )
        )

        inferred_latlng = pitts_to_latlon(inferred_pose.t[0], inferred_pose.t[2])
        inferred_merc = merc(inferred_latlng[0], inferred_latlng[1])
        inferred_point = ColumnDataSource(
            data=dict(
                lat=[inferred_merc[0]],
                lon=[inferred_merc[1]],
            )
        )

        reference_latlngs = [
            pitts_to_latlon(scene.translation[0], scene.translation[2])
            for scene in retrieved
        ]
        reference_mercs = [merc(latlng[0], latlng[1]) for latlng in reference_latlngs]
        reference_points = ColumnDataSource(
            data=dict(
                lat=[latlng[0] for latlng in reference_mercs],
                lon=[latlng[1] for latlng in reference_mercs],
            )
        )

        gt_yaw = to_pitch_yaw_roll(*query.rotation)[1]
        inferred_yaw = to_pitch_yaw_roll(*inferred_pose.R)[1]
        reference_yaws = [to_pitch_yaw_roll(*scene.rotation)[1] for scene in retrieved]

        fig.scatter(
            x="lon",
            y="lat",
            size=30,
            fill_color=reference_head.fill_color,
            fill_alpha=1,
            source=reference_points,
        )

        for reference_yaw, reference_merc in zip(reference_yaws, reference_mercs):
            draw_arrow(
                reference_yaw, reference_head, (reference_merc[0], reference_merc[1])
            )

        fig.scatter(
            x="lon",
            y="lat",
            size=30,
            fill_color=gt_head.fill_color,
            fill_alpha=1,
            source=gt_point,
        )

        draw_arrow(gt_yaw, gt_head, (gt_merc[0], gt_merc[1]))

        fig.scatter(
            x="lon",
            y="lat",
            size=30,
            fill_color=inferred_head.fill_color,
            fill_alpha=1,
            source=inferred_point,
        )

        draw_arrow(inferred_yaw, inferred_head, (inferred_merc[0], inferred_merc[1]))

        return fig

    def generate_sample(seed: float, count: float, current_samples: list[Sample]):
        samples: list[Sample] = MixVprEssMatRprModel().generate_query_sample.remote(
            int(seed), int(count)
        )

        current_samples.clear()
        current_samples.extend(samples)

        dataset = [
            [sample.scene.name, sample.scene.width, sample.scene.height]
            for sample in samples
        ]

        return dataset, samples

    def on_query_preview(
        sample_query_gallery_index: int,
        query_samples: list[Sample],
    ):
        sample = query_samples[sample_query_gallery_index]
        return sample.image

    def on_query(
        sample_index: int,
        samples: list[Sample],
        top_k: float,
        rerank_k: float,
        pose_mode: Literal["max", "weighted"] = "max",
    ):
        image_name: str = samples[sample_index].scene.name

        response = MixVprEssMatRprModel().run_and_visualize.remote(
            image_name, int(top_k), int(rerank_k), pose_mode
        )

        query_sample, retrieved_db_samples, reranked_ordering, pose = response

        plot = generate_map(
            query_sample.scene, [sample.scene for sample in retrieved_db_samples], pose
        )

        return (
            query_sample.image,
            [sample.image for sample in retrieved_db_samples],
            [retrieved_db_samples[i].image for i in reranked_ordering],
            str(pose),
            plot,
        )

    default_sample_image = Image.open("/assets/default_sample/image.jpg").convert("RGB")
    with open("/assets/default_sample/scene.json") as f:
        string = f.read()
        default_sample_scene = Scene.from_json(string)

    default_sample = Sample(default_sample_scene, default_sample_image)

    with gr.Blocks(title="MixVPR Essential Matrix Pose Regression Model") as interface:
        gr.Markdown(
            """
# MixVPR Essential Matrix Pose Regression Model

This is a demo of the MixVPR Essential Matrix Pose Regression Model. 
The model takes in a query image and a gallery of images, and retrieves the top k images from the gallery that are most similar to the query image.
The model then reranks the top k images based on the pose similarity between the query image and the gallery images.
The model uses the Essential Matrix to estimate the pose similarity between the query image and the gallery images.
Finally, the model estimate the pose of the query image based on the similarity extracted.
"""
        )

        query_samples = gr.State([default_sample])

        with gr.Tab("Interface"):
            gr.Markdown("## Input")

            with gr.Row():
                sample_query_preview = gr.Image(label="Query preview")
                sample_query_gallery = gr.Dataset(
                    components=[
                        "text",
                        "text",
                        "text",
                    ],
                    headers=["Scene name", "Width", "Height"],
                    label="Query Samples",
                    type="index",
                    samples=[
                        [
                            default_sample.scene.name,
                            default_sample.scene.width,
                            default_sample.scene.height,
                        ]
                    ],
                )

            gr.Markdown("### Query Options")
            with gr.Row():
                query_top_k_slider = gr.Slider(
                    1, 10, 5, label="Retrieve top k images", step=1
                )

                query_rerank_k_slider = gr.Slider(
                    1, 10, 5, label="Rerank top k images", step=1
                )

                pose_mode_dropdown = gr.Dropdown(
                    ["max", "weighted"], label="Pose Mode", value="max"
                )

            query_button = gr.Button("Query", variant="primary")

            gr.Markdown("## Output")

            with gr.Row():
                query_image = gr.Image(label="Query Image")
                output_map = gr.Plot()

            with gr.Accordion("Step-by-step", open=False):
                retrieved_gallery = gr.Gallery(label="Retrieved Images", columns=1)
                reranked_gallery = gr.Gallery(label="Reranked Images", columns=1)
                pose_output = gr.Textbox(
                    label="Pose Output", value="", interactive=False
                )

        with gr.Tab("Configuration"):
            gr.Markdown("## Sample Configuration")

            sample_seed = gr.Number(
                label="Sample seed", value=0, minimum=0, maximum=1000, step=1
            )

            sample_count = gr.Number(
                label="Sample count", value=1, minimum=1, maximum=15, step=1
            )

            generate_new_sample_button = gr.Button("Generate New Sample")

        # interface.load(
        #     generate_sample,
        #     [sample_seed, sample_count, query_samples],
        #     [sample_query_gallery, query_samples],
        # )

        generate_new_sample_button.click(
            generate_sample,
            [sample_seed, sample_count, query_samples],
            [sample_query_gallery, query_samples],
        )

        sample_query_gallery.click(
            on_query_preview,
            [sample_query_gallery, query_samples],
            [sample_query_preview],
        )

        # sample_query_gallery.select(
        #     on_sample_gallery_select, None, sample_query_gallery_selected_index
        # )

        query_button.click(
            on_query,
            [
                sample_query_gallery,
                query_samples,
                query_top_k_slider,
                query_rerank_k_slider,
                pose_mode_dropdown,
            ],
            [query_image, retrieved_gallery, reranked_gallery, pose_output, output_map],
        )

    return mount_gradio_app(
        app=web_app,
        blocks=interface,
        path="/",
    )


@app.function(
    image=image,
    mounts=[
        Mount.from_local_python_packages(
            "gui_package.data", "gui_package.model", "depth_dpt", "SuperGlue.models"
        )
    ],
    volumes=VOLUMES,
)
def generate_default_sample():
    samples = MixVprEssMatRprModel().generate_query_sample.remote(0, 1)
    sample = samples[0]

    assets_path = Path(LOG_DIR, "assets", "default_sample")
    assets_path.mkdir(parents=True, exist_ok=True)

    # save the sample image to ./assets/default_sample/{Path(scene_name).name} (ext included in scene_name)
    sample_image: Image.Image = sample.image
    sample_image.save(assets_path / "image.jpg")

    # save the scene NamedTuple to ./assets/default_sample/{Path(scene_name).name}.json
    sample_scene: "Scene" = sample.scene
    with open(
        assets_path / "scene.json",
        "w",
    ) as f:
        string = sample_scene.to_json()
        f.write(string)

    for volume in VOLUMES.values():
        volume.commit()


@app.local_entrypoint()
def entry():
    generate_default_sample.remote()
