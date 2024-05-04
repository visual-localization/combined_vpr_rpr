from modal import App, Volume, Image, Mount, enter, method
from typing import Dict
from pathlib import Path

from const import PITTS


def lookup_volume(data_dict: Dict[str, str]):
    return dict((k, Volume.lookup(v)) for k, v in data_dict.items())


app = App(name="GUI Model modal backend")

image = (
    Image.debian_slim(python_version="3.11")
    .apt_install(["ffmpeg", "libsm6", "libxext6"])
    .pip_install_from_requirements("./requirements.txt")
)

vol_dict = {**PITTS, "/root/LOGS": "Pitts250k_Pipeline_LOGS"}


@app.cls(
    image=image,
    mounts=[Mount.from_local_python_packages("gui_package.data", "depth_dpt")],
    volumes=lookup_volume(vol_dict),
    memory=32768,
    gpu=False,
)
class Model:
    @enter()
    def setup(self):
        # NOTE: Remember to switch to absolute import paths instead of relative . paths after mounting the local libraries
        from gui_package.data import Pittsburgh250k
        from depth_dpt import DPT_DepthModel

        DEPTH_MODEL = DPT_DepthModel()

        BASE_DATA_DIR = Path("/pitts250k")
        self.database = Pittsburgh250k(
            base_dir=BASE_DATA_DIR,
            poses_file=BASE_DATA_DIR / "poses" / "pitts250k_test_db.txt",
            generate_depth_map=lambda i, o, r: DEPTH_MODEL.solo_generate_monodepth(
                i, o, r
            ),
        )

        self.queries = Pittsburgh250k(
            base_dir=Path("/pitts250k_queries_real"),
            poses_file=BASE_DATA_DIR / "poses" / "pitts250k_test_query.txt",
            remap_image_paths=False,
            generate_depth_map=lambda i, o, r: DEPTH_MODEL.solo_generate_monodepth(
                i, o, r
            ),
        )

    @method()
    def run(self):
        db_scene, _ = self.database[0]
        db_image = self.database.read_image(db_scene.image)
        db_depth = self.database.read_depth_map(db_scene.depth)

        query_scene, _ = self.queries[0]
        query_image = self.queries.read_image(query_scene.image)
        query_depth = self.queries.read_depth_map(query_scene.depth)

        return db_image, db_depth, query_image, query_depth


@app.function()
def main():
    model = Model()
    test = model.run.remote()
    print(test)


@app.local_entrypoint()
def entry():
    main.remote()
