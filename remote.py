from modal import Stub, Volume,Image,Mount,gpu

from typing import Dict
from pathlib import Path

from const import PITTS,GSV,CAM_LANDMARK



def lookup_volume(data_dict:Dict[str,str]):
    return dict((k, Volume.lookup(v)) for k, v in data_dict.items())
    
stub = Stub(
    name="Pipeline VLAD VPR Only OldHospital"
)

image = (
    Image.debian_slim(python_version="3.10")
    .apt_install(["ffmpeg","libsm6","libxext6"])
    .pip_install_from_requirements("./requirements.txt")
)


vol_dict = {
    # **GSV,
    **PITTS,
    **CAM_LANDMARK,
    "/root/LOGS": "MixVPR_Pitts250k_5BS8"
}

@stub.function(
    image=image,
    mounts=[Mount.from_local_dir("./", remote_path="/root/pipeline")],
    volumes=lookup_volume(vol_dict),
    _allow_background_volume_commits = True,
    gpu=gpu.A100(size="40GB"),
    timeout=86400,
    retries=0,
    memory=32768,
)
def entry():
    import sys
    import os
    sys.path.append("/root/pipeline")
    
    os.makedirs("/root/LOGS/torch_cache", exist_ok=True)
    os.makedirs("/root/LOGS/vpr_cache", exist_ok=True)
    os.makedirs("/root/LOGS/vpr_cache_cam", exist_ok=True)
    
    
    os.chdir("/root/pipeline")
    
    from main import RPR_Solver
    import torch
    
    PATH = "/pitts250k"
    torch.hub.set_dir("/root/LOGS/torch_cache")
    test = RPR_Solver(
        db_path = Path(PATH),
        query_path = Path(PATH),
        set_name="pitts250k_test",
        dataset = "Pittsburgh250k",
        vpr_type = "MixVPR",
        vpr_only = False,
        pose_mode="max",
    )
    print("VPR Module: ")
    top_k_match = test.run_vpr(top_k=1)
    
    # subset = 1000
    # sub_top_k = {k:top_k_match[k] for k in list(top_k_match.keys())[:subset]}
    
    # top_k_match = test.rerank(top_k_match,rerank_k=1)
    
    print("RPR Module: ")
    final_poses = test.run_rpr(top_k_match)
    print("Validation Step:")
    test.validation(final_poses)