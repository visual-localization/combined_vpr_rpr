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
    "/root/LOGS": "Pitts250k_Pipeline_LOGS"
}

@stub.function(
    image=image,
    mounts=[Mount.from_local_dir("./", remote_path="/root/pipeline")],
    volumes=lookup_volume(vol_dict),
    _allow_background_volume_commits = True,
    gpu="a10g",
    timeout=86400,
    retries=0,
    memory=32768
)
def entry():
    import sys
    sys.path.append("/root/pipeline")
    
    import os
    os.chdir("/root/pipeline")
    
    from main import RPR_Solver
    import torch
    
    PATH = "/cambridge_landmark/StMarysChurch"
    torch.hub.set_dir("/root/LOGS/torch_cache")
    os.makedirs("/root/LOGS/vpr_cache_cam", exist_ok=True)
    test = RPR_Solver(
        db_path = Path(PATH),
        query_path = Path(PATH),
        # set_name="pitts250k_test",
        dataset = "CamLandmark_Partial",
        vpr_type = "NetVLAD",
        vpr_only = True,
        pose_mode="weighted"
    )
    print("VPR Module: ")
    top_k_match = test.run_vpr(top_k=1)
    
    # subset = 1000
    # sub_top_k = {k:top_k_match[k] for k in list(top_k_match.keys())[:subset]}
    
    top_k_match = test.rerank(top_k_match,rerank_k=1)
    
    print("RPR Module: ")
    final_poses = test.run_rpr(top_k_match)
    print("Validation Step:")
    test.validation(final_poses)