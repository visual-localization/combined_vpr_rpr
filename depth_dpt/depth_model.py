import torch
from torchvision.transforms import Compose
import cv2
import numpy as np
from tqdm import tqdm

import glob 
import os

from .util.io import read_image,write_depth

from .dpt.models import DPTDepthModel
from .dpt.transforms import Resize, NormalizeImage, PrepareForNet


class DPT_DepthModel:
    def __init__(
        self, model_path = None, model_type="dpt_hybrid_kitti",
        optimize=True,
        kitti_crop=False,
        absolute_depth = False
    ):
        
        # set torch options
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        
        # Set device to send model to
        self.model_type = model_type
        self.optimize = optimize
        self.kitti_crop = kitti_crop
        self.absolute_depth = absolute_depth
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        default_models = {
            "midas_v21": "weights/midas_v21-f6b98070.pt",
            "dpt_large": "weights/dpt_large-midas-2f21e586.pt",
            "dpt_hybrid": "weights/dpt_hybrid-midas-501f0c75.pt",
            "dpt_hybrid_kitti": "/content/drive/MyDrive/Model/final_rpr-master/depth_dpt/weights/dpt_hybrid_kitti-cb926ef4.pt",
            "dpt_hybrid_nyu": "weights/dpt_hybrid_nyu-2ce69ec7.pt",
        }

        if model_path is None:
            model_path = default_models[self.model_type]
        
        ## Setting up Depth Model
        if(model_type == "dpt_hybrid_kitti"):
            net_w = 1216
            net_h = 352

            self.model = DPTDepthModel(
                path=model_path,
                scale=0.00006016,
                shift=0.00579,
                invert=True,
                backbone="vitb_rn50_384",
                non_negative=True,
                enable_attention_hooks=False,
            )
            normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        else:
            assert (
                False
            ), f"model_type '{model_type}' not implemented, use: [dpt_large|dpt_hybrid|dpt_hybrid_kitti|dpt_hybrid_nyu|midas_v21]"

        self.transform = Compose(
            [
                Resize(
                    net_w,
                    net_h,
                    resize_target=None,
                    keep_aspect_ratio=True,
                    ensure_multiple_of=32,
                    resize_method="minimal",
                    image_interpolation_method=cv2.INTER_CUBIC,
                ),
                normalization,
                PrepareForNet(),
            ]
        )
        
        if self.optimize == True and self.device == torch.device("cuda"):
            self.model = self.model.to(memory_format=torch.channels_last)
            self.model = self.model.half()

        self.model.to(self.device)
    
    def generate_monodepth(self,img_path):
        # input
        img:np.ndarray = read_image(img_path)

        if self.kitti_crop is True:
            height, width, _ = img.shape
            top = height - 352
            left = (width - 1216) // 2
            img = img[top : top + 352, left : left + 1216, :]

        img_input = self.transform({"image": img})["image"]

        # compute
        with torch.no_grad():
            sample = torch.from_numpy(img_input).to(self.device).unsqueeze(0)

            if self.optimize == True and self.device == torch.device("cuda"):
                sample = sample.to(memory_format=torch.channels_last)
                sample = sample.half()

            prediction = self.model.forward(sample)
            prediction = (
                torch.nn.functional.interpolate(
                    prediction.unsqueeze(1),
                    size=img.shape[:2],
                    mode="bicubic",
                    align_corners=False,
                )
                .squeeze()
                .cpu()
                .numpy()
            )

            if self.model_type == "dpt_hybrid_kitti":
                prediction *= 256
                
        return prediction
    #region Ihatemylife
    def batch_generate_monodepth(self, input_path, output_path):
  
        print("start processing")
        for ind, img_name in enumerate(os.listdir(input_path)):
            print(ind)
            filename = os.path.join(
                output_path, os.path.splitext(os.path.basename(img_name))[0]
            )
            if(os.path.exists(filename+".png")):
              print(f"skipped {ind}")
              continue
            print(img_name)
            prediction = self.generate_monodepth(os.path.join(input_path,img_name))
            write_depth(filename, prediction, bits=2, absolute_depth=self.absolute_depth)
        print("finished")
    
    def img_db_process(self,input_path,dataset:str):
        if(dataset == "Mapfree"):
            self.mapfree_img_process(input_path,dataset)
        else:
            #TODO
            raise NotImplementedError("Teehee")
    
    def mapfree_img_process(self,input_path):
        depth_path = os.path.join(input_path,'depth')
        img_path = os.path.join(input_path,"image")
        #If output folder exisit, return
        # if(os.path.exists(depth_path)):
        #     print("Skipped depth")
        #     return
        os.makedirs(depth_path, exist_ok=True)
        self.batch_generate_monodepth(img_path,depth_path)
    #endregion
    
    def solo_generate_monodepth(self, input_path, output_path):
        """
        The function received in a path to an image file and 
        generate the corresponding depth image to the output path
        """
        if(os.path.exists(output_path)):
            return
        prediction = self.generate_monodepth(input_path)
        write_depth(output_path, prediction, bits=2, absolute_depth=self.absolute_depth)