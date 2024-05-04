import torch
from torchvision.transforms import Compose
import cv2
import numpy as np

from pathlib import Path
import os

from .util.io import read_image,write_depth

from .dpt.models import DPTDepthModel
from .dpt.transforms import Resize, NormalizeImage, PrepareForNet
from .unidepth.models import UniDepthV1

DEFAULT_MODELS = {
    "midas_v21": "weights/midas_v21-f6b98070.pt",
    "dpt_large": "weights/dpt_large-midas-2f21e586.pt",
    "dpt_hybrid": "weights/dpt_hybrid-midas-501f0c75.pt",
    "dpt_hybrid_kitti": "./depth_dpt/weights/dpt_hybrid_kitti-cb926ef4.pt",
    "dpt_hybrid_nyu": "weights/dpt_hybrid_nyu-2ce69ec7.pt",
}


class DPT_DepthModel:
    def __init__(
        self, 
        model_name = "DPT", model_path = None, model_type="dpt_hybrid_kitti",
        optimize=True,
        kitti_crop=False,
        absolute_depth = True
    ):
        
        # set torch options
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        
        # Set device to send model to
        self.model_name = model_name
        self.model_type = model_type
        self.optimize = optimize
        self.kitti_crop = kitti_crop
        self.absolute_depth = absolute_depth
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        
        if model_path is None:
            model_path = DEFAULT_MODELS[self.model_type]
        
        ## Setting up Depth Model
        if(self.model_name == "DPT"):
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
            else:
                assert False, f"model_type '{model_type}' not implemented, use: [dpt_large|dpt_hybrid|dpt_hybrid_kitti|dpt_hybrid_nyu|midas_v21]"
        elif(self.model_name == "UniDepth"):
            self.model = UniDepthV1.from_pretrained("lpiccinelli/unidepth-v1-vitl14")
        
        else:
            assert False, f"model_name '{model_name}' not implemented, use: [DPT|UniDepth]"

        
        
        if self.model_name == "DPT" and self.optimize == True and self.device == torch.device("cuda"):
            self.model = self.model.to(memory_format=torch.channels_last)
            self.model = self.model.half()

        self.model.eval().to(self.device)
        
    def generate_monodepth(self,img_path,resize)->np.ndarray:
        if(self.model_name=="DPT"):
            return self.generate_monodepth_dpt(img_path,resize)
        elif(self.model_name == "UniDepth"):
            return self.generate_monodepth_unidepth(img_path,resize)
        else:
            raise NotImplementedError()
        
    def generate_monodepth_unidepth(self,img_path,resize)->np.ndarray:
        img:np.ndarray = read_image(img_path,resize)
        with torch.no_grad():
            sample = torch.from_numpy(img).permute(2, 0, 1).to(self.device) * 255.0
            predictions = self.model.infer(sample)
            depth = predictions["depth"]
        return depth[0][0].cpu().numpy()
    
    def generate_monodepth_dpt(self,img_path,resize)->np.ndarray:
        # input
        img:np.ndarray = read_image(img_path,resize)

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
                
        return prediction/1000
    #region Ihatemylife
    def batch_generate_monodepth(self, input_path, output_path,resize):
  
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
            prediction = self.generate_monodepth(os.path.join(input_path,img_name),resize)
            write_depth(filename, prediction, bits=2, absolute_depth=self.absolute_depth)
        print("finished")
    
    def img_db_process(self,input_path,dataset:str,resize):
        if(dataset == "Mapfree"):
            self.mapfree_img_process(input_path,resize)
        else:
            #TODO
            raise NotImplementedError("Teehee")
    
    def mapfree_img_process(self,input_path,resize):
        depth_path = os.path.join(input_path,'depth')
        img_path = os.path.join(input_path,"image")
        #If output folder exisit, return
        # if(os.path.exists(depth_path)):
        #     print("Skipped depth")
        #     return
        os.makedirs(depth_path, exist_ok=True)
        self.batch_generate_monodepth(img_path,depth_path,resize)
    #endregion
    
    def solo_generate_monodepth(self, input_path, output_path, resize):
        """
        The function received in a path to an image file and 
        generate the corresponding depth image to the output path
        :param input_path: /content/.../frame001.png
        :param output_path: /content/.../frame001 (no .png, it will be added in the prediction process)
        """
        if(os.path.exists(str(output_path)+".png") or os.path.exists(str(output_path)+".npy")):
            return
        os.makedirs(Path(output_path).parent,exist_ok=True)
        prediction = self.generate_monodepth(str(input_path),resize)
        # write_depth(str(output_path), prediction, bits=2, absolute_depth=self.absolute_depth)
        np.save(str(output_path)+".npy",prediction)
        