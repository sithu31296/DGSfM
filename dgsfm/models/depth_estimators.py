import torch
import numpy as np
from PIL import Image
from dgsfm.utils.geometry import focal2fov
from torch import nn, Tensor


class BaseDepthModel(nn.Module):
    def __init__(self, device=torch.device("cuda"), min_depth=0.0, max_depth=np.inf):
        super().__init__()
        self.device = device
        self.model = None
        self.max_depth = max_depth
        self.min_depth = min_depth
        self.dtype = np.float16

    def to_numpy(self, x: Tensor) -> np.ndarray:
        return x.detach().cpu().numpy().astype(self.dtype)
    
    def preprocess(self, image: np.ndarray, focal: float = None):
        image_tensor = torch.tensor(image, dtype=torch.float32, device=self.device).permute(2, 0, 1)
        return image_tensor

    def _forward(self, image_tensor: torch.Tensor):
        return self.model(image_tensor)

    def postprocess(self, output, image):   
        depth = self.to_numpy(output['depth'][0, 0])
        conf = np.ones_like(depth, dtype=self.dtype)
        intrinsic = np.eye(3, dtype=self.dtype)
        return depth, conf, intrinsic

    @torch.inference_mode()
    def forward(self, image: np.ndarray, focal: float = None):
        image_tensor = self.preprocess(image, focal)
        output = self._forward(image_tensor)
        return self.postprocess(output, image_tensor)


class UniDepth2(BaseDepthModel):
    def __init__(self):
        super().__init__()
        from unidepth.models import UniDepthV1 as UniDepth1Model, UniDepthV2 as UniDepth2Model
        from unidepth.utils.camera import Pinhole, Fisheye624
        self.model = UniDepth2Model.from_pretrained(f"lpiccinelli/unidepth-v2-vitl14").to(self.device).eval()

    def _forward(self, image_tensor: torch.Tensor):
        return self.model.infer(image_tensor)

    def postprocess(self, output, image):   
        depth = self.to_numpy(output['depth'][0, 0])
        conf = np.ones_like(depth, dtype=self.dtype)
        intrinsic = self.to_numpy(output['intrinsics'][0, 0])
        return depth, conf, intrinsic


class UniK3D(BaseDepthModel):
    def __init__(self):
        super().__init__()
        from unik3d import UniK3D as UniK3DImpl
        self.model = UniK3DImpl.from_pretrained(f"lpiccinelli/unik3d-vitl").to(self.device).eval()

    def _forward(self, image_tensor: torch.Tensor):
        return self.model.infer(image_tensor)


class MoGe1(BaseDepthModel):
    def __init__(self):
        super().__init__()
        from moge.model.v1 import MoGeModel as MoGeModelV1
        self.model = MoGeModelV1.from_pretrained("Ruicheng/moge-vitl").to(self.device).eval()

    def preprocess(self, image: np.ndarray, focal: float = None):
        image_tensor = torch.tensor(image / 255, dtype=torch.float32, device=self.device).permute(2, 0, 1)
        fov = np.rad2deg(focal2fov(focal, image_tensor.shape[2])) if focal is not None else None
        return image_tensor, fov
    
    def _forward(self, image_tensor: torch.Tensor, fov: float):
        return self.model.infer(image=image_tensor, fov_x=fov)

    def postprocess(self, output, image):   
        depth = self.to_numpy(output['depth'])
        conf = self.to_numpy(output['mask'])
        intrinsic = self.to_numpy(output['intrinsics'])
        intrinsic[0, :] *= image.shape[2]
        intrinsic[1, :] *= image.shape[1]
        return depth, conf, intrinsic

    @torch.inference_mode()
    def forward(self, image: np.ndarray, focal: float = None):
        image_tensor, fov = self.preprocess(image, focal)
        output = self._forward(image_tensor, fov)
        return self.postprocess(output, image_tensor)

    
class MoGe2(MoGe1):
    available_models = [
        "Ruicheng/moge-2-vitl",
        "Ruicheng/moge-2-vitl-normal",  # 60ms (A100 or RTX 3090, FP16)
        "Ruicheng/moge-2-vitb-normal",
        "Ruicheng/moge-2-vits-normal"
    ]
    def __init__(self):
        super().__init__()
        from moge.model.v2 import MoGeModel as MoGeModelV2
        self.model = MoGeModelV2.from_pretrained("Ruicheng/moge-2-vitl-normal").to(self.device).eval()


class MoGe3(MoGe1):
    def __init__(self):
        super().__init__()
        from moge.model.v3 import MoGeModel as MoGeModelV3
        self.model = MoGeModelV3.from_pretrained("./checkpoints/moge3_vitl.pt").to(self.device).eval()


class ZoeDepth(BaseDepthModel):
    def __init__(self):
        super().__init__()
        self.model = torch.hub.load("isl-org/ZoeDepth", "ZoeD_NK", pretrained=True).to(self.device).eval()

    def preprocess(self, image: np.ndarray, focal: float = None):
        return Image.fromarray(image)
    
    def _forward(self, image_tensor: torch.Tensor):
        return self.model.infer_pil(image_tensor)

    def postprocess(self, depth, image):   
        conf = np.ones_like(depth, dtype=self.dtype)
        intrinsic = np.eye(3, dtype=self.dtype)
        return depth.astype(self.dtype), conf, intrinsic
