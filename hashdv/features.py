"""Frozen timm DINOv2-B/14 features using the paper's preprocessing."""
from pathlib import Path
import numpy as np


def read_rgb(path):
    import cv2
    image = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot decode image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


class Encoder:
    def __init__(self, checkpoint=None, device=None):
        import torch
        import timm
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = timm.create_model("vit_base_patch14_dinov2.lvd142m", pretrained=checkpoint is None,
                                      dynamic_img_size=True)
        if checkpoint is not None:
            from safetensors.torch import load_file
            self.model.load_state_dict(load_file(str(Path(checkpoint))), strict=True)
        self.model = self.model.to(self.device).eval()

    def encode(self, images, dense=False):
        import cv2
        import torch
        images = [read_rgb(p) if isinstance(p, (str, Path)) else p for p in images]
        size = 448 if dense else 224
        x = torch.from_numpy(np.stack([cv2.resize(im, (size, size), interpolation=cv2.INTER_LINEAR)
                                      for im in images])).to(self.device).permute(0, 3, 1, 2).float() / 255
        mean = torch.tensor([.485, .456, .406], device=self.device).view(1, 3, 1, 1)
        std = torch.tensor([.229, .224, .225], device=self.device).view(1, 3, 1, 1)
        with torch.inference_mode():
            if self.device.type == "cuda":
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    tokens = self.model.forward_features((x - mean) / std)
            else:
                tokens = self.model.forward_features((x - mean) / std)
            tokens = tokens.float()
            if dense:
                return torch.nn.functional.normalize(tokens[:, self.model.num_prefix_tokens:], dim=2).half().cpu()
            return torch.nn.functional.normalize(tokens[:, 0], dim=1).cpu()
