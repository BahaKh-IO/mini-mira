import sys

import torch

# Bypass torch.hub.load's hubconf.py, which eagerly imports segmentation/detection/depth/dinotxt
# code (and their extra dependencies) mini_mira never uses. Importing the backbone function
# directly skips all of that and loads the exact same real code and weights.
sys.path.insert(0, r"C:\Users\Infoshop\.cache\torch\hub\facebookresearch_dinov3_main")
from dinov3.hub.backbones import dinov3_vitb16

weights = r"C:\Users\Infoshop\dino_weights\dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"

dino = dinov3_vitb16(pretrained=True, weights=weights)
dino.eval()

video = torch.randn(2, 3, 224, 224)      # 2 fake frames

with torch.no_grad():
    out = dino.get_intermediate_layers(video, n=1, norm=True, reshape=True)

print(type(out), len(out))
print(out[0].shape)