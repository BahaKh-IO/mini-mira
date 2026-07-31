import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


import torch
from mini_mira.bottleneck import BottleneckConfig , MyBottleneck
from mini_mira.decoder import DecoderConfig, MyDecoder
from mini_mira.world_model import WorldModelConfig, MyWorldModel

config = BottleneckConfig()
module = MyBottleneck(config)
x=torch.randn(2,40,1024,18,32)
z=module(x)


print("Day 1:", z.shape)

assert z.shape == (2, 20, 32, 9, 16)

decoder_config = DecoderConfig()
decoder = MyDecoder(decoder_config)
z2 = torch.randn(2, 20, 32, 9, 16)
out = decoder(z2)

print("Day 2:", out.shape)

assert out.shape == (2, 40, 3, 288, 512), f"got {out.shape}"

wm_config = WorldModelConfig()
world_model = MyWorldModel(wm_config)
z_t = torch.randn(2, 20, 32, 9, 16)
pred_v = world_model(z_t, actions=None, tau=None)

print("Day 3:", pred_v.shape)

assert pred_v.shape == z_t.shape, f"got {pred_v.shape}"