import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


import torch
from mini_mira.bottleneck import StridedConvBottleneckConfig, MyBottleneck
from mini_mira.decoder import ViTDecoderConfig, ViTVideoDecoder
from mini_mira.world_model import LatentWorldModelConfig, DiffusionTransformer
from mini_mira.pipeline import PipelineConfig, MyPipeline

with torch.inference_mode():
    config = StridedConvBottleneckConfig()
    module = MyBottleneck(config)
    x=torch.randn(2,40,1024,18,32)
    z=module(x)


    print("Day 1:", z.shape)

    assert z.shape == (2, 20, 32, 9, 16)


    decoder_config = ViTDecoderConfig()
    decoder = ViTVideoDecoder(decoder_config)
    z2 = torch.randn(2, 20, 32, 9, 16)
    out = decoder(z2)

    print("Day 2:", out.shape)

    assert out.shape == (2, 40, 3, 288, 512), f"got {out.shape}"

    wm_config = LatentWorldModelConfig()
    world_model = DiffusionTransformer(wm_config)
    z_t = torch.randn(2, 20, 32, 9, 16)
    tau = torch.rand(2, 20, 1, 1, 1)  # one noise level per latent frame, shape (b, t, 1, 1, 1)
    pred_v = world_model(z_t, actions=None, tau=tau)

    print("Day 3:", pred_v.shape)

    assert pred_v.shape == z_t.shape, f"got {pred_v.shape}"

    pipeline_config = PipelineConfig()
    pipeline = MyPipeline(pipeline_config)
    dino_features = torch.randn(2, 40, 1024, 18, 32)
    video = pipeline(dino_features, actions=None)

    print("Day 4:", video.shape)

    assert video.shape == (2, 40, 3, 288, 512), f"got {video.shape}"