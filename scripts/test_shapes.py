import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


import torch
from mini_mira.codec.bottleneck import StridedConvBottleneckConfig, MyBottleneck
from mini_mira.codec.decoder import ViTDecoderConfig, ViTVideoDecoder
from mini_mira.world_model.diffusion_transformer import LatentWorldModelConfig, DiffusionTransformer
from mini_mira.pipeline import PipelineConfig, MyPipeline

with torch.inference_mode():
    config = StridedConvBottleneckConfig()
    module = MyBottleneck(config)
    x=torch.randn(2,40,config.dino_dim,18,32)
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
    clean_past = torch.randn(2, 20, 32, 9, 16)  # same shape as z_t -- the frame-before content
    a = torch.randn(2, 20, 256)  # already-encoded actions -- ActionEncoder lives on MyPipeline, not here
    pred_v = world_model(z_t, a=a, tau=tau, clean_past=clean_past)

    print("Day 3:", pred_v.shape)

    assert pred_v.shape == z_t.shape, f"got {pred_v.shape}"

    pipeline_config = PipelineConfig()
    pipeline = MyPipeline(pipeline_config)
    dino_features = torch.randn(2, 40, pipeline_config.bottleneck.dino_dim, 18, 32)
    # raw key-presses at video frame rate: (T_latent - 1) * temporal_downsampling = (20-1)*2 = 38,
    # not 40 -- there's no real action for the frame before frame 0 (see ActionEncoder's
    # initial_action_token).
    actions = torch.randint(0, 2, (2, 38, 9))
    video = pipeline(dino_features, actions)

    print("Day 4:", video.shape)

    assert video.shape == (2, 40, 3, 288, 512), f"got {video.shape}"