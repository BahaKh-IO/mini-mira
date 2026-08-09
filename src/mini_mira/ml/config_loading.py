"""Load a PipelineConfig from a named YAML preset file.

Lightweight config system ("Option A"): one YAML file is one full preset mirroring
PipelineConfig's whole nested shape (bottleneck / world_model / decoder sections, plus the
top-level n_diffusion_steps / num_keys fields) -- not mira's own per-component-file
composition system (no Hydra, no config groups, no CLI overrides). See configs/small.yaml
(mirrors today's dataclass defaults exactly) and configs/scaled_300m.yaml (scaled toward a
~300M parameter target) at the repo root.

Each section is keyword-unpacked straight into its dataclass's constructor -- an unrecognized
key in that section's YAML (a typo) raises TypeError immediately, exactly as it would
constructing the dataclass by hand in Python.
"""

from pathlib import Path

import yaml

from mini_mira.codec.bottleneck import StridedConvBottleneckConfig
from mini_mira.codec.decoder import ViTDecoderConfig
from mini_mira.pipeline import PipelineConfig
from mini_mira.world_model.diffusion_transformer import LatentWorldModelConfig


def load_pipeline_config(yaml_path: str | Path) -> PipelineConfig:
    """Build a PipelineConfig by keyword-unpacking a YAML preset file's sections.

    A missing section falls back to that dataclass's own defaults; an unrecognized key
    anywhere raises TypeError rather than being silently dropped.
    """
    yaml_path = Path(yaml_path)
    data = yaml.safe_load(yaml_path.read_text()) or {}

    bottleneck_data = data.pop("bottleneck", {})
    world_model_data = data.pop("world_model", {})
    decoder_data = data.pop("decoder", {})

    return PipelineConfig(
        bottleneck=StridedConvBottleneckConfig(**bottleneck_data),
        world_model=LatentWorldModelConfig(**world_model_data),
        decoder=ViTDecoderConfig(**decoder_data),
        **data,  # remaining top-level keys: n_diffusion_steps, num_keys (typo -> TypeError)
    )
