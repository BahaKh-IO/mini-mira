"""Load config from YAML. Two separate axes: load_pipeline_config (architecture -- see
configs/small.yaml, configs/scaled_300m.yaml) and load_run_config (hyperparameters -- see
configs/runs/, ml/run_config.py). Same mechanism for both: keyword-unpack into a dataclass,
unrecognized key raises TypeError immediately rather than being silently dropped.
"""

import argparse
import dataclasses
from pathlib import Path
from typing import TypeVar

import yaml

from mini_mira.codec.bottleneck import StridedConvBottleneckConfig
from mini_mira.codec.decoder import ViTDecoderConfig
from mini_mira.pipeline import PipelineConfig
from mini_mira.world_model.diffusion_transformer import LatentWorldModelConfig

T = TypeVar("T")


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


def load_run_config(yaml_path: str | Path, config_cls: type[T]) -> T:
    """Build a WorldModelRunConfig/CodecRunConfig by keyword-unpacking a flat YAML file."""
    yaml_path = Path(yaml_path)
    data = yaml.safe_load(yaml_path.read_text()) or {}
    return config_cls(**data)


def apply_run_config(args: argparse.Namespace, run_config: T) -> None:
    """Fill every args field still at None from run_config. Explicit CLI values always win."""
    for f in dataclasses.fields(run_config):
        if getattr(args, f.name, None) is None:
            setattr(args, f.name, getattr(run_config, f.name))
