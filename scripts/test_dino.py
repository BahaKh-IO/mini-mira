"""Raw sanity check for the dinov3_vitb16 backbone itself, bypassing mini_mira's own DinoModel
wrapper entirely (no normalization, no freezing, no shape rearranging).

Distinct from verify_dino.py on purpose: verify_dino.py only ever tests mini_mira's own
DinoModel wrapper, so if it ever fails there's no way to tell from it alone whether the bug is
in that wrapper or in the underlying dinov3 library/weights themselves. This script reuses the
same loader helpers DinoModel itself uses (so there's no separate untested path-lookup logic to
go stale), but calls the raw upstream backbone directly -- if this fails too, the problem isn't
in mini_mira's code at all.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch
from mini_mira.codec.dino import _load_dinov3_backbone_fn, resolve_dino_weights

weights_path = resolve_dino_weights("dinov3_vitb16")
if weights_path is None:
    raise SystemExit(
        "RS_DINO_WEIGHTS_DIR is not set. Point it at the local directory containing "
        "dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth before running this script."
    )

dinov3_vitb16 = _load_dinov3_backbone_fn("dinov3_vitb16")
dino = dinov3_vitb16(pretrained=True, weights=str(weights_path))
dino.eval()

video = torch.randn(2, 3, 224, 224)      # 2 fake frames, no normalization -- this is deliberately
                                          # not what DinoModel.dino_forward feeds the real backbone

with torch.no_grad():
    out = dino.get_intermediate_layers(video, n=1, norm=True, reshape=True)

print(type(out), len(out))
print(out[0].shape)
