import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch
from mini_mira.codec.decoder import ViTDecoderConfig, ViTVideoDecoder
from mini_mira.world_model.diffusion_transformer import LatentWorldModelConfig, DiffusionTransformer

torch.manual_seed(0)

with torch.no_grad():
    # --- world model: causality ---
    wm_config = LatentWorldModelConfig()
    world_model = DiffusionTransformer(wm_config)
    world_model.eval()

    z = torch.randn(1, 6, 32, 4, 4)
    tau = torch.rand(1, 6, 1, 1, 1)  # fixed, shared tau -- these checks aren't testing tau
    clean_past = torch.randn(1, 6, 32, 4, 4)  # fixed, shared clean_past -- ditto
    a = torch.randn(1, 6, 256)  # fixed, shared already-encoded actions -- ditto
    out1 = world_model(z, a=a, tau=tau, clean_past=clean_past)

    # Perturb only the LAST frame. Earlier outputs must be byte-identical --
    # a causal model cannot let a future frame change the past.
    z_perturbed = z.clone()
    z_perturbed[:, -1] += 100.0
    out2 = world_model(z_perturbed, a=a, tau=tau, clean_past=clean_past)

    earlier_diff = (out1[:, :-1] - out2[:, :-1]).abs().max().item()
    last_diff = (out1[:, -1] - out2[:, -1]).abs().max().item()
    assert earlier_diff == 0.0, f"causality broken: earlier frames changed by {earlier_diff}"
    assert last_diff > 0.0, "last frame did not change at all -- suspicious"
    print(f"[PASS] world model causality: earlier frames diff={earlier_diff}, last frame diff={last_diff:.4f}")

    # --- world model: position sensitivity ---
    # Swap frames 0 and 1 in the INPUT. If RoPE is working, the output must NOT
    # be the same swap applied to the original output -- if it were, the model
    # would be treating position as irrelevant (the classic "all-zero freqs" bug).
    z_swapped = z.clone()
    z_swapped[:, [0, 1]] = z_swapped[:, [1, 0]]
    out3 = world_model(z_swapped, a=a, tau=tau, clean_past=clean_past)

    naive_swap = out1.clone()
    naive_swap[:, [0, 1]] = naive_swap[:, [1, 0]]
    swap_diff = (out3 - naive_swap).abs().max().item()
    assert swap_diff > 0.0, "position-blind: swapping frames just swapped the output too"
    print(f"[PASS] world model position sensitivity: diff from naive swap={swap_diff:.6f}")

    # --- decoder: position sensitivity ---
    dec_config = ViTDecoderConfig()
    decoder = ViTVideoDecoder(dec_config)
    decoder.eval()

    z2 = torch.randn(1, 4, 32, 4, 4)
    dec_out1 = decoder(z2)

    z2_swapped = z2.clone()
    z2_swapped[:, [0, 1]] = z2_swapped[:, [1, 0]]
    dec_out2 = decoder(z2_swapped)

    dec_naive_swap = dec_out1.clone()
    dec_naive_swap[:, [0, 1]] = dec_naive_swap[:, [1, 0]]
    dec_swap_diff = (dec_out2 - dec_naive_swap).abs().max().item()
    assert dec_swap_diff > 0.0, "decoder position-blind: swapping frames just swapped the output too"
    print(f"[PASS] decoder position sensitivity: diff from naive swap={dec_swap_diff:.6f}")

print("\nAll checks passed.")
