import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch
from mini_mira.pipeline import PipelineConfig, MyPipeline
from mini_mira.world_model import LatentWorldModelConfig, DiffusionTransformer

torch.manual_seed(0)

with torch.no_grad():
    # --- tau sensitivity ---
    # Same z_t, same clean_past, two different tau values -> the predicted velocities must
    # differ. If they don't, cond isn't reaching the blocks (e.g. still zeros, or AdaLN wired
    # wrong).
    wm_config = LatentWorldModelConfig()
    world_model = DiffusionTransformer(wm_config)
    world_model.eval()

    z = torch.randn(1, 6, 32, 4, 4)
    clean_past = torch.randn(1, 6, 32, 4, 4)  # fixed, shared -- this section isn't testing it
    a = torch.randn(1, 6, 256)  # fixed, shared already-encoded actions -- ditto

    tau_a = torch.zeros(1, 6, 1, 1, 1)
    tau_b = torch.ones(1, 6, 1, 1, 1)
    out_tau_a = world_model(z, a=a, tau=tau_a, clean_past=clean_past)
    out_tau_b = world_model(z, a=a, tau=tau_b, clean_past=clean_past)
    tau_diff = (out_tau_a - out_tau_b).abs().max().item()
    assert tau_diff > 0.0, "tau-blind: two different tau values produced the same velocity"
    print(f"[PASS] tau sensitivity: max abs diff between tau=0 and tau=1 outputs={tau_diff:.6f}")

    # --- clean_past sensitivity ---
    # Same z_t, same tau, two different clean_past values -> the predicted velocities must
    # differ. If they don't, clean_past isn't reaching the network -- this is the exact bug
    # that was measured before this task: two different input videos producing bit-identical
    # pipeline output.
    tau = torch.rand(1, 6, 1, 1, 1)  # fixed, shared -- this section isn't testing it
    clean_past_a = torch.randn(1, 6, 32, 4, 4)
    clean_past_b = torch.randn(1, 6, 32, 4, 4)
    out_cp_a = world_model(z, a=a, tau=tau, clean_past=clean_past_a)
    out_cp_b = world_model(z, a=a, tau=tau, clean_past=clean_past_b)
    cp_diff = (out_cp_a - out_cp_b).abs().max().item()
    assert cp_diff > 0.0, "clean_past-blind: two different clean_past values produced the same velocity"
    print(f"[PASS] clean_past sensitivity: max abs diff={cp_diff:.6f}")

    # Reverse sanity check: identical clean_past (everything else held fixed) must give
    # identical output -- confirms the model is deterministic, so the nonzero diff above is
    # really coming from clean_past and nothing else.
    out_cp_a2 = world_model(z, a=a, tau=tau, clean_past=clean_past_a)
    cp_same_diff = (out_cp_a - out_cp_a2).abs().max().item()
    assert cp_same_diff == 0.0, "non-deterministic: identical inputs produced different output"
    print(f"[PASS] determinism: identical clean_past -> diff={cp_same_diff}")

    # --- actions sensitivity ---
    # Same z_t, same tau, same clean_past, two different already-encoded action tensors ->
    # the predicted velocities must differ. If they don't, a isn't reaching cond.
    a_1 = torch.randn(1, 6, 256)
    a_2 = torch.randn(1, 6, 256)
    out_a_1 = world_model(z, a=a_1, tau=tau, clean_past=clean_past_a)
    out_a_2 = world_model(z, a=a_2, tau=tau, clean_past=clean_past_a)
    a_diff = (out_a_1 - out_a_2).abs().max().item()
    assert a_diff > 0.0, "action-blind: two different action embeddings produced the same velocity"
    print(f"[PASS] actions sensitivity: max abs diff={a_diff:.6f}")

    # --- end-to-end: same video, different actions, through the real pipeline and ActionEncoder ---
    # Same input video, same noise seed, two different RAW key-press sequences -> the decoded
    # output must differ. This exercises ActionEncoder itself (per-key embeddings, temporal
    # pooling, initial_action_token), not just the AdaLN pathway above.
    pipeline_actions = MyPipeline(PipelineConfig())
    pipeline_actions.eval()

    same_video = torch.randn(1, 4, 1024, 4, 4)
    keys_all_zero = torch.zeros(1, 2, 9, dtype=torch.long)  # no keys pressed, ever
    keys_all_w = torch.zeros(1, 2, 9, dtype=torch.long)
    keys_all_w[:, :, 0] = 1  # key index 0 held down the whole clip (no name<->index mapping exists in mini_mira)

    torch.manual_seed(0)
    out_keys_a = pipeline_actions(same_video, keys_all_zero)

    torch.manual_seed(0)
    out_keys_b = pipeline_actions(same_video, keys_all_w)

    actions_pipeline_diff = (out_keys_a - out_keys_b).abs().max().item()
    assert actions_pipeline_diff > 0.0, "pipeline ignores actions -- ActionEncoder isn't reaching the decoded output"
    print(f"[PASS] end-to-end actions: same video/seed, different keys -> output diff={actions_pipeline_diff:.6f}")

    # --- end-to-end: the actual bug, through the real pipeline ---
    # Same noise seed, two completely different input videos -> the DECODED OUTPUT must now
    # differ. Before clean_past existed, this printed exactly 0.0 (the encoded input was
    # computed and then discarded -- see pipeline.py's docstring for the full story).
    pipeline = MyPipeline(PipelineConfig())
    pipeline.eval()

    video_a = torch.randn(1, 4, 1024, 4, 4)
    video_b = torch.zeros(1, 4, 1024, 4, 4)
    # raw_t = (T_latent - 1) * temporal_downsampling = (2 - 1) * 2 = 2, for a 4-frame clip.
    shared_actions = torch.randint(0, 2, (1, 2, 9))

    torch.manual_seed(0)
    out_a = pipeline(video_a, shared_actions)

    torch.manual_seed(0)  # reset so the noise draw matches the run above exactly
    out_b = pipeline(video_b, shared_actions)

    pipeline_diff = (out_a - out_b).abs().max().item()
    assert pipeline_diff > 0.0, "pipeline still ignores its input -- clean_past isn't reaching the decoded output"
    print(f"[PASS] end-to-end: same noise seed, different input videos -> output diff={pipeline_diff:.6f}")

print("\nAll conditioning checks passed.")
