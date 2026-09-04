"""Interactive, browser-controlled rollout server for a trained V-JEPA world-model checkpoint.

Real, important limitation, not hidden: DiffusionTransformer has no streaming kv-cache (see
LatentWorldModel.rollout's own docstring), so every new frame re-runs the ENTIRE sequence built so
far -- generation gets slower the longer a session runs, not a fixed per-frame cost. This is a
turn-based demo (press keys, click Generate, wait a real few seconds, see the result), not smooth
real-time control -- that would need consistency-distillation-style speedups this project
deliberately never built (see notes/vjepa_codec_quality_research.md's PSD discussion).

generate_next_frame() is the one piece of new generation logic here -- a single-frame-at-a-time
variant of LatentWorldModel.rollout()'s inner loop, verified to produce numerically IDENTICAL
latents to rollout() itself for the same inputs (see scripts/verify_interactive_rollout.py). Kept
in this script, not added to LatentWorldModel, since nothing else in the project needs incremental
(unknown-length-upfront) generation -- rollout() already covers every existing training/eval use.

Run on the GPU box, then reach it from your own browser via an SSH tunnel (safer than exposing the
box's port directly on a shared network):
    ssh -L 5000:localhost:5000 salem@<box> -N   # separate terminal, leave running
    python scripts/serve_interactive_vjepa.py --codec-checkpoint ... --wm-checkpoint ... \\
        --latent-stats ... --index-path ... --require-pretrained-vjepa
Then open http://localhost:5000 in your own browser.
"""

import argparse
import io
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "mira" / "src"))

import torch
from mira.data.actions import DEFAULT_RL_KEYS
from mira.data.training_loader import create_loader

from mini_mira.codec.vjepa import DEFAULT_VJEPA_LAYERS, VjepaModel
from mini_mira.ml.config_loading import load_pipeline_config
from mini_mira.world_model.latent_world_model import LatentWorldModel
from mini_mira.world_model.rollout_visualization import video_to_uint8, write_video_ffmpeg


def generate_next_frame(
    model: LatentWorldModel, z_so_far: torch.Tensor, key_presses_so_far: torch.Tensor,
    n_diffusion_steps: int = 4, schedule_type: str = "linear",
) -> torch.Tensor:
    """z_so_far: (b,t,c,h,w) normalized latents already generated/seeded. key_presses_so_far:
    (b, raw_t, num_keys) real action history covering (and including) the window for the NEW
    frame about to be generated. Returns z_so_far with exactly one more frame appended (t+1).

    Mirrors LatentWorldModel.rollout()'s inner-loop body for a single iteration at k=t (the new
    frame's index) -- same math, restructured to not need the final rollout length known upfront.
    Verified against rollout() directly, see scripts/verify_interactive_rollout.py.
    """
    from mira.world_model.schedule import build_inference_schedule  # noqa: PLC0415 -- see rollout()'s own import

    b, t, c, h, w = z_so_far.shape
    td = model.temporal_downsampling
    t_final = t + 1
    off = td - 1
    n_action_steps = (t_final - 1) * td
    assert key_presses_so_far.shape[1] >= off + n_action_steps, (
        f"need at least {off + n_action_steps} raw action steps for {t_final} total latent frames, "
        f"got {key_presses_so_far.shape[1]}"
    )
    key_presses = key_presses_so_far[:, off : off + n_action_steps].long()
    a = model.action_encoder(key_presses)

    bos = model.bos.view(1, 1, c, 1, 1).expand(b, 1, c, h, w)
    z_t = torch.cat([z_so_far, torch.randn(b, 1, c, h, w, device=z_so_far.device, dtype=z_so_far.dtype)], dim=1)
    k = t  # index of the new frame

    timesteps = build_inference_schedule(n_diffusion_steps, z_so_far.device, schedule_type)
    delta_ts = timesteps[1:] - timesteps[:-1]
    clean_past = torch.cat([bos, z_t[:, :k]], dim=1)
    tau = torch.ones(b, k + 1, 1, 1, 1, device=z_so_far.device, dtype=z_so_far.dtype)

    with torch.no_grad():
        for timestep, delta_t in zip(timesteps[:-1], delta_ts):
            tau[:, k] = timestep
            pred_v = model.world_model(z_t[:, : k + 1], a=a[:, : k + 1], tau=tau, clean_past=clean_past)
            z_t[:, k] = z_t[:, k] + delta_t * pred_v[:, k]

    return z_t


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/scaled_300m_vjepa.yaml")
    parser.add_argument("--codec-checkpoint", required=True)
    parser.add_argument("--wm-checkpoint", required=True)
    parser.add_argument("--latent-stats", required=True)
    parser.add_argument("--index-path", required=True, help="Real held-out dataset dir, to seed context frames from")
    parser.add_argument("--require-pretrained-vjepa", action="store_true")
    parser.add_argument("--height", type=int, default=448)
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--context-frames", type=int, default=4, help="Real raw video frames to seed a session from")
    parser.add_argument("--target-fps", type=int, default=20)
    parser.add_argument("--diffusion-steps", type=int, default=4)
    parser.add_argument("--schedule-type", choices=["linear", "linear_quadratic"], default="linear")
    parser.add_argument("--precision", choices=["fp16-hybrid", "bf16"], default="bf16")
    parser.add_argument("--port", type=int, default=5000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from flask import Flask, jsonify, request, send_file  # noqa: PLC0415 -- only this script needs it

    config = load_pipeline_config(args.config)
    latent_stats = json.loads(Path(args.latent_stats).read_text())
    vjepa = VjepaModel(
        require_pretrained=args.require_pretrained_vjepa,
        last_layer_only=False, layer_indices=DEFAULT_VJEPA_LAYERS,
    ).cuda()
    model = LatentWorldModel(
        config.world_model, config.bottleneck, config.decoder, num_keys=config.num_keys,
        codec_checkpoint=args.codec_checkpoint, latent_mean=latent_stats["latent_mean"],
        latent_std=latent_stats["latent_std"], dino=vjepa,
    ).cuda()
    ckpt = torch.load(args.wm_checkpoint, map_location="cpu", weights_only=False)
    model.world_model.load_state_dict(ckpt["world_model"])
    model.action_encoder.load_state_dict(ckpt["action_encoder"])
    with torch.no_grad():
        model.bos.copy_(ckpt["bos"].to(model.bos.device))
    model.eval()
    print(f"Loaded world-model checkpoint from step {ckpt['step']}")

    dtype = torch.bfloat16 if args.precision == "bf16" else torch.float16
    td = model.temporal_downsampling
    assert args.context_frames % td == 0, f"--context-frames must be a multiple of {td} (temporal_downsampling)"

    # One real held-out clip, pulled once at import time -- reused to seed every /reset (a fresh
    # clip per reset would need a live dataloader iterator kept around; out of scope for a demo).
    loader = create_loader(
        index_path=args.index_path, clip_len=args.context_frames, target_fps=args.target_fps,
        n_players=1, batch_size=1, frame_size=(args.height, args.width), seed=123,
    )
    seed_batch, _metadata = next(iter(loader))

    # Single global session -- a real multi-user server would key this by session id; deliberately
    # not built here, this is a one-person local demo.
    session: dict = {}

    def _decode_last_frames(z: torch.Tensor, n_raw_frames: int) -> torch.Tensor:
        """Decodes the FULL sequence (causal temporal attention means a frame's decode depends on
        every frame before it -- can't decode a suffix in isolation) but only returns the last
        n_raw_frames of pixels, keeping the response payload small regardless of session length."""
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=dtype):
            video = model.decode_to_video(z)
        return video[:, -n_raw_frames:]

    app = Flask(__name__)

    @app.route("/")
    def index():
        return _INDEX_HTML

    @app.route("/reset", methods=["POST"])
    def reset():
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=dtype):
            batch = seed_batch.to("cuda", non_blocking=True)
            z_context, _a = model._encode(batch)  # noqa: SLF001 -- established pattern in this codebase's own scripts
        session["z"] = z_context
        session["key_presses"] = batch.actions.key_presses.clone()
        session["all_frames"] = [video_to_uint8(model.decode_to_video(z_context))[0].cpu()]
        frame = _decode_last_frames(z_context, td)
        return jsonify({"frame_png_b64": _frame_to_png_b64(frame[0, -1]), "n_frames": z_context.shape[1]})

    @app.route("/step", methods=["POST"])
    def step():
        if "z" not in session:
            return jsonify({"error": "call /reset first"}), 400
        keys_held = set(request.json.get("keys", []))
        new_window = torch.zeros(1, td, len(DEFAULT_RL_KEYS), dtype=torch.int32)
        for i, key in enumerate(DEFAULT_RL_KEYS):
            if key in keys_held:
                new_window[:, :, i] = 1
        session["key_presses"] = torch.cat([session["key_presses"], new_window.to(session["key_presses"].device)], dim=1)
        with torch.autocast(device_type="cuda", dtype=dtype):
            session["z"] = generate_next_frame(
                model, session["z"], session["key_presses"].cuda(),
                n_diffusion_steps=args.diffusion_steps, schedule_type=args.schedule_type,
            )
            frame = _decode_last_frames(session["z"], td)
        session["all_frames"].append(video_to_uint8(frame)[0].cpu())
        return jsonify({"frame_png_b64": _frame_to_png_b64(frame[0, -1]), "n_frames": int(session["z"].shape[1])})

    @app.route("/download")
    def download():
        if "all_frames" not in session or not session["all_frames"]:
            return jsonify({"error": "nothing generated yet"}), 400
        full_video = torch.cat(session["all_frames"], dim=0)
        out_path = Path("interactive_session.mp4")
        write_video_ffmpeg(str(out_path), full_video, fps=args.target_fps)
        return send_file(out_path, as_attachment=True)

    print(f"Serving on http://0.0.0.0:{args.port} -- tunnel with: ssh -L {args.port}:localhost:{args.port} <this box> -N")
    app.run(host="0.0.0.0", port=args.port, threaded=False)


def _frame_to_png_b64(frame_uint8: torch.Tensor) -> str:
    """frame_uint8: (3,H,W) uint8. Returns a base64-encoded PNG for direct <img src> embedding."""
    import base64

    from PIL import Image

    img = Image.fromarray(frame_uint8.permute(1, 2, 0).numpy())
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


_INDEX_HTML = """<!doctype html>
<html><head><title>mini_mira -- interactive V-JEPA rollout</title>
<style>
body { font-family: sans-serif; background: #111; color: #eee; text-align: center; padding: 20px; }
#frame { border: 2px solid #444; max-width: 90vw; image-rendering: pixelated; }
.key { display: inline-block; padding: 8px 14px; margin: 3px; border-radius: 6px; background: #333; }
.key.held { background: #2a6; }
button { padding: 10px 20px; font-size: 16px; margin: 10px; cursor: pointer; }
#status { color: #999; }
</style></head>
<body>
<h2>mini_mira -- interactive V-JEPA world-model rollout</h2>
<p>Generation is NOT real-time -- each turn re-runs the whole sequence so far (no kv-cache),
so it gets slower as the session grows. Hold your keys, click Generate, wait a real few seconds.</p>
<img id="frame" src="" /><br/>
<div id="keys"></div>
<button onclick="doReset()">Reset session</button>
<button onclick="doStep()">Generate next</button>
<a href="/download"><button>Download session video</button></a>
<p id="status">Click Reset to start.</p>
<script>
const KEYS = ["W","A","S","D","Q","E","Space","LShiftKey","LControlKey"];
const CODE_TO_KEY = {KeyW:"W",KeyA:"A",KeyS:"S",KeyD:"D",KeyQ:"Q",KeyE:"E",Space:"Space",
  ShiftLeft:"LShiftKey",ShiftRight:"LShiftKey",ControlLeft:"LControlKey",ControlRight:"LControlKey"};
let held = new Set();
const keysDiv = document.getElementById("keys");
KEYS.forEach(k => { const d = document.createElement("span"); d.className="key"; d.id="k_"+k; d.textContent=k; keysDiv.appendChild(d); });
function render() { KEYS.forEach(k => document.getElementById("k_"+k).className = "key" + (held.has(k) ? " held" : "")); }
document.addEventListener("keydown", e => { const k = CODE_TO_KEY[e.code]; if (k) { held.add(k); render(); e.preventDefault(); } });
document.addEventListener("keyup", e => { const k = CODE_TO_KEY[e.code]; if (k) { held.delete(k); render(); e.preventDefault(); } });
async function doReset() {
  document.getElementById("status").textContent = "Resetting...";
  const r = await fetch("/reset", {method:"POST"});
  const j = await r.json();
  document.getElementById("frame").src = "data:image/png;base64," + j.frame_png_b64;
  document.getElementById("status").textContent = "Ready. " + j.n_frames + " latent frames so far.";
}
async function doStep() {
  document.getElementById("status").textContent = "Generating... (this can take several seconds, longer as the session grows)";
  const r = await fetch("/step", {method:"POST", headers:{"Content-Type":"application/json"}, body: JSON.stringify({keys: Array.from(held)})});
  const j = await r.json();
  if (j.error) { document.getElementById("status").textContent = "Error: " + j.error; return; }
  document.getElementById("frame").src = "data:image/png;base64," + j.frame_png_b64;
  document.getElementById("status").textContent = "Ready. " + j.n_frames + " latent frames so far.";
}
</script>
</body></html>
"""


if __name__ == "__main__":
    main()
