"""Parity check: ggml port vs the PyTorch reference, on the same inputs.

    python parity.py --ckpt ball.pt --gguf tracknet-f32.gguf --bin build/tracknet --frames DIR [--backend metal|cpu]

DIR holds consecutive video frames as jpg (sorted by name). Each test input is three consecutive
frames resized to the model resolution. Without --frames, random inputs are used.
"""
import argparse, glob, os, re, subprocess, sys, tempfile
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "training"))


def inputs(frames, w, h, n):
    if not frames:
        rng = np.random.default_rng(0)
        return [rng.random((9, h, w), dtype=np.float32) for _ in range(n)]
    from PIL import Image
    files = sorted(glob.glob(os.path.join(frames, "*.jpg")))
    step = max(1, (len(files) - 2) // n)
    out = []
    for i in range(0, len(files) - 2, step)[:n]:
        fr = [np.asarray(Image.open(f).convert("RGB").resize((w, h), Image.BILINEAR), dtype=np.float32) / 255 for f in files[i:i + 3]]
        out.append(np.concatenate([f.transpose(2, 0, 1) for f in fr], 0))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--gguf", required=True); ap.add_argument("--bin", required=True)
    ap.add_argument("--frames"); ap.add_argument("--backend", default="metal"); ap.add_argument("-n", type=int, default=10)
    ap.add_argument("--width", type=int, default=640); ap.add_argument("--height", type=int, default=360)
    ap.add_argument("--channels", default="0.5", help="channel width multiplier the checkpoint was trained with")
    a = ap.parse_args()

    os.environ.update(PP_W=str(a.width), PP_H=str(a.height), PP_WIDTH=a.channels)
    from tracknet import TrackNet
    net = TrackNet().eval()
    net.load_state_dict(torch.load(a.ckpt, map_location="cpu", weights_only=True))

    worst, dpx = 0.0, []
    with tempfile.TemporaryDirectory() as tmp:
        fi, fo = os.path.join(tmp, "in.bin"), os.path.join(tmp, "out.bin")
        for k, x in enumerate(inputs(a.frames, a.width, a.height, a.n)):
            with torch.no_grad():
                ref = net(torch.from_numpy(x)[None])[0, 0].numpy()
            x.tofile(fi)
            log = subprocess.run([a.bin, "-m", a.gguf, "-b", a.backend, "-i", fi, "-o", fo], capture_output=True, text=True)
            if log.returncode:
                sys.exit(log.stderr)
            got = np.fromfile(fo, dtype=np.float32).reshape(a.height, a.width)
            d = float(np.abs(got - ref).max())
            ry, rx = np.unravel_index(ref.argmax(), ref.shape); gy, gx = np.unravel_index(got.argmax(), got.shape)
            conf = 1 / (1 + np.exp(-ref.max()))
            dpx.append(float(np.hypot(rx - gx, ry - gy))); worst = max(worst, d)
            print(f"input {k}: max |logit diff| {d:.2e} | PyTorch peak ({rx},{ry}) conf {conf:.3f} | ggml peak ({gx},{gy})")
    print(f"{a.backend}: {len(dpx)} inputs, worst logit diff {worst:.2e}, peak position diff max {max(dpx):.1f} px")


if __name__ == "__main__":
    main()
