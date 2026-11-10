"""Ball tracker (PyTorch) -> Core ML, fp16, for the Apple Neural Engine.

    python convert.py ball.pt BallTracker.mlpackage [--frames DIR] [--bench]

Two choices that matter for speed in the app:

* The three frames are separate fp16 inputs (prev2, prev1, cur). The app converts every video frame
  once and reuses the array for the next two predictions, instead of rebuilding a 9-channel input.
* The model returns only the heatmap peak (x, y, confidence). The arg-max runs inside the model, so
  the 640x360 heatmap never crosses back to the CPU. It is done in two steps (per row, then over
  rows) because a flat index goes up to 230 400, which fp16 cannot represent exactly.

--frames DIR  parity check against PyTorch on consecutive jpg frames from DIR
--bench       per-frame time on this Mac for each compute unit
"""
import argparse, glob, os, sys, time
import numpy as np
import torch, torch.nn as nn
import coremltools as ct

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "training"))
NAMES = ("prev2", "prev1", "cur")


class Peak3(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, f0, f1, f2):
        p = torch.sigmoid(self.net(torch.cat([f0, f1, f2], 1)))[:, 0]   # B, H, W
        rowmax, colidx = p.max(2)                                       # B, H
        conf, ys = rowmax.max(1)                                        # B
        xs = colidx.gather(1, ys[:, None])[:, 0]
        return torch.stack([xs.float(), ys.float(), conf], 1)


def triplets(frames, w, h, n):
    from PIL import Image
    files = sorted(glob.glob(os.path.join(frames, "*.jpg")))
    step = max(1, (len(files) - 2) // n)
    for i in list(range(0, len(files) - 2, step))[:n]:
        yield [np.asarray(Image.open(f).convert("RGB").resize((w, h), Image.BILINEAR), dtype=np.float32).transpose(2, 0, 1)[None] / 255
               for f in files[i:i + 3]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt"); ap.add_argument("out")
    ap.add_argument("--width", type=int, default=640); ap.add_argument("--height", type=int, default=360)
    ap.add_argument("--channels", default="0.5", help="channel width multiplier the checkpoint was trained with")
    ap.add_argument("--frames"); ap.add_argument("--bench", action="store_true")
    a = ap.parse_args()

    os.environ.update(PP_W=str(a.width), PP_H=str(a.height), PP_WIDTH=a.channels)
    from tracknet import TrackNet
    net = TrackNet().eval()
    net.load_state_dict(torch.load(a.ckpt, map_location="cpu", weights_only=True))
    mod = Peak3(net).eval()

    shape = (1, 3, a.height, a.width)
    traced = torch.jit.trace(mod, [torch.rand(*shape) for _ in NAMES])
    ml = ct.convert(traced, inputs=[ct.TensorType(name=n, shape=shape, dtype=np.float16) for n in NAMES],
                    outputs=[ct.TensorType(name="peak", dtype=np.float32)], convert_to="mlprogram",
                    compute_precision=ct.precision.FLOAT16, minimum_deployment_target=ct.target.iOS17)
    ml.short_description = f"Ball tracker {a.width}x{a.height}: peak (x, y, confidence) in model pixels"
    ml.save(a.out)
    print("saved", a.out)

    if a.frames:
        d = []
        with torch.no_grad():
            for fr in triplets(a.frames, a.width, a.height, 20):
                t = mod(*[torch.from_numpy(f) for f in fr])[0].numpy()
                c = ml.predict({n: f.astype(np.float16) for n, f in zip(NAMES, fr)})["peak"][0]
                d.append(float(np.hypot(*(t[:2] - c[:2]))))
        print(f"parity vs PyTorch on {len(d)} inputs: peak position diff max {max(d):.2f} px, mean {np.mean(d):.2f} px")

    if a.bench:
        x = {n: np.random.rand(*shape).astype(np.float16) for n in NAMES}
        for cu in ("CPU_ONLY", "CPU_AND_GPU", "CPU_AND_NE", "ALL"):
            m = ct.models.MLModel(a.out, compute_units=getattr(ct.ComputeUnit, cu))
            for _ in range(5): m.predict(x)
            t0 = time.perf_counter()
            for _ in range(40): m.predict(x)
            print(f"{cu:12s} {(time.perf_counter() - t0) / 40 * 1000:6.1f} ms per frame")


if __name__ == "__main__":
    main()
