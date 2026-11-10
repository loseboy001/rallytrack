"""Quantization-aware-trained ball tracker (TorchScript from training/train_qat.py) -> Core ML, 8-bit.

    python convert_qat.py traced.pt BallTracker_q8.mlpackage [--ref BallTracker.mlpackage --videos DIR]

train_qat.py inserts fake-quantization layers (coremltools LinearQuantizer, symmetric, weights and
activations), fine-tunes, and writes a finalized TorchScript module. coremltools turns those layers
into real 8-bit ops here.

With --ref and --videos, the 8-bit model is compared with the fp16 model on frame triplets sampled
from the mp4 files in DIR: how many balls each one detects (confidence >= 0.5) and how far apart
the positions are.
"""
import argparse, glob, os, random
import numpy as np
import torch
import coremltools as ct

NAMES = ("prev2", "prev1", "cur")
W, H = 640, 360


def triplets(video, n, seed):
    import cv2
    cap = cv2.VideoCapture(video); total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); rng = random.Random(seed); out = []
    for _ in range(n):
        k = rng.randint(2, max(3, total - 2)); fr = []
        for j in (k - 2, k - 1, k):
            cap.set(cv2.CAP_PROP_POS_FRAMES, j); ok, f = cap.read()
            if not ok: break
            f = cv2.cvtColor(cv2.resize(f, (W, H), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
            fr.append(f.transpose(2, 0, 1)[None].astype(np.float16) / 255)
        if len(fr) == 3: out.append(dict(zip(NAMES, fr)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("traced"); ap.add_argument("out")
    ap.add_argument("--ref", help="fp16 model to compare with"); ap.add_argument("--videos", help="directory of mp4 files")
    a = ap.parse_args()

    traced = torch.jit.load(a.traced).eval()
    ml = ct.convert(traced, inputs=[ct.TensorType(name=n, shape=(1, 3, H, W), dtype=np.float16) for n in NAMES],
                    outputs=[ct.TensorType(name="peak", dtype=np.float32)], convert_to="mlprogram",
                    compute_precision=ct.precision.FLOAT16, minimum_deployment_target=ct.target.iOS17)
    ml.save(a.out)
    print("saved", a.out)
    if not (a.ref and a.videos):
        return

    test = sum((triplets(v, 8, 2) for v in sorted(glob.glob(os.path.join(a.videos, "*.mp4")))), [])
    ref = ct.models.MLModel(a.ref)
    p16 = np.stack([ref.predict(x)["peak"][0] for x in test]); p8 = np.stack([ml.predict(x)["peak"][0] for x in test])
    d16, d8 = p16[:, 2] >= 0.5, p8[:, 2] >= 0.5; both = d16 & d8
    dist = np.hypot(p16[both, 0] - p8[both, 0], p16[both, 1] - p8[both, 1])
    print(f"{len(test)} inputs | detected: fp16 {int(d16.sum())}, 8-bit {int(d8.sum())}, both {int(both.sum())}, "
          f"fp16 only {int((d16 & ~d8).sum())}, 8-bit only {int((d8 & ~d16).sum())}")
    if len(dist):
        print(f"position diff where both detect: median {np.median(dist):.2f} px, more than 4 px apart: {int((dist > 4).sum())}")


if __name__ == "__main__":
    main()
