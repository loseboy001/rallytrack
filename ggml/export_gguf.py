"""Export the ball tracker (PyTorch state_dict) to GGUF for the ggml port.

BatchNorm is folded into the preceding convolution, so every layer becomes conv + bias + ReLU.

    python export_gguf.py ball.pt tracknet-f32.gguf           # fp32 weights
    python export_gguf.py ball.pt tracknet-f16.gguf --f16     # fp16 conv kernels, fp32 biases

Tensor layout: a numpy array of shape (OC, IC, KH, KW) is stored by gguf with reversed
dimensions, which is exactly ggml's conv kernel layout [KW, KH, IC, OC].
"""
import argparse
import numpy as np
import torch
import gguf

BLOCKS = ["e1.0", "e1.1", "e2.0", "e2.1", "e3.0", "e3.1", "e3.2", "b.0", "b.1", "b.2",
          "d3.0", "d3.1", "d3.2", "d2.0", "d2.1", "d1.0", "d1.1"]


def fold(sd, name, eps=1e-5):
    """conv (no bias) + BatchNorm -> conv weight and bias."""
    w = sd[f"{name}.0.weight"].double()
    gamma, beta = sd[f"{name}.1.weight"].double(), sd[f"{name}.1.bias"].double()
    mean, var = sd[f"{name}.1.running_mean"].double(), sd[f"{name}.1.running_var"].double()
    s = gamma / torch.sqrt(var + eps)
    return (w * s[:, None, None, None]).float().numpy(), (beta - mean * s).float().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt"); ap.add_argument("out")
    ap.add_argument("--f16", action="store_true", help="store conv kernels as fp16")
    ap.add_argument("--width", type=int, default=640); ap.add_argument("--height", type=int, default=360)
    a = ap.parse_args()

    sd = torch.load(a.ckpt, map_location="cpu", weights_only=True)
    wt = np.float16 if a.f16 else np.float32
    wr = gguf.GGUFWriter(a.out, "tracknet")
    wr.add_name("RallyTrack ball tracker")
    wr.add_uint32("tracknet.input_width", a.width)
    wr.add_uint32("tracknet.input_height", a.height)
    wr.add_uint32("tracknet.input_channels", int(sd["e1.0.0.weight"].shape[1]))
    wr.add_uint32("tracknet.base_channels", int(sd["e1.0.0.weight"].shape[0]))

    n = 0
    for name in BLOCKS:
        w, b = fold(sd, name)
        wr.add_tensor(f"{name}.weight", w.astype(wt))
        wr.add_tensor(f"{name}.bias", b.reshape(1, -1, 1, 1))      # ggml shape [1, 1, OC, 1], broadcasts over W, H
        n += w.size + b.size
    wr.add_tensor("head.weight", sd["head.weight"].numpy().astype(wt))
    wr.add_tensor("head.bias", sd["head.bias"].numpy().reshape(1, -1, 1, 1))
    n += sd["head.weight"].numel() + 1

    wr.write_header_to_file(); wr.write_kv_data_to_file(); wr.write_tensors_to_file(); wr.close()
    print(f"{a.out}: {len(BLOCKS) + 1} conv layers, {n / 1e6:.2f} M parameters, kernels {'fp16' if a.f16 else 'fp32'}")


if __name__ == "__main__":
    main()
