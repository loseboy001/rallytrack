# Benchmarks

Ball tracker, one frame, input 3 x (3 x 360 x 640), 2.83 M parameters, about 42 G multiply-adds.

Mac: MacBook with Apple M1, 16 GB, macOS 15.6. Phone: iPhone 15 Pro Max.
Times are means over the runs stated; the first runs are discarded as warm-up.

## Core ML

| Model | Device | CPU only | CPU + GPU | CPU + Neural Engine | All |
|---|---|---|---|---|---|
| fp16 | M1 | 58.5 ms | 36.2 ms | 10.5 ms | 10.4 ms |
| 8-bit (QAT) | M1 | 76.6 ms | 39.6 ms | 9.3 ms | 9.8 ms |
| fp16 | iPhone 15 Pro Max | | | 13.2 ms | |
| 8-bit (QAT) | iPhone 15 Pro Max | | | 5.66 ms | |

* Mac: `app/cli/bench2.swift`, 100 runs per compute unit.
* iPhone: `app/RallyTrack/Analysis/ModelBench.swift` in a debug build, 200 runs.
* 8-bit means weights and activations, from quantization-aware training (`training/train_qat.py`,
  coremltools `LinearQuantizer`), converted with `coreml/convert_qat.py`.
* On the M1 the 8-bit model gains little on the Neural Engine and is slower on the CPU. On the
  iPhone 15 Pro Max it is 2.3x faster. The phone is the target, so the app ships the 8-bit model.

Whole-video analysis, 90 s clip, M1: 78 s before frame skipping, 42 s after. Inference was 93% of
the time before the change.

## ggml

Built from llama.cpp commit `17701e4`, Release. GGUF exported with `ggml/export_gguf.py`
(BatchNorm folded). `./build/tracknet -n 10`, 3 warm-up runs.

| Backend | Convolution | Weights | Time | Graph nodes | Graph memory |
|---|---|---|---|---|---|
| Metal | direct kernel, as shipped (`GGML_METAL_CONV_2D_GENERIC=1`) | fp32 | 2482 ms | 62 | 205 MB |
| Metal | im2col + matrix multiply (`-c im2col`) | fp32 | 746 ms | 170 | 577 MB |
| Metal | **3x3 kernel from the patch** | fp32 | **191 ms** | 62 | 205 MB |
| Metal | **3x3 kernel from the patch** | fp16 | **177 ms** | 62 | 205 MB |
| CPU, 4 threads | direct | fp32 | 710 to 775 ms | 62 | 205 MB |
| CPU, 4 threads | direct | fp16 | 1170 ms | 62 | 205 MB |
| CPU, 4 threads | im2col + matrix multiply | fp32 | 1127 ms | 170 | 577 MB |

What the numbers say:

* The direct Metal kernel as shipped is slower than the CPU for this network. It uses one thread per
  output value.
* The patched kernel is 13x faster than it and 3.9x faster than im2col, at the memory of the direct path.
* fp16 weights are a little faster on the GPU and slower on the CPU.
* Core ML on the Neural Engine, 10.5 ms, is 18x faster than the best ggml number. ggml has no
  Neural Engine backend.

## Correctness

Same inputs through PyTorch (fp32, CPU) and the ggml binary, 10 triplets of real video frames,
`ggml/parity.py`:

| Backend | Weights | Largest logit difference | Peak position difference |
|---|---|---|---|
| CPU | fp32 | 4.9e-05 | 0 px |
| Metal, patched kernel | fp32 | 7.6e-05 | 0 px |
| Metal, patched kernel | fp16 | 5.2e-02 | 0 px |

The fp16 row differs because the weights were rounded to fp16 at export, not because of the kernel.

Kernel against the ggml CPU backend on random data, 16 shape cases, `./build/conv2d_3x3_test`:
largest difference 8.3e-07.

Core ML fp16 against PyTorch, 20 triplets of real frames, `coreml/convert.py --frames`: peak position
difference 0 px.

## llama.cpp

`./build/coach` with Qwen2.5-1.5B-Instruct, Q4_K_M, 0.99 GB, M1:

| | Metal (`-ngl 99`) | CPU (`-ngl 0`) |
|---|---|---|
| Prompt, about 190 tokens | 530 to 600 tokens/s | 189 tokens/s |
| Answer | 45 to 47 tokens/s | 45 tokens/s |
| Load | 0.2 to 0.7 s | 1.2 s |

Generation speed is the same on GPU and CPU, and the GPU is about 3x faster on the prompt, where many
tokens are processed at once. That fits generation being limited by how fast the weights can be read
from memory, which GPU and CPU share on this machine. This was not measured directly.
