// Correctness test for the Metal 3x3 convolution kernel (ggml/patches/metal-conv2d-3x3.patch):
// the same random input and weights go through the GPU backend and the CPU backend, results must match.
//
// The kernel computes a 2x2 output tile for 4 output channels per thread, so the cases cover the edges
// of that scheme: odd sizes, channel counts that are not multiples of 4, batches, padding 0..3,
// fp32 and fp16 weights.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

static std::vector<float> run(ggml_backend_t be, int W, int H, int IC, int OC, int N, int pad, bool f16,
                              const std::vector<float> & xin, const std::vector<float> & win) {
    ggml_init_params ip = { ggml_tensor_overhead() * 16 + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * w = ggml_new_tensor_4d(ctx, f16 ? GGML_TYPE_F16 : GGML_TYPE_F32, 3, 3, IC, OC);
    ggml_tensor * x = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, W, H, IC, N);
    ggml_tensor * y = ggml_conv_2d_direct(ctx, w, x, 1, 1, pad, pad, 1, 1);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, y);
    ggml_gallocr_t ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(be));
    ggml_gallocr_alloc_graph(ga, gf);

    if (f16) {
        std::vector<ggml_fp16_t> h(win.size());
        ggml_fp32_to_fp16_row(win.data(), h.data(), (int64_t) win.size());
        ggml_backend_tensor_set(w, h.data(), 0, h.size() * sizeof(ggml_fp16_t));
    } else {
        ggml_backend_tensor_set(w, win.data(), 0, win.size() * sizeof(float));
    }
    ggml_backend_tensor_set(x, xin.data(), 0, xin.size() * sizeof(float));
    ggml_backend_graph_compute(be, gf);

    std::vector<float> out((size_t) ggml_nelements(y));
    ggml_backend_tensor_get(y, out.data(), 0, out.size() * sizeof(float));
    ggml_gallocr_free(ga);
    ggml_free(ctx);
    return out;
}

int main() {
    ggml_backend_t gpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU, nullptr);
    ggml_backend_t cpu = ggml_backend_cpu_init();
    if (!gpu) { fprintf(stderr, "no GPU backend in this build\n"); return 1; }

    //                 W   H  IC  OC  N  pad
    const int cases[][6] = {
        {  7,  5,  3,  5, 1, 1 },
        {  8,  8,  4,  4, 2, 1 },
        {  1,  1,  1,  1, 1, 1 },
        {  4,  3,  6,  7, 3, 0 },
        { 33, 17,  9, 10, 1, 2 },
        { 16,  9,  5,  3, 2, 1 },
        {  3,  3,  2,  9, 1, 0 },
        {  5,  4,  3,  6, 1, 3 },
    };

    double worst = 0;
    for (const auto & c : cases) {
        for (int f16 = 0; f16 < 2; f16++) {
            const int W = c[0], H = c[1], IC = c[2], OC = c[3], N = c[4], pad = c[5];
            std::vector<float> x((size_t) W * H * IC * N), w((size_t) 9 * IC * OC);
            for (auto & v : x) v = rand() / (float) RAND_MAX - 0.5f;
            for (auto & v : w) v = rand() / (float) RAND_MAX - 0.5f;

            const auto a = run(gpu, W, H, IC, OC, N, pad, f16, x, w);
            const auto b = run(cpu, W, H, IC, OC, N, pad, f16, x, w);
            double d = a.size() == b.size() && !a.empty() ? 0 : 1;
            for (size_t i = 0; i < a.size() && i < b.size(); i++) d = fmax(d, fabs(a[i] - b[i]));
            printf("W%-2d H%-2d IC%-2d OC%-2d N%d pad%d %s: %5zu outputs, max diff %.2e\n",
                   W, H, IC, OC, N, pad, f16 ? "f16" : "f32", a.size(), d);
            worst = fmax(worst, d);
        }
    }
    printf("worst %.2e -> %s\n", worst, worst < 1e-4 ? "PASS" : "FAIL");
    return worst < 1e-4 ? 0 : 1;
}
