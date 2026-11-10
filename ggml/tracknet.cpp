// Ball tracker (TrackNet-style U-Net) on ggml.
//
//   tracknet -m tracknet-f32.gguf [-b metal|cpu] [-c direct|im2col] [-t threads] [-i input.bin] [-o heatmap.bin] [-n iters]
//
// input.bin : float32, shape [9, H, W] (three RGB frames, values 0..1), same layout as the PyTorch input
// output    : float32 logits, shape [H, W]; the peak (x, y, confidence) is printed
//
// The whole graph is built from five ggml ops: CONV_2D, ADD (bias), UNARY relu, POOL_2D, UPSCALE, CONCAT.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "gguf.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <map>
#include <string>
#include <vector>

struct Model {
    ggml_backend_t backend = nullptr;
    ggml_context * ctx_w   = nullptr;          // weights (metadata only, data lives in the backend buffer)
    ggml_backend_buffer_t buf_w = nullptr;
    std::map<std::string, ggml_tensor *> t;
    int W = 640, H = 360, C = 9;
    bool im2col = false;                       // conv as im2col + matmul instead of the direct kernel
};

static double now_ms() {
    using namespace std::chrono;
    return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}

static bool load_model(const char * path, Model & m) {
    ggml_context * ctx_meta = nullptr;
    gguf_init_params gp = { /*no_alloc=*/ true, &ctx_meta };
    gguf_context * g = gguf_init_from_file(path, gp);
    if (!g) { fprintf(stderr, "cannot read %s\n", path); return false; }

    auto u32 = [&](const char * key, int def) {
        int64_t id = gguf_find_key(g, key);
        return id < 0 ? def : (int) gguf_get_val_u32(g, id);
    };
    m.W = u32("tracknet.input_width", 640);
    m.H = u32("tracknet.input_height", 360);
    m.C = u32("tracknet.input_channels", 9);

    const int64_t n = gguf_get_n_tensors(g);
    ggml_init_params ip = { ggml_tensor_overhead() * (size_t) (n + 1), nullptr, /*no_alloc=*/ true };
    m.ctx_w = ggml_init(ip);
    for (int64_t i = 0; i < n; i++) {
        const char * name = gguf_get_tensor_name(g, i);
        ggml_tensor * src = ggml_get_tensor(ctx_meta, name);
        ggml_tensor * dst = ggml_dup_tensor(m.ctx_w, src);
        ggml_set_name(dst, name);
        m.t[name] = dst;
    }
    m.buf_w = ggml_backend_alloc_ctx_tensors(m.ctx_w, m.backend);

    FILE * f = fopen(path, "rb");
    std::vector<char> tmp;
    for (int64_t i = 0; i < n; i++) {
        ggml_tensor * dst = m.t[gguf_get_tensor_name(g, i)];
        const size_t sz = ggml_nbytes(dst);
        tmp.resize(sz);
        fseek(f, (long) (gguf_get_data_offset(g) + gguf_get_tensor_offset(g, i)), SEEK_SET);
        if (fread(tmp.data(), 1, sz, f) != sz) { fprintf(stderr, "short read\n"); return false; }
        ggml_backend_tensor_set(dst, tmp.data(), 0, sz);
    }
    fclose(f);
    gguf_free(g);
    ggml_free(ctx_meta);
    return true;
}

static ggml_tensor * conv(ggml_context * ctx, Model & m, ggml_tensor * w, ggml_tensor * x, int pad) {
    return m.im2col ? ggml_conv_2d(ctx, w, x, 1, 1, pad, pad, 1, 1)
                    : ggml_conv_2d_direct(ctx, w, x, 1, 1, pad, pad, 1, 1);
}

// conv 3x3 (BatchNorm already folded in) + bias + ReLU
static ggml_tensor * cbr(ggml_context * ctx, Model & m, const std::string & name, ggml_tensor * x) {
    ggml_tensor * w = m.t.at(name + ".weight");
    ggml_tensor * b = m.t.at(name + ".bias");
    x = conv(ctx, m, w, x, 1);
    x = ggml_add(ctx, x, b);
    return ggml_relu(ctx, x);
}

static ggml_tensor * stage(ggml_context * ctx, Model & m, const char * prefix, int n, ggml_tensor * x) {
    for (int i = 0; i < n; i++) x = cbr(ctx, m, std::string(prefix) + "." + std::to_string(i), x);
    return x;
}

static ggml_cgraph * build(ggml_context * ctx, Model & m, ggml_tensor ** in, ggml_tensor ** out) {
    ggml_tensor * x = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, m.W, m.H, m.C, 1);
    ggml_set_name(x, "input"); ggml_set_input(x);

    auto pool = [&](ggml_tensor * t) { return ggml_pool_2d(ctx, t, GGML_OP_POOL_MAX, 2, 2, 2, 2, 0, 0); };
    auto up   = [&](ggml_tensor * t) { return ggml_upscale(ctx, t, 2, GGML_SCALE_MODE_NEAREST); };

    ggml_tensor * e1 = stage(ctx, m, "e1", 2, x);
    ggml_tensor * e2 = stage(ctx, m, "e2", 2, pool(e1));
    ggml_tensor * e3 = stage(ctx, m, "e3", 3, pool(e2));
    ggml_tensor * b  = stage(ctx, m, "b",  3, pool(e3));
    ggml_tensor * d3 = stage(ctx, m, "d3", 3, ggml_concat(ctx, up(b),  e3, 2));
    ggml_tensor * d2 = stage(ctx, m, "d2", 2, ggml_concat(ctx, up(d3), e2, 2));
    ggml_tensor * d1 = stage(ctx, m, "d1", 2, ggml_concat(ctx, up(d2), e1, 2));

    ggml_tensor * y = conv(ctx, m, m.t.at("head.weight"), d1, 0);
    y = ggml_add(ctx, y, m.t.at("head.bias"));
    ggml_set_name(y, "logits"); ggml_set_output(y);

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, y);
    *in = x; *out = y;
    return gf;
}

int main(int argc, char ** argv) {
    std::string model, backend = "metal", in_path, out_path, conv_mode = "direct";
    int threads = 4, iters = 0;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&]() { if (i + 1 >= argc) { fprintf(stderr, "missing value for %s\n", a.c_str()); exit(1); } return std::string(argv[++i]); };
        if      (a == "-m") model    = next();
        else if (a == "-b") backend  = next();
        else if (a == "-t") threads  = atoi(next().c_str());
        else if (a == "-i") in_path  = next();
        else if (a == "-o") out_path = next();
        else if (a == "-n") iters    = atoi(next().c_str());
        else if (a == "-c") conv_mode = next();
        else { fprintf(stderr, "unknown option %s\n", a.c_str()); return 1; }
    }
    if (model.empty()) { fprintf(stderr, "usage: tracknet -m model.gguf [-b metal|cpu] [-c direct|im2col] [-t N] [-i in.bin] [-o out.bin] [-n iters]\n"); return 1; }

    Model m;
    m.im2col = conv_mode == "im2col";
    if (backend == "cpu") {
        m.backend = ggml_backend_cpu_init();
        ggml_backend_cpu_set_n_threads(m.backend, threads);
    } else {
        m.backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU, nullptr);
        if (!m.backend) { fprintf(stderr, "no GPU backend in this build\n"); return 1; }
    }
    if (!load_model(model.c_str(), m)) return 1;

    ggml_init_params ip = { ggml_tensor_overhead() * 256 + ggml_graph_overhead(), nullptr, /*no_alloc=*/ true };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * x = nullptr, * y = nullptr;
    ggml_cgraph * gf = build(ctx, m, &x, &y);

    // every node must run on the chosen backend, no silent CPU fallback
    int unsupported = 0;
    for (int i = 0; i < ggml_graph_n_nodes(gf); i++) {
        ggml_tensor * node = ggml_graph_node(gf, i);
        if (!ggml_backend_supports_op(m.backend, node)) {
            fprintf(stderr, "not supported on %s: %s (%s)\n", ggml_backend_name(m.backend), ggml_op_name(node->op), node->name);
            unsupported++;
        }
    }
    if (unsupported) return 1;

    ggml_gallocr_t ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(m.backend));
    if (!ggml_gallocr_alloc_graph(ga, gf)) { fprintf(stderr, "graph allocation failed\n"); return 1; }

    const size_t n_in = (size_t) m.W * m.H * m.C, n_out = (size_t) m.W * m.H;
    std::vector<float> in(n_in), out(n_out);
    if (!in_path.empty()) {
        FILE * f = fopen(in_path.c_str(), "rb");
        if (!f || fread(in.data(), sizeof(float), n_in, f) != n_in) { fprintf(stderr, "cannot read %zu floats from %s\n", n_in, in_path.c_str()); return 1; }
        fclose(f);
    } else {
        for (size_t i = 0; i < n_in; i++) in[i] = (float) (i % 255) / 255.0f;
    }

    auto run = [&]() {
        ggml_backend_tensor_set(x, in.data(), 0, n_in * sizeof(float));
        if (ggml_backend_graph_compute(m.backend, gf) != GGML_STATUS_SUCCESS) { fprintf(stderr, "compute failed\n"); exit(1); }
        ggml_backend_tensor_get(y, out.data(), 0, n_out * sizeof(float));   // also waits for the GPU
    };

    run();
    size_t best = std::max_element(out.begin(), out.end()) - out.begin();
    printf("backend %s | graph %d nodes | memory %.1f MB | peak x=%zu y=%zu conf=%.4f\n",
           ggml_backend_name(m.backend), ggml_graph_n_nodes(gf), ggml_gallocr_get_buffer_size(ga, 0) / 1048576.0,
           best % m.W, best / m.W, 1.0 / (1.0 + std::exp(-out[best])));

    if (!out_path.empty()) {
        FILE * f = fopen(out_path.c_str(), "wb");
        fwrite(out.data(), sizeof(float), n_out, f);
        fclose(f);
    }

    if (iters > 0) {
        for (int i = 0; i < 3; i++) run();                                   // warm-up
        std::vector<double> ms(iters);
        for (int i = 0; i < iters; i++) { double t = now_ms(); run(); ms[i] = now_ms() - t; }
        std::sort(ms.begin(), ms.end());
        double sum = 0; for (double v : ms) sum += v;
        printf("%d runs: mean %.2f ms | median %.2f ms | min %.2f ms\n", iters, sum / iters, ms[iters / 2], ms[0]);
    }

    ggml_gallocr_free(ga);
    ggml_free(ctx);
    ggml_backend_buffer_free(m.buf_w);
    ggml_free(m.ctx_w);
    ggml_backend_free(m.backend);
    return 0;
}
