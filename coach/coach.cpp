// Session summary written by a local LLM (llama.cpp), fully on device.
//
//   coach -m qwen2.5-1.5b-Q4_K_M.gguf -i session.json [-l en|zh] [-ngl 99]
//
// session.json is what the analysis CLI writes (app/cli/main.swift): per rally the net crossings,
// landing positions on the table in metres, ball speeds and out flags.
//
// The numbers and the choice of what to practise next are computed here, deterministically. The model
// only turns them into two sentences, and is told to use nothing but what it is given.

#include "llama.h"
#include "nlohmann/json.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

using json = nlohmann::json;

static const double TABLE_LENGTH = 2.74;   // metres

struct Stats {
    int rallies = 0, shots = 0, longest = 0, outs = 0, landings = 0;
    int deep = 0, mid = 0, shortb = 0;      // landing depth: < 0.5 m from the end line / in between / < 0.5 m from the net
    double top_speed = 0, median_speed = 0;
};

static Stats summarize(const json & session) {
    Stats s;
    std::vector<double> speeds;
    for (const auto & r : session.at("rallies")) {
        const int n = (int) r.at("crossings").size();
        if (n == 0) continue;
        s.rallies++;
        s.shots += n;
        s.longest = std::max(s.longest, n);
        const auto & tx = r.at("tx"); const auto & out = r.at("out"); const auto & sp = r.at("speeds");
        for (size_t i = 0; i < tx.size(); i++) {
            if (out[i].get<bool>()) { s.outs++; continue; }
            const double x = tx[i].get<double>();
            const double to_end = std::min(x, TABLE_LENGTH - x), to_net = std::fabs(x - TABLE_LENGTH / 2);
            s.landings++;
            if (to_end < 0.5) s.deep++; else if (to_net < 0.5) s.shortb++; else s.mid++;
        }
        for (const auto & v : sp) if (v.get<double>() > 0) speeds.push_back(v.get<double>());
    }
    if (!speeds.empty()) {
        std::sort(speeds.begin(), speeds.end());
        s.top_speed = speeds.back();
        s.median_speed = speeds[speeds.size() / 2];
    }
    return s;
}

// What to work on next is picked by fixed rules, not by the model.
static const char * focus(const Stats & s, bool zh) {
    const int balls = s.landings + s.outs;
    if (balls && s.outs * 5 > balls)
        return zh ? "出界偏多，先求稳，把球打在台内" : "too many balls land out: aim for control and keep the ball on the table";
    if (s.rallies && s.shots < 4 * s.rallies)
        return zh ? "回合偏短，练多拍相持" : "rallies are short: practise keeping the ball in play for more shots";
    if (s.landings && s.deep * 10 < s.landings * 3)
        return zh ? "落点偏浅，多打靠近底线的长球" : "few balls land deep: practise long balls close to the end line";
    if (s.landings && s.shortb * 100 < s.landings * 15)
        return zh ? "落点几乎都是长球，加练近网短球，长短结合" : "almost every ball lands long: add short balls near the net to vary the length";
    return zh ? "落点和相持都比较均衡，可以提高球速" : "length and consistency are balanced: work on ball speed";
}

static std::string facts(const Stats & s, bool zh) {
    char buf[1024];
    auto pct = [&](int n) { return s.landings ? (int) std::lround(100.0 * n / s.landings) : 0; };
    snprintf(buf, sizeof(buf), zh ?
        "回合数：%d\n"
        "总击球数：%d\n"
        "平均每回合击球数：%.1f\n"
        "最长回合：%d 拍\n"
        "最快球速：%.0f 公里/小时\n"
        "球速中位数：%.0f 公里/小时\n"
        "出界球数：%d\n"
        "落点靠近底线（长球）：%d%%\n"
        "落点在中间：%d%%\n"
        "落点靠近球网（短球）：%d%%\n"
        "下次练习重点：%s\n" :
        "rallies: %d\n"
        "total shots: %d\n"
        "average shots per rally: %.1f\n"
        "longest rally: %d shots\n"
        "fastest ball: %.0f km/h\n"
        "median ball speed: %.0f km/h\n"
        "balls landing out: %d\n"
        "balls landing deep, near the end line: %d%%\n"
        "balls landing in the middle: %d%%\n"
        "balls landing short, near the net: %d%%\n"
        "focus for next time: %s\n",
        s.rallies, s.shots, s.rallies ? (double) s.shots / s.rallies : 0.0, s.longest,
        s.top_speed, s.median_speed, s.outs, pct(s.deep), pct(s.mid), pct(s.shortb), focus(s, zh));
    return buf;
}

int main(int argc, char ** argv) {
    std::string model_path, in_path, lang = "en";
    int ngl = 99, max_tokens = 200;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&]() { if (i + 1 >= argc) { fprintf(stderr, "missing value for %s\n", a.c_str()); exit(1); } return std::string(argv[++i]); };
        if      (a == "-m")   model_path = next();
        else if (a == "-i")   in_path    = next();
        else if (a == "-l")   lang       = next();
        else if (a == "-ngl") ngl        = std::stoi(next());
        else if (a == "-n")   max_tokens = std::stoi(next());
        else { fprintf(stderr, "unknown option %s\n", a.c_str()); return 1; }
    }
    if (model_path.empty() || in_path.empty()) {
        fprintf(stderr, "usage: coach -m model.gguf -i session.json [-l en|zh] [-ngl N] [-n max_tokens]\n");
        return 1;
    }

    std::ifstream f(in_path);
    if (!f) { fprintf(stderr, "cannot open %s\n", in_path.c_str()); return 1; }
    const Stats st = summarize(json::parse(f));
    const bool zh = lang == "zh";
    const std::string fact_text = facts(st, zh);
    printf("--- facts ---\n%s", fact_text.c_str());

    llama_log_set([](enum ggml_log_level level, const char * text, void *) {
        if (level == GGML_LOG_LEVEL_ERROR) fprintf(stderr, "%s", text);
    }, nullptr);
    llama_backend_init();

    const auto t_load = std::chrono::steady_clock::now();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = ngl;
    llama_model * model = llama_model_load_from_file(model_path.c_str(), mp);
    if (!model) { fprintf(stderr, "cannot load %s\n", model_path.c_str()); return 1; }
    const llama_vocab * vocab = llama_model_get_vocab(model);

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 1024;
    cp.n_batch = 1024;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) { fprintf(stderr, "cannot create context\n"); return 1; }

    // greedy decoding: the same session always gives the same text
    llama_sampler * smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(smpl, llama_sampler_init_greedy());

    // The instruction goes after the numbers and in the language of the answer: a small model follows that best.
    const std::string system = zh
        ? "你是乒乓球教练。下面是从一段练习视频里测出来的数据。只能使用给出的数字，不要编造数字或事实。"
        : "You are a table tennis coach. Below are statistics measured from a video of one practice session. "
          "Use only the numbers you are given. Do not invent numbers or facts.";
    const std::string user = fact_text + (zh
        ? "\n用中文写两句话。第一句概括这次练习，引用两三个数字。第二句把“下次练习重点”说成给球员的建议。"
        : "\nWrite two sentences. The first sums up the session and quotes two or three of the numbers. "
          "The second turns the focus for next time into advice for the player.");
    std::vector<llama_chat_message> msgs = { { "system", system.c_str() }, { "user", user.c_str() } };

    const char * tmpl = llama_model_chat_template(model, nullptr);
    std::vector<char> prompt(4096);
    int len = llama_chat_apply_template(tmpl, msgs.data(), msgs.size(), true, prompt.data(), prompt.size());
    if (len < 0 || len > (int) prompt.size()) { fprintf(stderr, "chat template failed\n"); return 1; }

    std::vector<llama_token> tokens(len + 16);
    const int n_prompt = llama_tokenize(vocab, prompt.data(), len, tokens.data(), tokens.size(), true, true);
    if (n_prompt < 0) { fprintf(stderr, "tokenize failed\n"); return 1; }
    const double load_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_load).count();

    llama_batch_ext * batch = llama_batch_ext_init(ctx);
    auto feed = [&](const llama_token * tok, int n, llama_pos pos0) {
        llama_batch_ext_clear(batch);
        for (int i = 0; i < n; i++) {
            const int32_t idx = llama_batch_ext_add_token(batch, 0, tok[i]);
            const llama_pos pos = pos0 + i;
            llama_batch_ext_set_pos(batch, idx, &pos);
        }
        llama_batch_ext_set_output_logits(batch, n - 1, true);
        if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch) != 0) { fprintf(stderr, "decode failed\n"); exit(1); }
        llama_synchronize(ctx);   // the GPU backend returns before the work is done; wait, so the timings below are real
    };

    printf("--- summary ---\n");
    const auto t0 = std::chrono::steady_clock::now();
    feed(tokens.data(), n_prompt, 0);
    const auto t1 = std::chrono::steady_clock::now();

    int n_gen = 0;
    llama_pos pos = n_prompt;
    while (n_gen < max_tokens) {
        llama_token id = llama_sampler_sample(smpl, ctx, -1);
        if (llama_vocab_is_eog(vocab, id)) break;
        char buf[256];
        const int n = llama_token_to_piece(vocab, id, buf, sizeof(buf), 0, true);
        if (n > 0) { fwrite(buf, 1, n, stdout); fflush(stdout); }
        feed(&id, 1, pos++);
        n_gen++;
    }
    const auto t2 = std::chrono::steady_clock::now();

    const double pp = std::chrono::duration<double>(t1 - t0).count(), tg = std::chrono::duration<double>(t2 - t1).count();
    printf("\n--- timing ---\nload %.0f ms | prompt %d tokens, %.0f tokens/s | generated %d tokens, %.0f tokens/s\n",
           load_ms, n_prompt, n_prompt / pp, n_gen, n_gen / tg);

    llama_batch_ext_free(batch);
    llama_sampler_free(smpl);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
