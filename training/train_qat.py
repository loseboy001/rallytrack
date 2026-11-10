"""球模型量化感知训练（QAT）：在 v5 的基础上插入 8 位假量化层微调，让模型适应 8 位计算。
torchrun --nproc_per_node=G train_qat.py --cache DIR --out DIR --init v5.pt --train ... --val2 ...（参数同 train_ball.py）
输出：qat_state.pt（训练中的权重）、traced.pt（finalize 后的 TorchScript，拿到 Mac 上转 CoreML）
"""
import argparse, json, math, os, time
import torch, torch.nn as nn, torch.nn.functional as F
import torch.distributed as dist
import train_ball as T
from coremltools.optimize.torch.quantization import LinearQuantizer, LinearQuantizerConfig


class Peak3(nn.Module):  # 和 App 里一样的输出：三帧分开输入，输出 (x, y, conf)
    def __init__(s, net):
        super().__init__(); s.net = net
    def forward(s, f0, f1, f2):
        p = torch.sigmoid(s.net(torch.cat([f0, f1, f2], 1)))[:, 0]
        rowmax, colidx = p.max(2); conf, ys = rowmax.max(1)
        xs = colidx.gather(1, ys[:, None])[:, 0]
        return torch.stack([xs.float(), ys.float(), conf], 1)


def main():
    ap = argparse.ArgumentParser()
    for k in ("cache", "out", "init", "train", "val", "val2"): ap.add_argument("--" + k, default="")
    ap.add_argument("--epochs", type=int, default=6); ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4); ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--web_repeat", type=int, default=2)
    a = ap.parse_args()
    ddp = "LOCAL_RANK" in os.environ
    if ddp: dist.init_process_group("nccl"); lr_ = int(os.environ["LOCAL_RANK"]); torch.cuda.set_device(lr_)
    rank, ws = (dist.get_rank(), dist.get_world_size()) if ddp else (0, 1)
    dev = torch.device("cuda")
    tr = T.load_split(a.cache, a.train.split(",")); tr = tr + [it for it in tr if "/web" in it[0]] * (a.web_repeat - 1)
    va2 = T.load_split(a.cache, a.val2.split(",")) if a.val2 else []
    va = T.load_split(a.cache, a.val.split(",")) if a.val else []
    if rank == 0: os.makedirs(a.out, exist_ok=True); print(f"train {len(tr)} val {len(va)} web val {len(va2)} world {ws}", flush=True)
    dtr = T.DS(tr, True); s_tr = torch.utils.data.DistributedSampler(dtr) if ddp else None
    ltr = torch.utils.data.DataLoader(dtr, a.bs, shuffle=s_tr is None, sampler=s_tr, num_workers=a.workers, pin_memory=True, drop_last=True, persistent_workers=True)
    def vloader(items):
        d = T.DS(items, False)
        return torch.utils.data.DataLoader(d, a.bs, sampler=torch.utils.data.DistributedSampler(d, shuffle=False) if ddp else None, num_workers=a.workers, pin_memory=True)
    lva, lva2 = (vloader(va) if va else None), (vloader(va2) if va2 else None)

    net = T.TrackNet(); net.load_state_dict(torch.load(a.init, map_location="cpu"))
    steps = a.epochs * len(ltr)
    # 里程碑：一开始就统计数值范围；300 步后开始模拟 8 位；50% 时冻结量化范围；40% 时冻结 BN 统计
    cfg = LinearQuantizerConfig.from_dict({"global_config": {"quantization_scheme": "symmetric",
                                                             "milestones": [0, 300, int(steps * 0.5), int(steps * 0.4)]}})
    q = LinearQuantizer(net, cfg)
    ex = torch.rand(1, 9, T.H, T.W)
    model = q.prepare(example_inputs=(ex,), inplace=False).to(dev)
    if ddp: model = nn.parallel.DistributedDataParallel(model, device_ids=[lr_])
    opt = torch.optim.AdamW(model.parameters(), a.lr * math.sqrt(ws), weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    step, t0, best = 0, time.time(), -1
    for ep in range(a.epochs):
        if s_tr: s_tr.set_epoch(ep)
        model.train()
        for x, hm, _ in ltr:
            x, hm = x.to(dev, non_blocking=True), hm.to(dev, non_blocking=True)
            loss = T.wbce(model(x).float(), hm)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step(); q.step(); step += 1
            if rank == 0 and step % 100 == 0:
                print(f"ep {ep} step {step}/{steps} loss {loss.item():.5f} {(time.time() - t0) / step:.3f}s/step", flush=True)
        r = {}
        if lva is not None: r.update(T.evaluate(model, lva, dev))
        if lva2 is not None: r.update({"web_" + k: v for k, v in T.evaluate(model, lva2, dev).items()})
        if rank == 0:
            r.update(epoch=ep, step=step); print("EVAL", json.dumps(r), flush=True)
            open(os.path.join(a.out, "eval.jsonl"), "a").write(json.dumps(r) + "\n")
            m = model.module if ddp else model
            torch.save(m.state_dict(), os.path.join(a.out, "qat_state.pt"))
            score = r.get("web_recall@8px", 0) + r.get("recall@4px", 0)
            if score > best and step >= steps * 0.5:   # 量化范围冻结之后的才算数
                best = score
                fin = q.finalize((model.module if ddp else model), inplace=False).cpu().eval()
                torch.jit.trace(Peak3(fin).eval(), [torch.rand(1, 3, T.H, T.W) for _ in range(3)]).save(os.path.join(a.out, "best_traced.pt"))
                print("saved best_traced.pt epoch", ep, flush=True)
    if rank == 0:
        m = (model.module if ddp else model).cpu().eval()
        fin = q.finalize(m, inplace=False)
        traced = torch.jit.trace(Peak3(fin).eval(), [torch.rand(1, 3, T.H, T.W) for _ in range(3)])
        traced.save(os.path.join(a.out, "traced.pt")); print("saved traced.pt", flush=True)
    if ddp: dist.destroy_process_group()


if __name__ == "__main__":
    main()
