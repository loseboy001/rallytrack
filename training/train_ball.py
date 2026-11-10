"""Ball heatmap detector (TrackNetV2-style U-Net), causal: frames t-2,t-1,t -> heatmap of ball at t.
DDP via torchrun. Data = a frame cache: <cache>/<clip>/frames/<idx>.jpg (960x540) and <cache>/<clip>/labels.json
(ball position per frame, optional table corners). No data or weights are part of this repository.
torchrun --nproc_per_node=G train_ball.py --cache DIR --out DIR --train clip_a,clip_b --val clip_c
"""
import argparse, json, math, os, random, time
import torch, torch.nn as nn, torch.nn.functional as F
import torch.distributed as dist
from torchvision.io import decode_jpeg, read_file

from tracknet import W, H, WIDTH, TrackNet  # 模型定义和分辨率（缓存是 960x540，训练/推理用 W x H）

STRIDES = os.environ.get("PP_STRIDES", "1") == "1"  # 是否用抽帧模拟低帧率
ROI = os.environ.get("PP_ROI", "1") == "1"  # 是否按球台裁剪（远机位时球不会太小）


def table_roi(quad, fw, fh, jitter=False):
    """球台周围的裁剪框 (x0, y0, w, h)，宽 = 2 倍台宽、16:9。App 里同样的规则（VideoAnalyzer.tableROI），改这里要一起改。"""
    xs = [p[0] for p in quad]; ys = [p[1] for p in quad]
    tw = max(xs) - min(xs); cx = (max(xs) + min(xs)) / 2
    rw = 2.0 * tw * (random.uniform(0.85, 1.25) if jitter else 1.0); rh = rw * H / W
    cy = (min(ys) + max(ys)) / 2 - 0.2 * rh
    if jitter: cx += random.uniform(-0.1, 0.1) * rw; cy += random.uniform(-0.1, 0.1) * rh
    if rw > fw: rw = fw; rh = rw * H / W
    if rh > fh: rh = fh; rw = rh * W / H
    x0 = min(max(0, cx - rw / 2), fw - rw); y0 = min(max(0, cy - rh / 2), fh - rh)
    return x0, y0, rw, rh


def load_split(cache, names):
    items = []
    for n in names:
        lab = json.load(open(os.path.join(cache, n, "labels.json")))
        have = set(lab["frames"]); fw, fh = lab["wh"]
        quad = lab.get("quad")  # 缓存坐标系（960x540）里的球台四角；没有就用整帧
        # 帧间隔 s：120fps 素材用 s=2/4 模拟 60/30fps（手机和网上视频大多 30fps），网络视频本身就是 30fps 只用 s=1
        cand = (1, 2, 4) if lab.get("fps", 30) >= 100 and STRIDES else (1,)
        for k, v in lab["ball"].items():
            k = int(k)
            ss = tuple(s for s in cand if k - 2 * s in have and k - s in have and k in have)
            if ss:
                items.append((os.path.join(cache, n, "frames"), k, None if v is None else (v[0], v[1]), ss, quad, (fw, fh)))
    return items


class DS(torch.utils.data.Dataset):
    def __init__(self, items, aug):
        self.items, self.aug = items, aug

    def __len__(self): return len(self.items)

    def __getitem__(self, i):
        d, k, xy, ss, quad, (fw, fh) = self.items[i]
        s = random.choice(ss) if self.aug else ss[-1]
        fr = [decode_jpeg(read_file(os.path.join(d, f"{k + o * s}.jpg"))) for o in (-2, -1, 0)]
        x = torch.cat(fr, 0).float().div_(255)  # 9xHxW
        x0, y0, rw, rh = table_roi(quad, fw, fh, self.aug) if (quad and ROI) else (0, 0, fw, fh)
        xi, yi, wi, hi = int(round(x0)), int(round(y0)), max(8, int(round(rw))), max(8, int(round(rh)))
        x = F.interpolate(x[None, :, yi:yi + hi, xi:xi + wi], size=(H, W), mode="bilinear", align_corners=False, antialias=True)[0]
        xy = None if xy is None else [(xy[0] - xi) * W / wi, (xy[1] - yi) * H / hi]
        if xy and not (0 <= xy[0] < W and 0 <= xy[1] < H): xy = None
        if self.aug:
            if random.random() < 0.5:
                x = x.flip(-1)
                if xy: xy[0] = W - 1 - xy[0]
            if random.random() < 0.35:  # 缩小 + 补边：模拟机位更远、球台更小的画面
                z = random.uniform(0.55, 0.95); nh, nw = int(H * z), int(W * z); y0 = random.randint(0, H - nh); x0 = random.randint(0, W - nw)
                canvas = x.new_zeros(x.shape) + x.mean((1, 2), keepdim=True)
                canvas[:, y0:y0 + nh, x0:x0 + nw] = F.interpolate(x[None], size=(nh, nw), mode="bilinear", align_corners=False, antialias=True)[0]
                x = canvas
                if xy: xy = [xy[0] * z + x0, xy[1] * z + y0]
            s = random.uniform(1.0, 1.3) if random.random() < 0.5 else 1.0  # zoom-in crop keeps ball scale variety
            if s > 1.01:
                ch, cw = int(H / s), int(W / s); y0 = random.randint(0, H - ch); x0 = random.randint(0, W - cw)
                x = F.interpolate(x[None, :, y0:y0 + ch, x0:x0 + cw], size=(H, W), mode="bilinear", align_corners=False)[0]
                if xy:
                    xy = [(xy[0] - x0) * W / cw, (xy[1] - y0) * H / ch]
                    if not (0 <= xy[0] < W and 0 <= xy[1] < H): xy = None
            x = x.view(3, 3, H, W)
            gain = torch.empty(1, 3, 1, 1).uniform_(0.8, 1.2)  # 分通道颜色扰动：不同场馆灯光色温不同
            x = (x * random.uniform(0.7, 1.3) * gain + random.uniform(-0.1, 0.1)).clamp_(0, 1).view(9, H, W)
        hm = torch.zeros(1, H, W)
        if xy:
            yy, xx = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
            hm[0] = torch.exp(-((xx - xy[0]) ** 2 + (yy - xy[1]) ** 2) / (2 * 2.5 ** 2))
        return x, hm, torch.tensor(xy if xy else [-1.0, -1.0])


def wbce(logit, hm):  # TrackNetV2 weighted BCE on soft heatmap
    p = torch.sigmoid(logit).clamp(1e-6, 1 - 1e-6)
    return -((1 - p) ** 2 * hm * torch.log(p) + p ** 2 * (1 - hm) * torch.log(1 - p)).mean()


def peaks(logit, thr=0.5):
    p = torch.sigmoid(logit)[:, 0]; B = p.shape[0]
    v, idx = p.view(B, -1).max(1)
    xy = torch.stack([idx % W, idx // W], 1).float()
    xy[v < thr] = -1
    return xy


@torch.no_grad()
def evaluate(model, dl, dev, tol=(4, 8)):
    model.eval(); st = torch.zeros(4 + len(tol), device=dev)  # tp_any, fp, fn, tn, tp@tol...
    errs = []
    for x, _, gt in dl:
        x, gt = x.to(dev, non_blocking=True), gt.to(dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pr = peaks(model(x).float())
        has_g, has_p = gt[:, 0] >= 0, pr[:, 0] >= 0
        d = (pr - gt).norm(dim=1)
        st[0] += (has_g & has_p).sum(); st[1] += (~has_g & has_p).sum(); st[2] += (has_g & ~has_p).sum(); st[3] += (~has_g & ~has_p).sum()
        for j, t in enumerate(tol): st[4 + j] += (has_g & has_p & (d <= t)).sum()
        errs.append(d[has_g & has_p])
    if dist.is_initialized(): dist.all_reduce(st)
    e = torch.cat(errs) if errs else torch.zeros(0)
    model.train()
    ng = st[0] + st[2]
    r = {"n_gt": int(ng), "recall@%dpx" % tol[0]: float(st[4] / ng.clamp(min=1)), "recall@%dpx" % tol[1]: float(st[5] / ng.clamp(min=1)),
         "miss": float(st[2] / ng.clamp(min=1)), "fp_on_empty": int(st[1]), "median_err_px_local": float(e.median()) if len(e) else -1}
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--train", required=True, help="comma-separated clip names")
    ap.add_argument("--val", required=True, help="comma-separated clip names")
    ap.add_argument("--epochs", type=int, default=30); ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3); ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--max_items", type=int, default=0, help="smoke test: cap dataset size")
    ap.add_argument("--val2", default="", help="第二组验证集（网络视频），单独报指标")
    ap.add_argument("--init", default="", help="从已有权重继续训练")
    ap.add_argument("--web_repeat", type=int, default=1, help="名字以 web_ 开头的训练集重复几遍（数据少，提高比重）")
    a = ap.parse_args()
    ddp = "LOCAL_RANK" in os.environ
    if ddp:
        dist.init_process_group("nccl"); lr_ = int(os.environ["LOCAL_RANK"]); torch.cuda.set_device(lr_)
    rank, ws = (dist.get_rank(), dist.get_world_size()) if ddp else (0, 1)
    dev = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    tr, va = load_split(a.cache, a.train.split(",")), load_split(a.cache, a.val.split(","))
    tr = tr + [it for it in tr if "/web" in it[0]] * (a.web_repeat - 1)
    va2 = load_split(a.cache, a.val2.split(",")) if a.val2 else []
    if a.max_items: random.Random(0).shuffle(tr); tr, va = tr[:a.max_items], va[:a.max_items // 4 + 1]
    if rank == 0:
        os.makedirs(a.out, exist_ok=True); json.dump({**vars(a), "W": W, "H": H, "WIDTH": WIDTH}, open(os.path.join(a.out, "args.json"), "w"), indent=1)
        print(f"train {len(tr)} val {len(va)} world {ws}", flush=True)
    dtr, dva = DS(tr, True), DS(va, False)
    str_ = torch.utils.data.DistributedSampler(dtr) if ddp else None
    sva = torch.utils.data.DistributedSampler(dva, shuffle=False) if ddp else None
    ltr = torch.utils.data.DataLoader(dtr, a.bs, shuffle=str_ is None, sampler=str_, num_workers=a.workers, pin_memory=True, drop_last=True, persistent_workers=True)
    lva = torch.utils.data.DataLoader(dva, a.bs, sampler=sva, num_workers=a.workers, pin_memory=True)
    dva2 = DS(va2, False) if va2 else None
    lva2 = torch.utils.data.DataLoader(dva2, a.bs, sampler=torch.utils.data.DistributedSampler(dva2, shuffle=False) if ddp else None, num_workers=a.workers, pin_memory=True) if va2 else None
    model = TrackNet()
    if a.init: model.load_state_dict(torch.load(a.init, map_location="cpu"))
    model = model.to(dev).to(memory_format=torch.channels_last)
    if ddp: model = nn.SyncBatchNorm.convert_sync_batchnorm(model); model = nn.parallel.DistributedDataParallel(model, device_ids=[lr_])
    opt = torch.optim.AdamW(model.parameters(), a.lr * math.sqrt(ws), weight_decay=1e-4)
    total = a.epochs * len(ltr)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr * math.sqrt(ws), total_steps=total, pct_start=0.05)
    best, step, t0 = -1, 0, time.time()
    for ep in range(a.epochs):
        if str_: str_.set_epoch(ep)
        for x, hm, _ in ltr:
            x, hm = x.to(dev, non_blocking=True).to(memory_format=torch.channels_last), hm.to(dev, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = wbce(model(x).float(), hm)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step(); step += 1
            if rank == 0 and step % 50 == 0:
                print(f"ep {ep} step {step}/{total} loss {loss.item():.5f} lr {sched.get_last_lr()[0]:.2e} {(time.time() - t0) / step:.3f}s/step", flush=True)
        r = evaluate(model, lva, dev)
        if lva2 is not None:
            r2 = evaluate(model, lva2, dev); r.update({"web_" + k: v for k, v in r2.items()})
        if rank == 0:
            r.update(epoch=ep, step=step); print("EVAL", json.dumps(r), flush=True)
            open(os.path.join(a.out, "eval.jsonl"), "a").write(json.dumps(r) + "\n")
            sd = (model.module if ddp else model).state_dict()
            torch.save(sd, os.path.join(a.out, "last.pt"))
            score = (r["recall@4px"] + r.get("web_recall@8px", r["recall@4px"])) / 2  # 原数据和网络视频各占一半
            if score > best: best = score; torch.save(sd, os.path.join(a.out, "best.pt"))
    if ddp: dist.destroy_process_group()


if __name__ == "__main__":
    main()
