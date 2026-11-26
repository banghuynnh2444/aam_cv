"""v2 retraining of DAN (RAF-DB) and Multi-task EfficientNet-B0 (FairFace) with a proper protocol.

What changes vs v1 (scripts/train_full_40ep_dan_35ep_effnet.py + refine_effnet_fairface_gender.py):
  Protocol
  - A stratified 10% VALIDATION split is carved out of each TRAIN set. Checkpoints are selected on
    validation only. The test set is evaluated ONCE, with the selected checkpoint.
    (v1 selected the best epoch on the test set, which makes its test numbers optimistic.)
  - FairFace test set is the exact same 2,191 images as v1 (same seed-2026 permutation), so the
    numbers are directly comparable. RAF-DB test = official 3,068 images, as in v1.
  - Age classes renamed to the real FairFace bins (0-9, 10-19, 20-29, 30-39, 40-49, 50+).
  Training
  - Exponential Moving Average (EMA, decay 0.999) of the weights, used for validation/test.
  - DAN: Mixup / CutMix (p=0.5) on top of the v1 GPU augmentation, +-10 deg random rotation.
  - EfficientNet: distance-blur + multi-margin crop augmentation from the start (v1 only used it in
    a 12-epoch refinement), balanced age/gender loss weights.
"""
import os
import sys
import csv
import json
import time
import copy
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, r"D:\AAM_cv")
sys.path.insert(0, r"D:\AAM_cv\scripts")
from src.emotion.emotion_dan import EMOTION_7_CLASSES, DANEmotionNet
from src.attributes.age_gender_effnet import (
    AGE_GROUP_CLASSES, GENDER_LABEL_CLASSES, MultiTaskEfficientNetB0, compute_classification_metrics,
)
from train_full_40ep_dan_35ep_effnet import _load_one_image_224, map_fairface_age, gpu_augment_batch
from refine_effnet_fairface_gender import augment_gender_robust_batch
import glob

torch.backends.cudnn.benchmark = True
DEV = torch.device("cuda")
MEAN = torch.tensor([0.485, 0.456, 0.406], device=DEV).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225], device=DEV).view(1, 3, 1, 1)
OUT = r"D:\AAM_cv\experiments\v2_dl_results.json"


def stratified_val(y: np.ndarray, frac: float, seed: int):
    rng = np.random.default_rng(seed)
    val = []
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        val.extend(idx[: max(1, int(round(frac * len(idx))))])
    val = np.array(sorted(val))
    tr = np.setdiff1d(np.arange(len(y)), val)
    return tr, val


class EMA:
    def __init__(self, model, decay=0.999):
        self.m = copy.deepcopy(model).eval()
        self.decay = decay
        for p in self.m.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        for e, p in zip(self.m.state_dict().values(), model.state_dict().values()):
            if e.dtype.is_floating_point:
                e.mul_(self.decay).add_(p.detach(), alpha=1 - self.decay)
            else:
                e.copy_(p)


def rotate_batch(x, max_deg, rng):
    B = x.shape[0]
    th = torch.from_numpy(rng.uniform(-max_deg, max_deg, B).astype(np.float32) * np.pi / 180).to(x.device)
    cos, sin = torch.cos(th), torch.sin(th)
    A = torch.zeros(B, 2, 3, device=x.device)
    A[:, 0, 0], A[:, 0, 1], A[:, 1, 0], A[:, 1, 1] = cos, -sin, sin, cos
    grid = F.affine_grid(A, x.shape, align_corners=False)
    return F.grid_sample(x, grid, mode="bilinear", padding_mode="border", align_corners=False)


def lr_mult(ep, epochs, warm=3):
    if ep <= warm:
        return ep / warm
    p = (ep - warm) / max(1, epochs - warm)
    return 0.01 + 0.99 * 0.5 * (1 + np.cos(np.pi * p))


def to_tensor(Xb):
    return torch.from_numpy(Xb).to(DEV, non_blocking=True).permute(0, 3, 1, 2).float() / 255.0


@torch.no_grad()
def predict_probs(model, X, n_out):
    """Returns list of averaged (orig + flip) softmax probs for each model output."""
    model.eval()
    outs = [[] for _ in range(n_out)]
    for s in range(0, len(X), 64):
        xb = to_tensor(X[s:s + 64])
        with torch.amp.autocast("cuda", dtype=torch.float16):
            o1 = model((xb - MEAN) / STD)
            o2 = model((torch.flip(xb, [3]) - MEAN) / STD)
        if n_out == 1:
            o1, o2 = (o1[0],), (o2[0],)
        for k in range(n_out):
            outs[k].append((0.5 * (F.softmax(o1[k].float(), 1) + F.softmax(o2[k].float(), 1))).cpu().numpy())
    return [np.concatenate(o) for o in outs]


# =============================================================================
def train_dan(epochs=40):
    print("\n=== v2 DAN on RAF-DB (val-selected, EMA, Mixup/CutMix) ===")
    f2i = {"neutral": 0, "happy": 1, "sad": 2, "surprise": 3, "fear": 4, "disgust": 5, "angry": 6}
    base = r"D:\AAM_cv\datasets\raw\RAFDB\Organized"

    def load(split):
        items = [(p, c) for f, c in f2i.items() for p in sorted(glob.glob(os.path.join(base, split, f, "*.jpg")))]
        with ThreadPoolExecutor(12) as ex:
            ims = list(ex.map(_load_one_image_224, [p for p, _ in items]))
        return np.stack(ims), np.array([c for _, c in items], np.int64)

    X_all, y_all = load("train")
    X_te, y_te = load("test")
    tr, va = stratified_val(y_all, 0.10, seed=42)
    X_tr, y_tr, X_va, y_va = X_all[tr], y_all[tr], X_all[va], y_all[va]
    del X_all
    print(f"train={len(X_tr)} val={len(X_va)} test={len(X_te)}")

    counts = np.bincount(y_tr, minlength=7).astype(np.float32)
    w = (counts.sum() / (7 * counts)) ** 0.35
    cls_w = torch.tensor(w / w.mean(), device=DEV)

    torch.manual_seed(2026)
    model = DANEmotionNet(num_classes=7, num_heads=4, feat_dim=256, pretrained_backbone=True).to(DEV)
    ema = EMA(model)
    heads = list(model.heads.parameters()) + list(model.classifier.parameters())
    opt = torch.optim.AdamW([{"params": heads, "lr": 3e-4}, {"params": model.backbone.parameters(), "lr": 1e-4}],
                            weight_decay=1e-4)
    base_lrs = [3e-4, 1e-4]
    scaler = torch.amp.GradScaler("cuda")
    rng = np.random.default_rng(2026)
    torch.cuda.reset_peak_memory_stats()
    best, best_ep, hist = -1, 0, []
    ckpt = r"D:\AAM_cv\checkpoints\dan_rafdb_v2.pth"
    t0 = time.perf_counter()
    for ep in range(1, epochs + 1):
        for g, b in zip(opt.param_groups, base_lrs):
            g["lr"] = b * lr_mult(ep, epochs)
        model.train()
        perm = rng.permutation(len(X_tr))
        tot, n = 0.0, 0
        for s in range(0, len(perm), 32):
            idx = perm[s:s + 32]
            if len(idx) < 2:
                continue
            xb = to_tensor(X_tr[idx])
            yb = torch.from_numpy(y_tr[idx]).to(DEV)
            if rng.random() < 0.5:
                xb = rotate_batch(xb, 10.0, rng)
            xb = gpu_augment_batch(xb, MEAN, STD, rng)
            lam, yb2 = 1.0, yb
            if rng.random() < 0.5:
                lam = float(rng.beta(0.4, 0.4))
                j = torch.randperm(len(idx), device=DEV)
                yb2 = yb[j]
                if rng.random() < 0.5:  # mixup
                    xb = lam * xb + (1 - lam) * xb[j]
                else:                    # cutmix
                    H = xb.shape[2]
                    r = int(H * np.sqrt(1 - lam))
                    cy, cx = rng.integers(0, H), rng.integers(0, H)
                    y0, y1, x0, x1 = max(0, cy - r // 2), min(H, cy + r // 2), max(0, cx - r // 2), min(H, cx + r // 2)
                    xb[:, :, y0:y1, x0:x1] = xb[j][:, :, y0:y1, x0:x1]
                    lam = 1 - (y1 - y0) * (x1 - x0) / (H * H)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.float16):
                logits, smaps = model(xb)
                ce = lam * F.cross_entropy(logits, yb, weight=cls_w, label_smoothing=0.08) + \
                    (1 - lam) * F.cross_entropy(logits, yb2, weight=cls_w, label_smoothing=0.08)
                loss = ce + 0.08 * model.partition_affinity_loss(smaps)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            ema.update(model)
            tot += float(loss.item()); n += 1
        pv = predict_probs(ema.m, X_va, 1)[0].argmax(1)
        m = compute_classification_metrics(y_va, pv, EMOTION_7_CLASSES)
        score = 0.5 * m["overall_accuracy"] + 0.5 * m["macro_f1"]
        hist.append({"epoch": ep, "train_loss": round(tot / n, 4), "val_acc": m["overall_accuracy"],
                     "val_macro_f1": m["macro_f1"], "val_balanced_acc": m["balanced_accuracy"]})
        print(f"  [DAN v2 {ep:02d}/{epochs}] loss={tot/n:.4f} val acc={m['overall_accuracy']*100:.2f} "
              f"F1={m['macro_f1']*100:.2f}", flush=True)
        if score > best:
            best, best_ep = score, ep
            torch.save({"epoch": ep, "model_state_dict": ema.m.state_dict(), "classes": EMOTION_7_CLASSES,
                        "selection": "validation (10% of RAF-DB train), EMA weights"}, ckpt)
    train_sec = time.perf_counter() - t0
    model.load_state_dict(torch.load(ckpt, weights_only=False)["model_state_dict"])
    pt = predict_probs(model, X_te, 1)[0].argmax(1)
    test = compute_classification_metrics(y_te, pt, EMOTION_7_CLASSES)
    print(f"--> DAN v2 best val epoch {best_ep}; TEST acc={test['overall_accuracy']*100:.2f} "
          f"F1={test['macro_f1']*100:.2f} bal={test['balanced_accuracy']*100:.2f}")
    res = {"train": len(X_tr), "val": len(X_va), "test": len(X_te), "epochs": epochs, "best_val_epoch": best_ep,
           "training_time_sec": round(train_sec, 1),
           "peak_vram_mb": round(torch.cuda.max_memory_allocated() / 2 ** 20, 1),
           "test_metrics": test, "history": hist, "checkpoint": ckpt}
    del model, ema, opt, X_tr, X_te, X_va
    torch.cuda.empty_cache()
    return res


# =============================================================================
def train_effnet(epochs=35):
    print("\n=== v2 EfficientNet-B0 on FairFace (val-selected, EMA, robust aug from start) ===")
    ff = r"D:\AAM_cv\datasets\raw\FairFace"
    with open(os.path.join(ff, "fairface_labeled_val.csv"), encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    with ThreadPoolExecutor(12) as ex:
        ims = list(ex.map(_load_one_image_224, [os.path.join(ff, "fairface_val_images", r["image_name"]) for r in rows]))
    X = np.stack(ims)
    ya = np.array([map_fairface_age(r["age"]) for r in rows], np.int64)
    yg = np.array([0 if r["gender"] == "Female" else 1 for r in rows], np.int64)
    perm = np.random.default_rng(2026).permutation(len(rows))   # identical to v1 split
    n_tr = int(round(0.80 * len(rows)))
    trall, te = perm[:n_tr], perm[n_tr:]
    strat = ya[trall] * 2 + yg[trall]
    tr_rel, va_rel = stratified_val(strat, 0.10, seed=42)
    tr, va = trall[tr_rel], trall[va_rel]
    X_tr, X_va, X_te = X[tr], X[va], X[te]
    ya_tr, yg_tr, ya_va, yg_va, ya_te, yg_te = ya[tr], yg[tr], ya[va], yg[va], ya[te], yg[te]
    del X
    print(f"train={len(X_tr)} val={len(X_va)} test={len(X_te)} (test identical to v1)")

    cnt = np.bincount(ya_tr, minlength=6).astype(np.float32)
    aw = (cnt.sum() / (6 * cnt)) ** 0.30
    age_w = torch.tensor(aw / aw.mean(), device=DEV)
    grid = torch.arange(6, device=DEV, dtype=torch.float32)

    torch.manual_seed(2026)
    model = MultiTaskEfficientNetB0(pretrained_backbone=True).to(DEV)
    ema = EMA(model)
    opt = torch.optim.AdamW([{"params": model.age_head.parameters(), "lr": 3e-4},
                             {"params": model.gender_head.parameters(), "lr": 3e-4},
                             {"params": model.features.parameters(), "lr": 1e-4}], weight_decay=1e-4)
    base_lrs = [3e-4, 3e-4, 1e-4]
    scaler = torch.amp.GradScaler("cuda")
    rng = np.random.default_rng(2026)
    torch.cuda.reset_peak_memory_stats()
    best, best_ep, hist = -1, 0, []
    ckpt = r"D:\AAM_cv\checkpoints\effnet_b0_fairface_v2.pth"
    t0 = time.perf_counter()
    for ep in range(1, epochs + 1):
        for g, b in zip(opt.param_groups, base_lrs):
            g["lr"] = b * lr_mult(ep, epochs)
        model.train()
        p = rng.permutation(len(X_tr))
        tot, n = 0.0, 0
        for s in range(0, len(p), 32):
            idx = p[s:s + 32]
            if len(idx) < 2:
                continue
            xb = augment_gender_robust_batch(to_tensor(X_tr[idx]), MEAN, STD, rng)
            if rng.random() < 0.35:   # cutout
                eh, ew = int(rng.integers(18, 42)), int(rng.integers(18, 42))
                ey, ex_ = int(rng.integers(0, 224 - eh)), int(rng.integers(0, 224 - ew))
                xb[:, :, ey:ey + eh, ex_:ex_ + ew] = 0.0
            yab = torch.from_numpy(ya_tr[idx]).to(DEV)
            ygb = torch.from_numpy(yg_tr[idx]).to(DEV)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.float16):
                al, gl = model(xb)
                l_age = F.cross_entropy(al, yab, weight=age_w, label_smoothing=0.08) + \
                    0.25 * F.smooth_l1_loss((F.softmax(al.float(), 1) * grid).sum(1), yab.float())
                l_gen = F.cross_entropy(gl, ygb, label_smoothing=0.05)
                loss = l_age + 1.5 * l_gen
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            ema.update(model)
            tot += float(loss.item()); n += 1
        pa, pg = predict_probs(ema.m, X_va, 2)
        ma = compute_classification_metrics(ya_va, pa.argmax(1), AGE_GROUP_CLASSES)
        mg = compute_classification_metrics(yg_va, pg.argmax(1), GENDER_LABEL_CLASSES)
        w1 = float((np.abs(ya_va - pa.argmax(1)) <= 1).mean())
        score = 0.5 * (ma["overall_accuracy"] + ma["macro_f1"]) + 0.5 * (mg["overall_accuracy"] + mg["macro_f1"])
        hist.append({"epoch": ep, "train_loss": round(tot / n, 4), "val_age_acc": ma["overall_accuracy"],
                     "val_age_within1": round(w1, 4), "val_age_macro_f1": ma["macro_f1"],
                     "val_gender_acc": mg["overall_accuracy"]})
        print(f"  [EffNet v2 {ep:02d}/{epochs}] loss={tot/n:.4f} val age={ma['overall_accuracy']*100:.2f} "
              f"(+-1 {w1*100:.2f}) gender={mg['overall_accuracy']*100:.2f}", flush=True)
        if score > best:
            best, best_ep = score, ep
            torch.save({"epoch": ep, "model_state_dict": ema.m.state_dict(), "age_classes": AGE_GROUP_CLASSES,
                        "gender_classes": GENDER_LABEL_CLASSES,
                        "selection": "validation (10% of FairFace train part), EMA weights"}, ckpt)
    train_sec = time.perf_counter() - t0
    model.load_state_dict(torch.load(ckpt, weights_only=False)["model_state_dict"])
    pa, pg = predict_probs(model, X_te, 2)
    ta = compute_classification_metrics(ya_te, pa.argmax(1), AGE_GROUP_CLASSES)
    ta["within_1_group_accuracy"] = round(float((np.abs(ya_te - pa.argmax(1)) <= 1).mean()), 4)
    tg = compute_classification_metrics(yg_te, pg.argmax(1), GENDER_LABEL_CLASSES)
    print(f"--> EffNet v2 best val epoch {best_ep}; TEST gender={tg['overall_accuracy']*100:.2f} "
          f"age={ta['overall_accuracy']*100:.2f} (+-1 {ta['within_1_group_accuracy']*100:.2f})")
    res = {"train": len(X_tr), "val": len(X_va), "test": len(X_te), "epochs": epochs, "best_val_epoch": best_ep,
           "training_time_sec": round(train_sec, 1),
           "peak_vram_mb": round(torch.cuda.max_memory_allocated() / 2 ** 20, 1),
           "test_age_metrics": ta, "test_gender_metrics": tg, "history": hist, "checkpoint": ckpt}
    del model, ema, opt
    torch.cuda.empty_cache()
    return res


if __name__ == "__main__":
    results = {}
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if os.path.isfile(OUT):
        with open(OUT, encoding="utf-8") as f:
            results = json.load(f)
    if which in ("all", "dan"):
        results["dan_v2"] = train_dan(40)
        with open(OUT, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
    if which in ("all", "effnet"):
        results["effnet_v2"] = train_effnet(35)
        with open(OUT, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
    print("saved", OUT)
