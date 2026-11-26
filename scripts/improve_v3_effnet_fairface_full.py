"""v3: Multi-task EfficientNet-B0 (age 6 groups + gender) trained on the OFFICIAL FairFace train set.

Data
  - Train: official FairFace train (86,744 images, padding 0.25, github.com/joojs/fairface, CC BY 4.0)
           + the v2 train part of FairFace val (7,888 images). A stratified 4% of the official train is held out.
  - Validation (checkpoint selection only): v2 validation (875 val-domain images) + the 4% official holdout.
  - Test: the exact same 2,191 FairFace-val images as v1/v2 (seed-2026 permutation), evaluated ONCE.
    Official train and val are disjoint, so there is no leakage into the test set.
Training: same recipe as v2 (EMA 0.999, robust distance-blur aug, cutout, weighted CE + expectation loss),
images are decoded on the fly with a thread pool (the full set does not fit comfortably in RAM).
"""
import os
import sys
import csv
import json
import time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, r"D:\AAM_cv")
sys.path.insert(0, r"D:\AAM_cv\scripts")
from src.attributes.age_gender_effnet import (
    AGE_GROUP_CLASSES, GENDER_LABEL_CLASSES, MultiTaskEfficientNetB0, compute_classification_metrics,
)
from train_full_40ep_dan_35ep_effnet import _load_one_image_224, map_fairface_age
from refine_effnet_fairface_gender import augment_gender_robust_batch
from improve_v2_dan_effnet import stratified_val, EMA, lr_mult, to_tensor, predict_probs, DEV, MEAN, STD

FF = r"D:\AAM_cv\datasets\raw\FairFace"
OFF = r"D:\AAM_cv\datasets\raw\FairFace_official"
CKPT = r"D:\AAM_cv\checkpoints\effnet_b0_fairface_v3.pth"
OUT = r"D:\AAM_cv\experiments\v3_effnet_results.json"
EPOCHS = int(os.environ.get("V3_EPOCHS", "14"))
BS = 48


def load_split():
    with open(os.path.join(FF, "fairface_labeled_val.csv"), encoding="utf-8") as f:
        vrows = list(csv.DictReader(f))
    vpaths = np.array([os.path.join(FF, "fairface_val_images", r["image_name"]) for r in vrows])
    vya = np.array([map_fairface_age(r["age"]) for r in vrows], np.int64)
    vyg = np.array([0 if r["gender"] == "Female" else 1 for r in vrows], np.int64)
    perm = np.random.default_rng(2026).permutation(len(vrows))          # identical to v1/v2
    n_tr = int(round(0.80 * len(vrows)))
    trall, te = perm[:n_tr], perm[n_tr:]
    tr_rel, va_rel = stratified_val(vya[trall] * 2 + vyg[trall], 0.10, seed=42)   # identical to v2
    vtr, vva = trall[tr_rel], trall[va_rel]

    with open(os.path.join(OFF, "fairface_label_train.csv"), encoding="utf-8") as f:
        orows = list(csv.DictReader(f))
    opaths = np.array([os.path.join(OFF, r["file"].replace("/", os.sep)) for r in orows])
    oya = np.array([map_fairface_age(r["age"]) for r in orows], np.int64)
    oyg = np.array([0 if r["gender"] == "Female" else 1 for r in orows], np.int64)
    otr, ova = stratified_val(oya * 2 + oyg, 0.04, seed=7)

    tr = (np.concatenate([opaths[otr], vpaths[vtr]]), np.concatenate([oya[otr], vya[vtr]]),
          np.concatenate([oyg[otr], vyg[vtr]]))
    va = (np.concatenate([vpaths[vva], opaths[ova]]), np.concatenate([vya[vva], oya[ova]]),
          np.concatenate([vyg[vva], oyg[ova]]), len(vva))
    tst = (vpaths[te], vya[te], vyg[te])
    return tr, va, tst


def load_images(paths, ex):
    ims = list(ex.map(_load_one_image_224, paths))
    bad = [p for p, im in zip(paths, ims) if im is None]
    if bad:
        raise RuntimeError(f"{len(bad)} unreadable images, e.g. {bad[:3]}")
    return np.stack(ims)


def metrics(ya, yg, pa, pg):
    ma = compute_classification_metrics(ya, pa.argmax(1), AGE_GROUP_CLASSES)
    ma["within_1_group_accuracy"] = round(float((np.abs(ya - pa.argmax(1)) <= 1).mean()), 4)
    mg = compute_classification_metrics(yg, pg.argmax(1), GENDER_LABEL_CLASSES)
    return ma, mg


def main():
    (ptr, ya_tr, yg_tr), (pva, ya_va, yg_va, n_vdom), (pte, ya_te, yg_te) = load_split()
    ex = ThreadPoolExecutor(12)
    X_va, X_te = load_images(pva, ex), load_images(pte, ex)
    print(f"train={len(ptr)} val={len(pva)} (val-domain {n_vdom}) test={len(pte)} (identical to v1/v2)", flush=True)

    cnt = np.bincount(ya_tr, minlength=6).astype(np.float32)
    aw = (cnt.sum() / (6 * cnt)) ** 0.30
    age_w = torch.tensor(aw / aw.mean(), device=DEV)
    grid = torch.arange(6, device=DEV, dtype=torch.float32)

    torch.manual_seed(2026)
    model = MultiTaskEfficientNetB0(pretrained_backbone=True).to(DEV)
    ema = EMA(model, decay=0.9995)
    base_lrs = [3e-4, 3e-4, 1.5e-4]
    opt = torch.optim.AdamW([{"params": model.age_head.parameters(), "lr": base_lrs[0]},
                             {"params": model.gender_head.parameters(), "lr": base_lrs[1]},
                             {"params": model.features.parameters(), "lr": base_lrs[2]}], weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda")
    rng = np.random.default_rng(2026)
    torch.cuda.reset_peak_memory_stats()
    best, best_ep, hist = -1.0, 0, []
    t0 = time.perf_counter()
    for ep in range(1, EPOCHS + 1):
        model.train()
        p = rng.permutation(len(ptr))
        batches = [p[s:s + BS] for s in range(0, len(p) - 1, BS) if len(p[s:s + BS]) >= 2]
        nb = len(batches)
        # prefetch: decode batch i+1 while the GPU trains on batch i
        pool = ThreadPoolExecutor(1)
        fut = pool.submit(lambda b: np.stack(list(ex.map(_load_one_image_224, ptr[b]))), batches[0])
        tot, n, te0 = 0.0, 0, time.perf_counter()
        for i, idx in enumerate(batches):
            Xb = fut.result()
            if i + 1 < nb:
                fut = pool.submit(lambda b: np.stack(list(ex.map(_load_one_image_224, ptr[b]))), batches[i + 1])
            for g, b in zip(opt.param_groups, base_lrs):
                g["lr"] = b * lr_mult(ep - 1 + i / nb, EPOCHS, warm=1)
            xb = augment_gender_robust_batch(to_tensor(Xb), MEAN, STD, rng)
            if rng.random() < 0.35:
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
            if i % 400 == 0:
                print(f"    ep{ep} batch {i}/{nb} loss={tot/n:.4f} {(time.perf_counter()-te0)/(i+1)*1000:.0f} ms/batch",
                      flush=True)
        pool.shutdown()
        pa, pg = predict_probs(ema.m, X_va, 2)
        ma, mg = metrics(ya_va, yg_va, pa, pg)
        mad, _ = metrics(ya_va[:n_vdom], yg_va[:n_vdom], pa[:n_vdom], pg[:n_vdom])
        score = 0.5 * (ma["overall_accuracy"] + ma["macro_f1"]) + 0.5 * (mg["overall_accuracy"] + mg["macro_f1"])
        hist.append({"epoch": ep, "train_loss": round(tot / n, 4), "val_age_acc": ma["overall_accuracy"],
                     "val_age_within1": ma["within_1_group_accuracy"], "val_age_macro_f1": ma["macro_f1"],
                     "val_gender_acc": mg["overall_accuracy"], "val_domain_age_acc": mad["overall_accuracy"],
                     "epoch_sec": round(time.perf_counter() - te0, 1)})
        print(f"  [EffNet v3 {ep:02d}/{EPOCHS}] loss={tot/n:.4f} val age={ma['overall_accuracy']*100:.2f} "
              f"(+-1 {ma['within_1_group_accuracy']*100:.2f}, F1 {ma['macro_f1']*100:.2f}) "
              f"gender={mg['overall_accuracy']*100:.2f} | val-domain age={mad['overall_accuracy']*100:.2f} "
              f"[{time.perf_counter()-te0:.0f}s]", flush=True)
        if score > best:
            best, best_ep = score, ep
            torch.save({"epoch": ep, "model_state_dict": ema.m.state_dict(), "age_classes": AGE_GROUP_CLASSES,
                        "gender_classes": GENDER_LABEL_CLASSES,
                        "selection": "validation (v2 val + 4% official train), EMA weights"}, CKPT)
    train_sec = time.perf_counter() - t0
    model.load_state_dict(torch.load(CKPT, weights_only=False)["model_state_dict"])
    pa, pg = predict_probs(model, X_te, 2)
    ta, tg = metrics(ya_te, yg_te, pa, pg)
    print(f"--> EffNet v3 best val epoch {best_ep}; TEST gender={tg['overall_accuracy']*100:.2f} "
          f"age={ta['overall_accuracy']*100:.2f} (+-1 {ta['within_1_group_accuracy']*100:.2f}, "
          f"F1 {ta['macro_f1']*100:.2f})", flush=True)
    res = {"train": len(ptr), "val": len(pva), "test": len(pte), "epochs": EPOCHS, "best_val_epoch": best_ep,
           "training_time_sec": round(train_sec, 1),
           "peak_vram_mb": round(torch.cuda.max_memory_allocated() / 2 ** 20, 1),
           "test_age_metrics": ta, "test_gender_metrics": tg, "history": hist, "checkpoint": CKPT,
           "data_source": "FairFace official train (padding 0.25, CC BY 4.0) + v2 train part of FairFace val"}
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
