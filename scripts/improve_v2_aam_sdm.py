"""v2 improvement - AAM landmark fitting with a HOG Cascaded Shape Regressor (SDM).

Uses EXACTLY the same data split as the original AAM run (scripts/train_aam_dan_effnet_real.py):
  - train: 250 x 300-W (sorted, first 250) + 312 personal s30   (the existing AAM was built on these)
  - test : 45 x 300-W held out (last 45) + 176 personal s10
and the same test-time initialisation (5 keypoints from GT + N(0, 0.025*IOD) noise, seed 2026).

10% of the training images are held out as a validation set to monitor the cascade stages.
Unlike the v1 evaluation, the reported fitted NME is the actual output of the fitter: v1 kept
min(NME_init, NME_fit), which needs the ground truth at test time and is therefore optimistic.
"""
import os
import sys
import csv
import glob
import json
import time
import numpy as np

sys.path.insert(0, r"D:\AAM_cv")
sys.path.insert(0, r"D:\AAM_cv\scripts")
from src.aam.classical_aam import ClassicalAAM, compute_nme_68
from src.aam.cascade_sdm import CascadeSDM, nme68
from src.geometry.landmarks68 import read_pts68, extract_5kps_from_68
from train_aam_dan_effnet_real import imread_unicode, crop_face_with_pts

OUT_JSON = r"D:\AAM_cv\experiments\v2_aam_sdm_results.json"
SDM_CKPT = r"D:\AAM_cv\checkpoints\aam_cascade_sdm_v2.npz"


def load_data():
    pts_files = sorted(glob.glob(r"D:\AAM_cv\datasets\raw\300W\*.pts"))
    w_imgs, w_shapes = [], []
    for pp in pts_files:
        im = imread_unicode(pp[:-4] + ".png")
        if im is None:
            continue
        try:
            pts = read_pts68(pp)
        except Exception:
            continue
        ci, cp = crop_face_with_pts(im, pts, target_size=256, margin=0.25)
        w_imgs.append(ci)
        w_shapes.append(cp)
    n_tr = len(w_imgs) - 45
    lm = np.load(r"D:\AAM_cv\datasets\processed\landmarks68\personal_landmarks68_all.npz")
    coords = lm["coords_clipped"]
    with open(r"D:\AAM_cv\datasets\processed\landmarks68\landmarks68_manifest.csv", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    s30_i, s30_s, s10_i, s10_s = [], [], [], []
    for idx, r in enumerate(rows):
        im = imread_unicode(os.path.join(r"D:\AAM_cv", r["image_path"]))
        if r["session"] == "s30":
            s30_i.append(im); s30_s.append(coords[idx])
        else:
            s10_i.append(im); s10_s.append(coords[idx])
    return (w_imgs[:n_tr], np.stack(w_shapes[:n_tr]), w_imgs[n_tr:], np.stack(w_shapes[n_tr:]),
            s30_i, np.stack(s30_s), s10_i, np.stack(s10_s))


def main():
    t0 = time.perf_counter()
    w_tr_i, w_tr_s, w_te_i, w_te_s, s30_i, s30_s, s10_i, s10_s = load_data()
    tr_imgs = w_tr_i + s30_i
    tr_shapes = np.concatenate([w_tr_s, s30_s]).astype(np.float64)
    print(f"Loaded train={len(tr_imgs)}  test300W={len(w_te_i)}  testS10={len(s10_i)}  ({time.perf_counter()-t0:.1f}s)")

    aam = ClassicalAAM.load(r"D:\AAM_cv\checkpoints\classical_aam_model.npz")

    def init_train(gt, rng):
        iod = float(np.linalg.norm(gt[36] - gt[45]))
        sd = rng.uniform(0.015, 0.05) * iod
        k5 = extract_5kps_from_68(gt) + rng.normal(0, sd, size=(5, 2))
        return aam.init_shape_from_5kps(k5)

    # validation = 10% of training images (never the test images)
    rng = np.random.default_rng(7)
    perm = rng.permutation(len(tr_imgs))
    n_val = len(tr_imgs) // 10
    val_idx, fit_idx = perm[:n_val], perm[n_val:]
    vrng = np.random.default_rng(11)
    val_imgs = [tr_imgs[i] for i in val_idx]
    val_gt = tr_shapes[val_idx]
    val_inits = [init_train(g, vrng) for g in val_gt]

    sdm = CascadeSDM(aam.mean_shape_, aam.shape_pcs_, aam.shape_eigvals_, n_stages=5)
    t1 = time.perf_counter()
    info = sdm.fit([tr_imgs[i] for i in fit_idx], tr_shapes[fit_idx], init_train, n_inits=10,
                   val_images=val_imgs, val_gt=val_gt, val_inits=val_inits, rng=np.random.default_rng(3))
    train_sec = time.perf_counter() - t1
    sdm.save(SDM_CKPT)

    def evaluate(imgs, shapes):
        rng = np.random.default_rng(2026)  # same protocol & seed as v1 evaluation
        r = {"init": [], "aam_po_raw": [], "aam_po_min_v1": [], "sdm": []}
        t_aam, t_sdm = 0.0, 0.0
        for im, gt in zip(imgs, shapes):
            iod = float(np.linalg.norm(gt[36] - gt[45]))
            k5 = extract_5kps_from_68(gt) + rng.normal(0.0, 0.025 * iod, size=(5, 2)).astype(np.float32)
            init = aam.init_shape_from_5kps(k5)
            ni = compute_nme_68(init, gt)
            ta = time.perf_counter(); fit, _ = aam.fit_image(im, init, num_iters=8); t_aam += time.perf_counter() - ta
            nf = compute_nme_68(fit, gt)
            ts = time.perf_counter(); ps = sdm.predict(im, init); t_sdm += time.perf_counter() - ts
            r["init"].append(ni); r["aam_po_raw"].append(nf); r["aam_po_min_v1"].append(min(ni, nf))
            r["sdm"].append(nme68(ps, gt))
        out = {}
        for k, v in r.items():
            v = np.array(v)
            out[k] = {"nme_mean_pct": round(float(v.mean()) * 100, 2), "nme_median_pct": round(float(np.median(v)) * 100, 2),
                      "failure_rate_gt_8pct": round(float((v > 0.08).mean()) * 100, 2),
                      "ced_auc_0_08": round(float(np.mean([(v <= t).mean() for t in np.linspace(0, 0.08, 81)])), 4)}
        out["ms_per_face"] = {"aam_po": round(1000 * t_aam / len(imgs), 1), "sdm": round(1000 * t_sdm / len(imgs), 1)}
        out["n"] = len(imgs)
        return out

    res = {
        "method": "HOG cascaded shape regression (SDM), 5 stages, PCA-constrained, 10 inits/image",
        "train_images": len(fit_idx), "val_images": int(n_val), **info,
        "train_time_sec": round(train_sec, 1),
        "test_300w_45": evaluate(w_te_i, w_te_s),
        "test_personal_s10_176": evaluate(s10_i, s10_s),
        "checkpoint": SDM_CKPT,
    }
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    print(json.dumps({k: res[k] for k in ("test_300w_45", "test_personal_s10_176")}, indent=2))


if __name__ == "__main__":
    main()
