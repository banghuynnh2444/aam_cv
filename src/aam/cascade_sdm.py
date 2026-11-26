"""Cascaded Shape Regression (Supervised Descent Method style) on top of the Classical AAM.

Why: the original Project-Out AAM fitting (classical_aam.ClassicalAAM.fit_image) learns a single
linear update from raw grayscale residuals of only 480 small perturbations. In practice it barely
moves the shape away from its 5-keypoint initialisation.

This module replaces that step with a cascade of K ridge regressors:
    x_{k+1} = x_k + R_k * phi(I, x_k) + b_k
where phi = HOG descriptors sampled around each of the 68 current landmarks, computed in a
similarity-normalised canonical frame. Every stage is trained on the residuals left by the previous
stage, using many perturbed initialisations per training image drawn from the same distribution the
detector produces at test time. After each stage the shape is projected onto the AAM shape PCA
(clipped to +/-3 sigma), so outputs always stay plausible faces.
"""
from typing import List, Optional, Tuple, Dict, Any
import cv2
import numpy as np


def _similarity(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """2x3 similarity transform mapping src (N,2) -> dst (N,2) (least squares, Umeyama)."""
    src = np.asarray(src, np.float64)
    dst = np.asarray(dst, np.float64)
    mu_s, mu_d = src.mean(0), dst.mean(0)
    s0, d0 = src - mu_s, dst - mu_d
    var_s = (s0 ** 2).sum() / len(src)
    cov = d0.T @ s0 / len(src)
    U, S, Vt = np.linalg.svd(cov)
    D = np.eye(2)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        D[1, 1] = -1
    R = U @ D @ Vt
    scale = np.trace(np.diag(S) @ D) / max(var_s, 1e-12)
    t = mu_d - scale * R @ mu_s
    M = np.zeros((2, 3))
    M[:, :2] = scale * R
    M[:, 2] = t
    return M


def _apply(M: np.ndarray, pts: np.ndarray) -> np.ndarray:
    return pts @ M[:, :2].T + M[:, 2]


def _invert(M: np.ndarray) -> np.ndarray:
    A = M[:, :2]
    Ai = np.linalg.inv(A)
    out = np.zeros((2, 3))
    out[:, :2] = Ai
    out[:, 2] = -Ai @ M[:, 2]
    return out


class CascadeSDM:
    """Cascade of HOG-feature linear regressors for 68-point landmark refinement."""

    def __init__(
        self,
        mean_shape: np.ndarray,
        shape_pcs: np.ndarray,
        shape_eigvals: np.ndarray,
        n_stages: int = 5,
        canvas: int = 160,
        win: int = 24,
        ridge_lambdas: Tuple[float, ...] = (1.0, 3.0, 10.0, 30.0, 100.0, 300.0),
    ):
        self.n_stages = int(n_stages)
        self.canvas = int(canvas)
        self.win = int(win)
        self.ridge_lambdas = tuple(ridge_lambdas)
        self.shape_pcs = np.asarray(shape_pcs, np.float64)          # (136, k)
        self.shape_eigvals = np.asarray(shape_eigvals, np.float64)  # (k,)
        # Reference shape: AAM mean shape fitted into the canonical canvas with a 15% margin.
        ms = np.asarray(mean_shape, np.float64).reshape(68, 2)
        lo, hi = ms.min(0), ms.max(0)
        side = float((hi - lo).max())
        margin = 0.15 * canvas
        scale = (canvas - 2 * margin) / side
        self.ref_shape = (ms - (lo + hi) / 2.0) * scale + canvas / 2.0
        self.ref_mean_ms = ms
        self.ref_scale = scale
        self.cell = 8          # 3x3 cells of 8x8 px around each landmark (24x24 window), 9 bins
        self.n_bins = 9
        self.pad = win
        self.stages_: List[Dict[str, np.ndarray]] = []
        self.feat_mu_: List[np.ndarray] = []
        self.feat_sd_: List[np.ndarray] = []

    # ------------------------------------------------------------------ features
    def _to_canvas(self, gray: np.ndarray, shape: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Warps the image so that `shape` is similarity-aligned to the reference frame."""
        M = _similarity(shape, self.ref_shape)
        size = self.canvas + 2 * self.pad
        M_pad = M.copy()
        M_pad[:, 2] += self.pad
        warped = cv2.warpAffine(gray, M_pad, (size, size), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        return warped, M

    def _features(self, warped: np.ndarray, canvas_shape: np.ndarray) -> np.ndarray:
        """HOG (9 unsigned orientation bins, 3x3 cells of 8x8 px, L2-Hys) around every landmark."""
        img = warped.astype(np.float32)
        gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=1)
        gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=1)
        mag, ang = cv2.cartToPolar(gx, gy, angleInDegrees=True)
        ang = np.mod(ang, 180.0)
        b = np.minimum((ang / (180.0 / self.n_bins)).astype(np.int32), self.n_bins - 1)
        c = self.cell
        cell_sums = np.stack([
            cv2.boxFilter(np.where(b == k, mag, 0.0).astype(np.float32), -1, (c, c), normalize=False,
                          borderType=cv2.BORDER_REPLICATE)
            for k in range(self.n_bins)
        ], axis=0)  # (9, H, W): sum of magnitudes in the 8x8 cell centred at each pixel
        H, W = img.shape
        lim_lo, lim_hi = c + 1, min(H, W) - c - 2
        xs = np.clip(np.round(canvas_shape[:, 0] + self.pad).astype(int), lim_lo, lim_hi)
        ys = np.clip(np.round(canvas_shape[:, 1] + self.pad).astype(int), lim_lo, lim_hi)
        offs = np.array([-c, 0, c])
        yy = (ys[:, None, None] + offs[None, :, None]).repeat(3, axis=2)   # (68,3,3)
        xx = (xs[:, None, None] + offs[None, None, :]).repeat(3, axis=1)
        f = cell_sums[:, yy, xx]                       # (9, 68, 3, 3)
        f = f.transpose(1, 2, 3, 0).reshape(68, -1)    # (68, 81)
        f = f / np.sqrt((f ** 2).sum(1, keepdims=True) + 1e-6)
        f = np.minimum(f, 0.2)
        f = f / np.sqrt((f ** 2).sum(1, keepdims=True) + 1e-6)
        return f.reshape(-1).astype(np.float32)

    def _project(self, canvas_shape: np.ndarray) -> np.ndarray:
        """Constrain a canvas-frame shape to the AAM shape PCA subspace (+/-3 sigma)."""
        M = _similarity(canvas_shape, self.ref_shape)
        aligned = _apply(M, canvas_shape)
        d = (aligned - self.ref_shape).reshape(-1) / self.ref_scale
        b = self.shape_pcs.T @ d
        lim = 3.0 * np.sqrt(np.maximum(self.shape_eigvals, 1e-9))
        b = np.clip(b, -lim, lim)
        recon = self.ref_shape + (self.shape_pcs @ b).reshape(68, 2) * self.ref_scale
        return _apply(_invert(M), recon)

    # ------------------------------------------------------------------ training
    @staticmethod
    def _gray(img: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img

    def fit(
        self,
        images: List[np.ndarray],
        gt_shapes: np.ndarray,
        init_fn,
        n_inits: int = 10,
        val_images: Optional[List[np.ndarray]] = None,
        val_gt: Optional[np.ndarray] = None,
        val_inits: Optional[np.ndarray] = None,
        rng: Optional[np.random.Generator] = None,
        log=print,
    ) -> Dict[str, Any]:
        """init_fn(gt_shape, rng) -> initial (68,2) shape in image coordinates."""
        rng = rng or np.random.default_rng(0)
        grays = [self._gray(im) for im in images]
        samples = []  # (img_idx, current_image_shape)
        for i, gt in enumerate(gt_shapes):
            for _ in range(n_inits):
                samples.append((i, init_fn(gt, rng).astype(np.float64)))
        cur = [s for _, s in samples]
        idxs = [i for i, _ in samples]
        history = []
        for k in range(self.n_stages):
            X = np.zeros((len(cur), 0), np.float32)
            feats, targets = [], []
            for j, (i, shp) in enumerate(zip(idxs, cur)):
                warped, M = self._to_canvas(grays[i], shp)
                cs = _apply(M, shp)
                gs = _apply(M, gt_shapes[i].astype(np.float64))
                feats.append(self._features(warped, cs))
                targets.append((gs - cs).reshape(-1))
            X = np.stack(feats).astype(np.float64)
            Y = np.stack(targets)
            mu = X.mean(0)
            sd = X.std(0) + 1e-3
            Z = (X - mu) / sd
            Zb = np.hstack([Z, np.ones((len(Z), 1))])
            G = Zb.T @ Zb
            ZY = Zb.T @ Y
            # pick ridge lambda on held-out 20% of the training samples
            perm = rng.permutation(len(Z))
            ho = perm[: len(Z) // 5]
            tr = perm[len(Z) // 5:]
            Gtr = Zb[tr].T @ Zb[tr]
            ZYtr = Zb[tr].T @ Y[tr]
            best_lam, best_err = None, np.inf
            for lam in self.ridge_lambdas:
                reg = lam * len(tr) * np.eye(G.shape[0]) / 1000.0
                reg[-1, -1] = 0.0
                W = np.linalg.solve(Gtr + reg, ZYtr)
                err = float(np.mean((Zb[ho] @ W - Y[ho]) ** 2))
                if err < best_err:
                    best_lam, best_err = lam, err
            reg = best_lam * len(Z) * np.eye(G.shape[0]) / 1000.0
            reg[-1, -1] = 0.0
            W = np.linalg.solve(G + reg, ZY).astype(np.float32)
            self.stages_.append({"W": W})
            self.feat_mu_.append(mu.astype(np.float32))
            self.feat_sd_.append(sd.astype(np.float32))
            # update training shapes
            pred = Zb @ W
            new_cur = []
            for j, (i, shp) in enumerate(zip(idxs, cur)):
                M = _similarity(shp, self.ref_shape)
                cs = _apply(M, shp) + pred[j].reshape(68, 2)
                cs = self._project(cs)
                new_cur.append(_apply(_invert(M), cs))
            cur = new_cur
            tr_nme = float(np.mean([nme68(c, gt_shapes[i]) for c, i in zip(cur, idxs)]))
            rec = {"stage": k + 1, "lambda": best_lam, "train_nme_pct": round(tr_nme * 100, 3)}
            if val_images is not None:
                pv = [self.predict(im, ini, n_stages=k + 1) for im, ini in zip(val_images, val_inits)]
                rec["val_nme_pct"] = round(float(np.mean([nme68(p, g) for p, g in zip(pv, val_gt)])) * 100, 3)
            history.append(rec)
            log(f"  [SDM stage {k + 1}/{self.n_stages}] {rec}")
        return {"history": history, "n_train_samples": len(samples), "feature_dim": int(X.shape[1])}

    # ------------------------------------------------------------------ inference
    def predict(self, img: np.ndarray, init_shape: np.ndarray, n_stages: Optional[int] = None) -> np.ndarray:
        gray = self._gray(img)
        shp = np.asarray(init_shape, np.float64).reshape(68, 2)
        n = len(self.stages_) if n_stages is None else min(n_stages, len(self.stages_))
        for k in range(n):
            warped, M = self._to_canvas(gray, shp)
            cs = _apply(M, shp)
            z = (self._features(warped, cs) - self.feat_mu_[k]) / self.feat_sd_[k]
            delta = np.append(z, 1.0).astype(np.float32) @ self.stages_[k]["W"]
            cs = self._project(cs + delta.reshape(68, 2))
            shp = _apply(_invert(M), cs)
        return shp.astype(np.float32)

    # ------------------------------------------------------------------ io
    def save(self, path: str) -> None:
        d = {
            "n_stages": len(self.stages_), "canvas": self.canvas, "win": self.win,
            "mean_shape": self.ref_mean_ms, "shape_pcs": self.shape_pcs, "shape_eigvals": self.shape_eigvals,
        }
        for k, st in enumerate(self.stages_):
            d[f"W_{k}"] = st["W"]
            d[f"mu_{k}"] = self.feat_mu_[k]
            d[f"sd_{k}"] = self.feat_sd_[k]
        np.savez_compressed(path, **d)

    @classmethod
    def load(cls, path: str) -> "CascadeSDM":
        z = np.load(path)
        obj = cls(z["mean_shape"], z["shape_pcs"], z["shape_eigvals"], n_stages=int(z["n_stages"]),
                  canvas=int(z["canvas"]), win=int(z["win"]))
        for k in range(int(z["n_stages"])):
            obj.stages_.append({"W": z[f"W_{k}"]})
            obj.feat_mu_.append(z[f"mu_{k}"])
            obj.feat_sd_.append(z[f"sd_{k}"])
        return obj


def nme68(pred: np.ndarray, gt: np.ndarray) -> float:
    pred = np.asarray(pred, np.float64).reshape(68, 2)
    gt = np.asarray(gt, np.float64).reshape(68, 2)
    iod = float(np.linalg.norm(gt[36] - gt[45]))
    return float(np.linalg.norm(pred - gt, axis=1).mean() / max(iod, 1e-6))
