"""Classical Active Appearance Model (AAM) Trained from Scratch.

Implements:
1. Generalized Procrustes Analysis (GPA) for 68-point facial shapes.
2. Statistical Shape Model via PCA: s = s_mean + P_s * b_s.
3. Delaunay Triangulation on canonical 128x128 reference frame & vectorized
   Piecewise Affine Warping (image <-> shape-free 128x128 texture).
4. Statistical Appearance (Texture) Model via PCA (with chunked processing):
   g = g_mean + P_g * b_g.
5. Combined Shape + Appearance AAM feature extraction & Nearest-Centroid / Linear
   Ridge-SVM identity classifier (for Classical AAM vs Deep Learning comparison).
6. Active AAM Fitting via Project-Out Gauss-Newton / Ridge-Jacobian iterative optimization
   from bounding-box mean-shape initialization, with NME & failure-rate evaluation.
"""
import os
from typing import Dict, Any, Tuple, List, Optional
import cv2
import numpy as np


def compute_nme_68(pred_pts: np.ndarray, gt_pts: np.ndarray) -> float:
    """Computes 300-W standard Normalized Mean Error (NME) as a ratio (e.g. 0.035 = 3.5%),
    normalized by outer-eye-corner inter-ocular distance ||gt[36] - gt[45]||_2.
    """
    pred = np.asarray(pred_pts, dtype=np.float64).reshape(68, 2)
    gt = np.asarray(gt_pts, dtype=np.float64).reshape(68, 2)
    iod = float(np.linalg.norm(gt[36] - gt[45]))
    if iod < 1e-6:
        iod = float(np.linalg.norm(gt.max(axis=0) - gt.min(axis=0)))
    pt_dists = np.linalg.norm(pred - gt, axis=1)
    return float(pt_dists.mean() / max(iod, 1e-6))


class ClassicalAAM:
    """True Classical Active Appearance Model (Procrustes + Delaunay PWA + PCA + Fitting)."""

    def __init__(
        self,
        texture_size: int = 128,
        shape_variance: float = 0.96,
        texture_variance: float = 0.95,
        max_shape_components: int = 25,
        max_texture_components: int = 60,
        chunk_size: int = 64,
    ):
        self.texture_size = int(texture_size)
        self.shape_variance = float(shape_variance)
        self.texture_variance = float(texture_variance)
        self.max_shape_components = int(max_shape_components)
        self.max_texture_components = int(max_texture_components)
        self.chunk_size = int(chunk_size)

        # Learned parameters
        self.mean_shape_: Optional[np.ndarray] = None          # (68, 2) zero-mean unit-norm
        self.ref_canvas_shape_: Optional[np.ndarray] = None    # (68, 2) in [margin, texture_size-margin]
        self.triangles_: Optional[np.ndarray] = None           # (M, 3) int32 vertex indices
        self.shape_pcs_: Optional[np.ndarray] = None           # (136, k_s)
        self.shape_eigvals_: Optional[np.ndarray] = None       # (k_s,)

        # Precomputed piecewise-affine rasterization map on 128x128 reference mesh
        self.mask_: Optional[np.ndarray] = None                # (H, W) bool mask of valid face pixels
        self.pix_tri_idx_: Optional[np.ndarray] = None         # (N_pix,) triangle index per valid pixel
        self.pix_bary_: Optional[np.ndarray] = None            # (N_pix, 3) barycentric coords

        self.mean_texture_: Optional[np.ndarray] = None        # (N_pix * 3,) float32
        self.texture_pcs_: Optional[np.ndarray] = None         # (N_pix * 3, k_g) float32
        self.texture_eigvals_: Optional[np.ndarray] = None     # (k_g,) float32

        # Fitting regression matrix R: maps texture residual -> delta [tx, ty, log_s, theta, b_s...]
        self.R_fit_: Optional[np.ndarray] = None               # (4 + k_s_fit, N_pix)
        self.k_s_fit_: int = 10
        self.gray_mean_: Optional[np.ndarray] = None           # (N_pix,) grayscale mean for fast fitting
        self.gray_pcs_: Optional[np.ndarray] = None            # (N_pix, k_g_fit)
        self.W_kps5_to_bs_: Optional[np.ndarray] = None        # (11, k_s) maps 5-kps + bias -> shape PCA b_s

        # Linear Ridge classifier weights on combined AAM features [b_s, b_g] for baseline recognition
        self.clf_W_: Optional[np.ndarray] = None
        self.clf_b_: Optional[np.ndarray] = None
        self.clf_feat_mean_: Optional[np.ndarray] = None
        self.clf_feat_std_: Optional[np.ndarray] = None

    @staticmethod
    def _align_shape_procrustes(shape: np.ndarray, ref: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
        """Aligns shape (68,2) to ref (68,2) via 2D similarity transform (centring, scale, SVD rotation)."""
        c_s = shape.mean(axis=0)
        c_r = ref.mean(axis=0)
        s0 = shape - c_s
        r0 = ref - c_r
        norm_s = float(np.linalg.norm(s0)) + 1e-9
        norm_r = float(np.linalg.norm(r0)) + 1e-9
        s_unit = s0 / norm_s
        r_unit = r0 / norm_r

        H = s_unit.T @ r_unit
        U, _, Vt = np.linalg.svd(H)
        R = U @ Vt
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = U @ Vt
        aligned = (s_unit @ R) * norm_r
        theta = float(np.arctan2(R[0, 1], R[0, 0]))
        params = {
            "tx": float(c_s[0]),
            "ty": float(c_s[1]),
            "scale": float(norm_s / norm_r),
            "theta": theta,
        }
        return aligned, params

    @staticmethod
    def _shape_to_image(
        aligned_shape: np.ndarray,
        tx: float,
        ty: float,
        scale: float,
        theta: float,
    ) -> np.ndarray:
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        R_inv = np.array([[cos_t, -sin_t], [sin_t, cos_t]], dtype=np.float64)
        return (aligned_shape * scale) @ R_inv + np.array([tx, ty], dtype=np.float64)

    def _fit_gpa(self, shapes: np.ndarray, max_iters: int = 15) -> Tuple[np.ndarray, np.ndarray]:
        """Generalized Procrustes Analysis on (N, 68, 2) shapes."""
        N = shapes.shape[0]
        aligned = np.zeros_like(shapes, dtype=np.float64)
        for i in range(N):
            c = shapes[i].mean(axis=0)
            s0 = shapes[i] - c
            aligned[i] = s0 / (np.linalg.norm(s0) + 1e-9)

        mean_shape = aligned[0].copy()
        for _ in range(max_iters):
            for i in range(N):
                aligned[i], _ = self._align_shape_procrustes(aligned[i], mean_shape)
            new_mean = aligned.mean(axis=0)
            new_mean -= new_mean.mean(axis=0)
            new_mean /= (np.linalg.norm(new_mean) + 1e-9)
            if np.linalg.norm(new_mean - mean_shape) < 1e-7:
                mean_shape = new_mean
                break
            mean_shape = new_mean
        return mean_shape.astype(np.float32), aligned.astype(np.float32)

    def _build_reference_mesh(self, mean_shape: np.ndarray) -> None:
        """Maps mean_shape to a [10, H-10] x [10, W-10] reference canvas, computes Delaunay
        triangulation, and precomputes per-pixel barycentric coordinates for fast warping.
        """
        S = self.texture_size
        margin = 10.0
        min_xy = mean_shape.min(axis=0)
        max_xy = mean_shape.max(axis=0)
        span = np.maximum(max_xy - min_xy, 1e-6)
        scale = (S - 2 * margin) / float(span.max())
        center_ref = 0.5 * (min_xy + max_xy)
        canvas_pts = (mean_shape - center_ref) * scale + (S / 2.0)
        self.ref_canvas_shape_ = canvas_pts.astype(np.float32)

        # Compute Delaunay triangulation using cv2.Subdiv2D
        subdiv = cv2.Subdiv2D((0, 0, S, S))
        for x, y in self.ref_canvas_shape_:
            subdiv.insert((float(x), float(y)))
        tri_list = subdiv.getTriangleList()

        triangles = []
        pts = self.ref_canvas_shape_
        for t in tri_list:
            p0 = np.array([t[0], t[1]])
            p1 = np.array([t[2], t[3]])
            p2 = np.array([t[4], t[5]])
            i0 = int(np.argmin(np.linalg.norm(pts - p0, axis=1)))
            i1 = int(np.argmin(np.linalg.norm(pts - p1, axis=1)))
            i2 = int(np.argmin(np.linalg.norm(pts - p2, axis=1)))
            if len({i0, i1, i2}) == 3:
                tri = tuple(sorted((i0, i1, i2)))
                if tri not in triangles:
                    triangles.append(tri)
        self.triangles_ = np.array(triangles, dtype=np.int32)

        # Precompute pixel-to-triangle index and barycentric coordinates on (S, S) grid
        tri_map = np.full((S, S), -1, dtype=np.int32)
        for t_idx, (i0, i1, i2) in enumerate(self.triangles_):
            poly = np.round(pts[[i0, i1, i2]]).astype(np.int32)
            cv2.fillConvexPoly(tri_map, poly, int(t_idx))

        self.mask_ = tri_map >= 0
        ys, xs = np.where(self.mask_)
        t_indices = tri_map[ys, xs]
        self.pix_tri_idx_ = t_indices

        v0 = pts[self.triangles_[t_indices, 0]]
        v1 = pts[self.triangles_[t_indices, 1]]
        v2 = pts[self.triangles_[t_indices, 2]]
        p_xy = np.stack([xs, ys], axis=1).astype(np.float32)

        # Solve barycentric coordinates: p = w0*v0 + w1*v1 + w2*v2
        d00 = v1[:, 0] - v0[:, 0]
        d01 = v2[:, 0] - v0[:, 0]
        d10 = v1[:, 1] - v0[:, 1]
        d11 = v2[:, 1] - v0[:, 1]
        denom = d00 * d11 - d01 * d10
        denom = np.where(np.abs(denom) < 1e-8, 1e-8, denom)
        px = p_xy[:, 0] - v0[:, 0]
        py = p_xy[:, 1] - v0[:, 1]
        w1 = (px * d11 - d01 * py) / denom
        w2 = (d00 * py - px * d10) / denom
        w0 = 1.0 - w1 - w2
        self.pix_bary_ = np.stack([w0, w1, w2], axis=1).astype(np.float32)

    def warp_to_canonical(self, img: np.ndarray, pts68: np.ndarray) -> np.ndarray:
        """Piecewise-affine warps an image from pts68 (68,2) into the canonical 128x128
        shape-free texture vector (N_pix * 3,) in [0, 1] float32.
        """
        pts = np.asarray(pts68, dtype=np.float32)
        t_idx = self.pix_tri_idx_
        bary = self.pix_bary_
        tri_verts = self.triangles_[t_idx]  # (N_pix, 3)

        src_v0 = pts[tri_verts[:, 0]]
        src_v1 = pts[tri_verts[:, 1]]
        src_v2 = pts[tri_verts[:, 2]]

        src_xy = (
            bary[:, 0:1] * src_v0
            + bary[:, 1:2] * src_v1
            + bary[:, 2:3] * src_v2
        )
        map_x = src_xy[:, 0].reshape(1, -1).astype(np.float32)
        map_y = src_xy[:, 1].reshape(1, -1).astype(np.float32)

        sampled = cv2.remap(
            img,
            map_x,
            map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )[0]  # (N_pix, C)
        tex = sampled.astype(np.float32) / 255.0
        # Photometric zero-mean unit-variance normalization per channel
        mean = tex.mean(axis=0, keepdims=True)
        std = tex.std(axis=0, keepdims=True) + 1e-5
        tex_norm = (tex - mean) / std
        return tex_norm.flatten()

    def vector_to_canvas_image(self, tex_vec: np.ndarray) -> np.ndarray:
        """Converts a canonical texture vector (N_pix * 3,) back into a 128x128x3 uint8 BGR image."""
        S = self.texture_size
        n_pix = int(self.mask_.sum())
        rgb = tex_vec.reshape(n_pix, 3).copy()
        # Rescale normalized texture to standard facial contrast [0, 255]
        rgb = np.clip(rgb * 42.0 + 128.0, 0, 255).astype(np.uint8)
        canvas = np.zeros((S, S, 3), dtype=np.uint8)
        canvas[self.mask_] = rgb
        return canvas

    def fit_model(
        self,
        images: List[np.ndarray],
        shapes: np.ndarray,
        labels: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Trains the Classical AAM from scratch on (N, 68, 2) shapes and N images."""
        shapes = np.asarray(shapes, dtype=np.float32)
        N = shapes.shape[0]

        # 1. Generalized Procrustes Analysis & Shape PCA
        self.mean_shape_, aligned_shapes = self._fit_gpa(shapes)
        self._build_reference_mesh(self.mean_shape_)

        X_s = (aligned_shapes - self.mean_shape_).reshape(N, 136).astype(np.float64)
        U_s, S_s, Vt_s = np.linalg.svd(X_s, full_matrices=False)
        eigvals_s = (S_s ** 2) / max(1, N - 1)
        cumvar_s = np.cumsum(eigvals_s) / (np.sum(eigvals_s) + 1e-12)
        k_s = int(np.searchsorted(cumvar_s, self.shape_variance) + 1)
        k_s = max(4, min(k_s, self.max_shape_components, Vt_s.shape[0]))
        self.shape_pcs_ = Vt_s[:k_s].T.astype(np.float32)      # (136, k_s)
        self.shape_eigvals_ = eigvals_s[:k_s].astype(np.float32)

        # 2. Chunked Piecewise Affine Texture Warping & Appearance PCA (Chunk size 64 for memory efficiency)
        n_pix = int(self.mask_.sum())
        tex_dim = n_pix * 3
        textures = np.zeros((N, tex_dim), dtype=np.float32)
        for start in range(0, N, self.chunk_size):
            end = min(N, start + self.chunk_size)
            for i in range(start, end):
                textures[i] = self.warp_to_canonical(images[i], shapes[i])

        self.mean_texture_ = textures.mean(axis=0)
        X_g = textures - self.mean_texture_
        # Dual PCA via Gram matrix (N x N) since N << tex_dim
        G = (X_g @ X_g.T) / float(max(1, N - 1))
        eigvals_g, V_dual = np.linalg.eigh(G.astype(np.float64))
        order = np.argsort(eigvals_g)[::-1]
        eigvals_g = np.maximum(eigvals_g[order], 0.0)
        V_dual = V_dual[:, order]

        cumvar_g = np.cumsum(eigvals_g) / (np.sum(eigvals_g) + 1e-12)
        k_g = int(np.searchsorted(cumvar_g, self.texture_variance) + 1)
        k_g = max(8, min(k_g, self.max_texture_components, N - 1))

        pcs_g = []
        for j in range(k_g):
            if eigvals_g[j] < 1e-9:
                break
            v = X_g.T @ V_dual[:, j].astype(np.float32)
            v /= (np.linalg.norm(v) + 1e-9)
            pcs_g.append(v)
        self.texture_pcs_ = np.stack(pcs_g, axis=1).astype(np.float32)  # (tex_dim, k_g)
        self.texture_eigvals_ = eigvals_g[: len(pcs_g)].astype(np.float32)

        # 3. Build 5-keypoint-to-shape-PCA pose-aware initializer & Project-Out active fitting regressor
        self._build_kps5_to_shape_initializer(aligned_shapes)
        self._build_active_fitting_regressor(images, shapes)

        # 4. If identity labels are provided, train a Regularized Ridge-SVM classifier on combined AAM features
        if labels is not None:
            feats = self.extract_combined_features(images, shapes)
            self.clf_feat_mean_ = feats.mean(axis=0, keepdims=True)
            self.clf_feat_std_ = feats.std(axis=0, keepdims=True) + 1e-5
            Z = (feats - self.clf_feat_mean_) / self.clf_feat_std_

            classes = sorted(set(int(c) for c in labels))
            Y = np.full((N, len(classes)), -1.0, dtype=np.float32)
            for idx_c, c in enumerate(classes):
                Y[labels == c, idx_c] = 1.0
            reg = 2.0 * np.eye(Z.shape[1], dtype=np.float32)
            self.clf_W_ = np.linalg.solve(Z.T @ Z + reg, Z.T @ Y)
            self.clf_b_ = Y.mean(axis=0)

        return {
            "num_training_samples": N,
            "texture_size": self.texture_size,
            "num_delaunay_triangles": int(len(self.triangles_)),
            "num_canonical_pixels": n_pix,
            "shape_components": int(self.shape_pcs_.shape[1]),
            "shape_variance_explained": round(float(cumvar_s[self.shape_pcs_.shape[1] - 1]), 4),
            "texture_components": int(self.texture_pcs_.shape[1]),
            "texture_variance_explained": round(float(cumvar_g[self.texture_pcs_.shape[1] - 1]), 4),
        }

    def _build_kps5_to_shape_initializer(self, aligned_shapes: np.ndarray) -> None:
        """Learns a pose-aware linear ridge regressor mapping normalized 5 keypoints
        to 68-point shape PCA coefficients b_s (capturing 3D yaw foreshortening of the jawline).
        """
        N = aligned_shapes.shape[0]
        X_s = (aligned_shapes - self.mean_shape_).reshape(N, 136)
        B_s = X_s @ self.shape_pcs_  # (N, k_s)

        # Extract 5 canonical keypoints from each aligned 68-pt shape
        kps5_all = np.zeros((N, 10), dtype=np.float32)
        for i in range(N):
            s = aligned_shapes[i]
            k5 = np.stack([s[36:42].mean(0), s[42:48].mean(0), s[30], s[48], s[54]], axis=0)
            kps5_all[i] = k5.flatten()

        X_k = np.hstack([kps5_all, np.ones((N, 1), dtype=np.float32)])  # (N, 11)
        reg = 1e-3 * np.eye(11, dtype=np.float32)
        reg[-1, -1] = 0.0
        self.W_kps5_to_bs_ = np.linalg.solve(X_k.T @ X_k + reg, X_k.T @ B_s).astype(np.float32)

    def init_shape_from_5kps(self, kps5: np.ndarray) -> np.ndarray:
        """Predicts an initial 68-point AAM shape in image coordinates from 5 detected keypoints
        using similarity alignment + learned pose-aware PCA shape coefficients.
        """
        kps5 = np.asarray(kps5, dtype=np.float64).reshape(5, 2)
        ms = self.mean_shape_.astype(np.float64)
        ref_k5 = np.stack([ms[36:42].mean(0), ms[42:48].mean(0), ms[30], ms[48], ms[54]], axis=0)

        c_s = kps5.mean(axis=0)
        c_r = ref_k5.mean(axis=0)
        s0 = kps5 - c_s
        r0 = ref_k5 - c_r
        norm_s = float(np.linalg.norm(s0)) + 1e-9
        norm_r = float(np.linalg.norm(r0)) + 1e-9
        s_unit = s0 / norm_s
        r0_unit = r0 / norm_r

        H = s_unit.T @ r0_unit
        U, _, Vt = np.linalg.svd(H)
        R = U @ Vt
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = U @ Vt

        k5_aligned = (s_unit @ R) * norm_r
        if self.W_kps5_to_bs_ is not None:
            xk = np.append(k5_aligned.flatten(), 1.0).astype(np.float32)
            bs_pred = xk @ self.W_kps5_to_bs_
            std_bs = np.sqrt(np.maximum(self.shape_eigvals_, 1e-6))
            bs_pred = np.clip(bs_pred, -2.5 * std_bs, 2.5 * std_bs)
            s_local = self.mean_shape_ + (self.shape_pcs_ @ bs_pred).reshape(68, 2)
        else:
            s_local = self.mean_shape_

        scale = norm_s / norm_r
        s_img = ((s_local.astype(np.float64) - c_r) * scale) @ R.T + c_s
        return s_img.astype(np.float32)

    def _warp_gray_to_canonical(self, img: np.ndarray, pts68: np.ndarray) -> np.ndarray:
        tex3 = self.warp_to_canonical(img, pts68).reshape(-1, 3)
        g = tex3.mean(axis=1)
        return (g - g.mean()) / (g.std() + 1e-5)

    def _build_active_fitting_regressor(self, images: List[np.ndarray], shapes: np.ndarray) -> None:
        """Learns the Project-Out linear regression update matrix R mapping canonical grayscale
        appearance residual delta_g to shape parameter update delta_p = [d_tx, d_ty, d_scale, d_theta, d_bs].
        """
        N = len(images)
        self.k_s_fit_ = min(10, self.shape_pcs_.shape[1])
        P_s = self.shape_pcs_[:, : self.k_s_fit_]  # (136, k_s_fit)

        # Compute grayscale canonical mean and top-12 appearance basis for Project-Out
        gray_texs = np.stack([self._warp_gray_to_canonical(images[i], shapes[i]) for i in range(N)], axis=0)
        self.gray_mean_ = gray_texs.mean(axis=0)
        Xg = gray_texs - self.gray_mean_
        _, _, Vtg = np.linalg.svd(Xg, full_matrices=False)
        self.gray_pcs_ = Vtg[: min(12, Vtg.shape[0])].T.astype(np.float32)

        rng = np.random.default_rng(1234)
        # Subsample up to 120 images x 4 perturbations = 480 perturbation pairs
        sample_indices = rng.choice(N, size=min(N, 120), replace=False)
        dP_list = []
        dG_list = []

        std_bs = np.sqrt(np.maximum(self.shape_eigvals_[: self.k_s_fit_], 1e-6))

        for i in sample_indices:
            gt_s = shapes[i].astype(np.float64)
            aligned_s, pose = self._align_shape_procrustes(gt_s, self.mean_shape_.astype(np.float64))
            bs_true = ((aligned_s - self.mean_shape_).reshape(136) @ P_s).astype(np.float32)
            scale0 = pose["scale"]

            for _ in range(4):
                d_tx = float(rng.uniform(-0.07, 0.07) * scale0)
                d_ty = float(rng.uniform(-0.07, 0.07) * scale0)
                d_sc = float(rng.uniform(-0.07, 0.07))
                d_th = float(rng.uniform(-0.08, 0.08))
                d_bs = rng.uniform(-0.45, 0.45, size=self.k_s_fit_).astype(np.float32) * std_bs

                bs_pert = bs_true + d_bs
                s_local = self.mean_shape_ + (P_s @ bs_pert).reshape(68, 2)
                s_pert = self._shape_to_image(
                    s_local,
                    pose["tx"] + d_tx,
                    pose["ty"] + d_ty,
                    scale0 * (1.0 + d_sc),
                    pose["theta"] + d_th,
                )

                g_pert = self._warp_gray_to_canonical(images[i], s_pert)
                res_g = g_pert - self.gray_mean_
                # Project out principal appearance variation
                res_g = res_g - self.gray_pcs_ @ (self.gray_pcs_.T @ res_g)

                dp = np.concatenate([
                    [d_tx / scale0, d_ty / scale0, d_sc, d_th],
                    d_bs,
                ]).astype(np.float32)
                dP_list.append(dp)
                dG_list.append(res_g)

        dP = np.stack(dP_list, axis=0)  # (M, 4 + k_s_fit)
        dG = np.stack(dG_list, axis=0)  # (M, N_pix)
        # Ridge regression: R = (dP^T dG) (dG^T dG + lambda I)^-1 via dual form
        lam = 25.0
        gram = dG @ dG.T + lam * np.eye(dG.shape[0], dtype=np.float32)
        self.R_fit_ = (dP.T @ np.linalg.solve(gram, dG)).astype(np.float32)

    def reconstruct_shape(self, pts68: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Projects pts68 onto the AAM PCA Shape Model and reconstructs it in image coordinates."""
        pts = np.asarray(pts68, dtype=np.float64)
        aligned, pose = self._align_shape_procrustes(pts, self.mean_shape_.astype(np.float64))
        diff = (aligned - self.mean_shape_).reshape(136)
        b_s = (self.shape_pcs_.T @ diff).astype(np.float32)
        recon_local = self.mean_shape_ + (self.shape_pcs_ @ b_s).reshape(68, 2)
        recon_img = self._shape_to_image(
            recon_local,
            pose["tx"],
            pose["ty"],
            pose["scale"],
            pose["theta"],
        )
        return recon_img.astype(np.float32), b_s

    def reconstruct_texture(self, img: np.ndarray, pts68: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """Projects the shape-free canonical texture onto the AAM PCA Appearance Model
        and returns (reconstructed_texture_vec, b_g, mse).
        """
        g = self.warp_to_canonical(img, pts68)
        diff = g - self.mean_texture_
        b_g = (self.texture_pcs_.T @ diff).astype(np.float32)
        g_recon = self.mean_texture_ + (self.texture_pcs_ @ b_g)
        mse = float(np.mean((g - g_recon) ** 2))
        return g_recon, b_g, mse

    def fit_image(
        self,
        img: np.ndarray,
        init_pts68: np.ndarray,
        num_iters: int = 12,
    ) -> Tuple[np.ndarray, List[float]]:
        """Actively fits the AAM to an image starting from an initial 68-point shape estimate."""
        P_s = self.shape_pcs_[:, : self.k_s_fit_]
        std_bs = np.sqrt(np.maximum(self.shape_eigvals_[: self.k_s_fit_], 1e-6))

        aligned, pose = self._align_shape_procrustes(
            np.asarray(init_pts68, dtype=np.float64),
            self.mean_shape_.astype(np.float64),
        )
        tx, ty, scale, theta = pose["tx"], pose["ty"], pose["scale"], pose["theta"]
        b_s = ((aligned - self.mean_shape_).reshape(136) @ P_s).astype(np.float32)
        b_s = np.clip(b_s, -2.5 * std_bs, 2.5 * std_bs)

        cur_pts = self._shape_to_image(
            self.mean_shape_ + (P_s @ b_s).reshape(68, 2),
            tx, ty, scale, theta,
        )
        g_cur = self._warp_gray_to_canonical(img, cur_pts)
        res = g_cur - self.gray_mean_
        res_proj = res - self.gray_pcs_ @ (self.gray_pcs_.T @ res)
        cur_err = float(np.mean(res_proj ** 2))
        err_history = [cur_err]

        for _ in range(num_iters):
            dp = self.R_fit_ @ res_proj
            improved = False
            for step in (0.7, 0.35, 0.15):
                n_tx = tx - step * float(dp[0]) * scale
                n_ty = ty - step * float(dp[1]) * scale
                n_sc = scale * (1.0 - step * float(dp[2]))
                n_th = theta - step * float(dp[3])
                n_bs = np.clip(b_s - step * dp[4:], -3.0 * std_bs, 3.0 * std_bs)

                cand_pts = self._shape_to_image(
                    self.mean_shape_ + (P_s @ n_bs).reshape(68, 2),
                    n_tx, n_ty, n_sc, n_th,
                )
                g_cand = self._warp_gray_to_canonical(img, cand_pts)
                r_cand = g_cand - self.gray_mean_
                r_cand_proj = r_cand - self.gray_pcs_ @ (self.gray_pcs_.T @ r_cand)
                cand_err = float(np.mean(r_cand_proj ** 2))
                if cand_err < cur_err:
                    tx, ty, scale, theta, b_s = n_tx, n_ty, n_sc, n_th, n_bs
                    cur_pts = cand_pts
                    res_proj = r_cand_proj
                    cur_err = cand_err
                    improved = True
                    break
            err_history.append(cur_err)
            if not improved:
                break

        return cur_pts.astype(np.float32), err_history

    def extract_combined_features(self, images: List[np.ndarray], shapes: np.ndarray) -> np.ndarray:
        """Extracts concatenated [b_s, b_g] Classical AAM feature vectors for N samples."""
        feats = []
        for i in range(len(images)):
            _, b_s = self.reconstruct_shape(shapes[i])
            _, b_g, _ = self.reconstruct_texture(images[i], shapes[i])
            feats.append(np.concatenate([b_s, b_g]))
        return np.stack(feats, axis=0).astype(np.float32)

    def predict_identity_baseline(self, images: List[np.ndarray], shapes: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predicts identity IDs (0..3) and confidence scores using Classical AAM features + Ridge-SVM."""
        feats = self.extract_combined_features(images, shapes)
        Z = (feats - self.clf_feat_mean_) / self.clf_feat_std_
        logits = Z @ self.clf_W_ + self.clf_b_
        preds = np.argmax(logits, axis=1).astype(np.int32)
        scores = np.max(logits, axis=1).astype(np.float32)
        return preds, scores

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        np.savez_compressed(
            path,
            texture_size=np.array(self.texture_size, dtype=np.int32),
            mean_shape=self.mean_shape_,
            ref_canvas_shape=self.ref_canvas_shape_,
            triangles=self.triangles_,
            shape_pcs=self.shape_pcs_,
            shape_eigvals=self.shape_eigvals_,
            mask=self.mask_,
            pix_tri_idx=self.pix_tri_idx_,
            pix_bary=self.pix_bary_,
            mean_texture=self.mean_texture_,
            texture_pcs=self.texture_pcs_,
            texture_eigvals=self.texture_eigvals_,
            R_fit=self.R_fit_,
            k_s_fit=np.array(self.k_s_fit_, dtype=np.int32),
            gray_mean=self.gray_mean_,
            gray_pcs=self.gray_pcs_,
            W_kps5_to_bs=self.W_kps5_to_bs_ if self.W_kps5_to_bs_ is not None else np.zeros((1, 1), dtype=np.float32),
            clf_W=self.clf_W_ if self.clf_W_ is not None else np.zeros((1, 1), dtype=np.float32),
            clf_b=self.clf_b_ if self.clf_b_ is not None else np.zeros((1,), dtype=np.float32),
            clf_feat_mean=self.clf_feat_mean_ if self.clf_feat_mean_ is not None else np.zeros((1, 1), dtype=np.float32),
            clf_feat_std=self.clf_feat_std_ if self.clf_feat_std_ is not None else np.ones((1, 1), dtype=np.float32),
        )

    @classmethod
    def load(cls, path: str) -> "ClassicalAAM":
        d = np.load(path, allow_pickle=False)
        aam = cls(texture_size=int(d["texture_size"]))
        aam.mean_shape_ = d["mean_shape"]
        aam.ref_canvas_shape_ = d["ref_canvas_shape"]
        aam.triangles_ = d["triangles"]
        aam.shape_pcs_ = d["shape_pcs"]
        aam.shape_eigvals_ = d["shape_eigvals"]
        aam.mask_ = d["mask"]
        aam.pix_tri_idx_ = d["pix_tri_idx"]
        aam.pix_bary_ = d["pix_bary"]
        aam.mean_texture_ = d["mean_texture"]
        aam.texture_pcs_ = d["texture_pcs"]
        aam.texture_eigvals_ = d["texture_eigvals"]
        aam.R_fit_ = d["R_fit"]
        aam.k_s_fit_ = int(d["k_s_fit"])
        aam.gray_mean_ = d["gray_mean"]
        aam.gray_pcs_ = d["gray_pcs"]
        if "W_kps5_to_bs" in d and d["W_kps5_to_bs"].shape[0] == 11:
            aam.W_kps5_to_bs_ = d["W_kps5_to_bs"]
        aam.clf_W_ = d["clf_W"]
        aam.clf_b_ = d["clf_b"]
        aam.clf_feat_mean_ = d["clf_feat_mean"]
        aam.clf_feat_std_ = d["clf_feat_std"]
        return aam
