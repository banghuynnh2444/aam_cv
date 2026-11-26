"""SCRFD-2.5G-KPS Face Detector & 5-Keypoint ArcFace Alignment Module.

Implements:
- SCRFD-2.5G-KPS ONNX inference (bounding box + 5 keypoints + confidence score)
- Exact Umeyama 2D similarity transform for 112x112 ArcFace face alignment
"""
import os
from typing import List, Dict, Any, Tuple, Optional
import cv2
import numpy as np
import onnxruntime as ort
from src.paths import project_path


DEFAULT_SCRFD_PATH = project_path("checkpoints", "scrfd_2.5g_bnkps.onnx")

# Standard ArcFace 112x112 5-point reference coordinates
ARCFACE_REF_5PTS = np.array([
    [38.2946, 51.6963],  # Viewer-left eye
    [73.5318, 51.5014],  # Viewer-right eye
    [56.0252, 71.7366],  # Nose tip
    [41.5493, 92.3655],  # Viewer-left mouth corner
    [70.7299, 92.2041],  # Viewer-right mouth corner
], dtype=np.float32)


def umeyama_similarity_transform(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Computes the 2x3 2D similarity transform matrix M mapping src (N,2) to dst (N,2)
    using Umeyama's closed-form SVD method (identical to InsightFace norm_crop).
    """
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    num, dim = src.shape

    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    src_demean = src - src_mean
    dst_demean = dst - dst_mean

    A = (dst_demean.T @ src_demean) / num
    d = np.ones((dim,), dtype=np.float64)
    if np.linalg.det(A) < 0:
        d[dim - 1] = -1

    T = np.eye(dim + 1, dtype=np.float64)
    U, S, Vt = np.linalg.svd(A)
    rank = np.linalg.matrix_rank(A)
    if rank == 0:
        return np.nan * T[:dim, :]
    elif rank == dim - 1:
        if np.linalg.det(U) * np.linalg.det(Vt) > 0:
            T[:dim, :dim] = U @ Vt
        else:
            s = d[dim - 1]
            d[dim - 1] = -1
            T[:dim, :dim] = U @ np.diag(d) @ Vt
            d[dim - 1] = s
    else:
        T[:dim, :dim] = U @ np.diag(d) @ Vt

    var_src = src_demean.var(axis=0).sum()
    scale = 1.0 / var_src * (S @ d) if var_src > 1e-12 else 1.0
    T[:dim, dim] = dst_mean - scale * (T[:dim, :dim] @ src_mean.T)
    T[:dim, :dim] *= scale
    return T[:dim, :].astype(np.float32)


def align_face_arcface(
    img: np.ndarray,
    kps5: np.ndarray,
    image_size: int = 112,
) -> Tuple[np.ndarray, np.ndarray]:
    """Warps a face image to 112x112 canonical ArcFace coordinates using 5 keypoints.

    Returns:
        aligned_img: (image_size, image_size, 3) uint8 BGR image
        M: (2, 3) float32 affine similarity transformation matrix
    """
    ref = ARCFACE_REF_5PTS.copy()
    if image_size != 112:
        ref *= (float(image_size) / 112.0)
    M = umeyama_similarity_transform(kps5.reshape(5, 2), ref)
    aligned = cv2.warpAffine(img, M, (image_size, image_size), borderValue=0.0)
    return aligned, M


def _nms(boxes: np.ndarray, scores: np.ndarray, iou_thresh: float) -> List[int]:
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        inds = np.where(ovr <= iou_thresh)[0]
        order = order[inds + 1]
    return keep


class SCRFDDetector:
    """SCRFD-2.5G-KPS ONNX Face & 5-Keypoint Detector."""

    def __init__(
        self,
        model_path: str = DEFAULT_SCRFD_PATH,
        input_size: Tuple[int, int] = (640, 640),
        conf_thresh: float = 0.50,
        nms_thresh: float = 0.40,
        use_cuda: bool = True,
    ):
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"SCRFD ONNX model not found at {model_path}")
        self.model_path = model_path
        self.input_size = input_size
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.strides = [8, 16, 32]
        self.num_anchors = 2
        self._center_cache = {}

        avail = ort.get_available_providers()
        providers = []
        if use_cuda and "CUDAExecutionProvider" in avail:
            providers.append("CUDAExecutionProvider")
        providers.append("CPUExecutionProvider")

        self.session = ort.InferenceSession(model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]

    def _get_anchor_centers(self, feat_h: int, feat_w: int, stride: int) -> np.ndarray:
        key = (feat_h, feat_w, stride)
        if key in self._center_cache:
            return self._center_cache[key]
        anchor_centers = np.stack(np.mgrid[:feat_h, :feat_w][::-1], axis=-1).astype(np.float32)
        anchor_centers = (anchor_centers * stride).reshape((-1, 2))
        if self.num_anchors > 1:
            anchor_centers = np.stack([anchor_centers] * self.num_anchors, axis=1).reshape((-1, 2))
        self._center_cache[key] = anchor_centers
        return anchor_centers

    def _detect_raw(self, img: np.ndarray, conf_thresh: Optional[float] = None) -> List[Dict[str, Any]]:
        thresh = self.conf_thresh if conf_thresh is None else conf_thresh
        in_w, in_h = self.input_size
        im_h, im_w = img.shape[:2]

        im_ratio = float(im_h) / float(im_w)
        model_ratio = float(in_h) / float(in_w)
        if im_ratio > model_ratio:
            new_h = in_h
            new_w = int(round(new_h / im_ratio))
        else:
            new_w = in_w
            new_h = int(round(new_w * im_ratio))
        det_scale = float(new_h) / float(im_h)

        resized = cv2.resize(img, (new_w, new_h))
        det_img = np.zeros((in_h, in_w, 3), dtype=np.uint8)
        det_img[:new_h, :new_w, :] = resized

        blob = cv2.dnn.blobFromImage(
            det_img,
            scalefactor=1.0 / 128.0,
            size=(in_w, in_h),
            mean=(127.5, 127.5, 127.5),
            swapRB=True,
        )
        net_outs = self.session.run(self.output_names, {self.input_name: blob})

        scores_list = []
        bboxes_list = []
        kps_list = []
        fmc = len(self.strides)

        for idx, stride in enumerate(self.strides):
            scores = net_outs[idx].reshape(-1)
            bbox_preds = net_outs[idx + fmc] * stride
            kps_preds = net_outs[idx + fmc * 2] * stride

            feat_h = in_h // stride
            feat_w = in_w // stride
            anchor_centers = self._get_anchor_centers(feat_h, feat_w, stride)

            pos_inds = np.where(scores >= thresh)[0]
            if pos_inds.size == 0:
                continue

            pos_scores = scores[pos_inds]
            pos_bbox_preds = bbox_preds[pos_inds]
            pos_centers = anchor_centers[pos_inds]

            x1 = pos_centers[:, 0] - pos_bbox_preds[:, 0]
            y1 = pos_centers[:, 1] - pos_bbox_preds[:, 1]
            x2 = pos_centers[:, 0] + pos_bbox_preds[:, 2]
            y2 = pos_centers[:, 1] + pos_bbox_preds[:, 3]
            bboxes = np.stack([x1, y1, x2, y2], axis=-1)

            pos_kps_preds = kps_preds[pos_inds]
            kps = np.zeros_like(pos_kps_preds)
            for k in range(0, 10, 2):
                kps[:, k] = pos_centers[:, 0] + pos_kps_preds[:, k]
                kps[:, k + 1] = pos_centers[:, 1] + pos_kps_preds[:, k + 1]

            scores_list.append(pos_scores)
            bboxes_list.append(bboxes)
            kps_list.append(kps)

        if not scores_list:
            return []

        all_scores = np.concatenate(scores_list, axis=0)
        all_bboxes = np.concatenate(bboxes_list, axis=0) / det_scale
        all_kps = np.concatenate(kps_list, axis=0) / det_scale

        keep = _nms(all_bboxes, all_scores, self.nms_thresh)
        results = []
        for i in keep:
            b = all_bboxes[i].copy()
            b[0] = np.clip(b[0], 0, im_w)
            b[1] = np.clip(b[1], 0, im_h)
            b[2] = np.clip(b[2], 0, im_w)
            b[3] = np.clip(b[3], 0, im_h)
            k = all_kps[i].reshape(5, 2)
            area = float(max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1]))
            results.append({
                "bbox": b.astype(np.float32),       # [x1, y1, x2, y2]
                "kps5": k.astype(np.float32),       # (5, 2)
                "score": float(all_scores[i]),
                "area": area,
            })
        # Sort by face box area descending
        results.sort(key=lambda d: d["area"], reverse=True)
        return results

    def detect(
        self,
        img: np.ndarray,
        is_cropped_face: bool = False,
        conf_thresh: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Detects faces in an image.
        For tightly-cropped single-face images, automatically applies border padding
        so SCRFD has sufficient context around the head boundary.
        """
        h, w = img.shape[:2]
        if not is_cropped_face:
            return self._detect_raw(img, conf_thresh=conf_thresh)

        # For cropped face images, pad 35% border first for reliable context
        pad_x, pad_y = int(round(w * 0.35)), int(round(h * 0.35))
        padded = cv2.copyMakeBorder(img, pad_y, pad_y, pad_x, pad_x, cv2.BORDER_REPLICATE)
        dets = self._detect_raw(padded, conf_thresh=conf_thresh)
        if not dets:
            return self._detect_raw(img, conf_thresh=conf_thresh)

        mapped = []
        for d in dets:
            b = d["bbox"].copy()
            b[0] = np.clip(b[0] - pad_x, 0, w)
            b[1] = np.clip(b[1] - pad_y, 0, h)
            b[2] = np.clip(b[2] - pad_x, 0, w)
            b[3] = np.clip(b[3] - pad_y, 0, h)
            k = d["kps5"].copy()
            k[:, 0] -= pad_x
            k[:, 1] -= pad_y
            area = float(max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1]))
            mapped.append({
                "bbox": b,
                "kps5": k,
                "score": d["score"],
                "area": area,
            })
        mapped.sort(key=lambda d: d["area"], reverse=True)
        return mapped
