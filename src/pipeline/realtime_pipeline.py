"""Phase 7 - Unified Real-Time Hybrid AAM + Deep Learning Face Analysis Pipeline
with Full Vietnamese UI Overlay & 10-Second Locked Age/Gender Stabilization.

Integrates:
1. Low-light / backlit adaptive CLAHE preprocessing when frame brightness < 65.
2. Multi-face detection & 5-keypoint localization via SCRFD-2.5G-KPS.
3. Classical AAM 68-landmark active fitting (initialized from 5-keypoint similarity transform
   of the learned Procrustes mean shape) + 3D Delaunay triangular mesh overlay.
4. 3D Head Pose Estimation (Yaw, Pitch, Roll in degrees via EPnP/Iterative PnP).
5. ArcFace ResNet50 512-D embedding extraction & thresholded recognition (Bằng, Lập, Thảo, Thuận, or Người lạ).
6. Multi-face temporal tracking via ByteTrack + Face Embedding Re-ID.
7. 7-Class Facial Expression analysis (DAN fine-tuned on RAF-DB) with EMA smoothing & Vietnamese labels.
8. Multi-Task EfficientNet-B0 (fine-tuned on FairFace) for Age & Gender with a 10-SECOND LOCK
   per tracked face ID so Age & Gender stay rock-solid on live webcam without jumping!
"""
import os
import time
from typing import Dict, Any, List, Optional, Tuple
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from src.paths import project_path

from src.detection.scrfd import SCRFDDetector, align_face_arcface
from src.aam.classical_aam import ClassicalAAM
from src.aam.cascade_sdm import CascadeSDM
from src.geometry.head_pose_3d import estimate_head_pose_68, draw_3d_face_mesh_and_pose
from src.recognition.arcface import ArcFaceEmbedder, RecognitionDatabase
from src.tracking.bytetrack_face import ByteTrackFaceTracker, TrackedFace, bbox_iou
from src.emotion.emotion_dan import EMOTION_7_CLASSES, DANEmotionNet, analyze_expression_from_landmarks68
from src.attributes.age_gender_effnet import AGE_GROUP_CLASSES, GENDER_LABEL_CLASSES, MultiTaskEfficientNetB0


VIETNAMESE_NAME_MAP: Dict[str, str] = {
    "Bang": "Bằng",
    "Lap": "Lập",
    "Thao": "Thảo",
    "Thuan": "Thuận",
    "Unknown": "Người lạ",
}

VIETNAMESE_EMOTION_MAP: Dict[str, str] = {
    "Neutral": "Bình thường",
    "Happy": "Vui vẻ",
    "Sad": "Buồn",
    "Surprise": "Ngạc nhiên",
    "Fear": "Sợ hãi",
    "Disgust": "Ghê tởm",
    "Anger": "Tức giận",
}

VIETNAMESE_GENDER_MAP: Dict[str, str] = {
    "Female_Label": "Nữ",
    "Male_Label": "Nam",
}

# Representative center age (in years) for each of the 6 age brackets:
# ("0-9", "10-19", "20-29", "30-39", "40-49", "50+") - FairFace bins
AGE_BRACKET_CENTERS = np.array([5.0, 15.0, 25.0, 35.0, 45.0, 60.0], dtype=np.float32)


def _load_vietnamese_font(size: int = 16) -> ImageFont.FreeTypeFont:
    candidates = [
        r"C:\Windows\Fonts\arialbd.ttf",
        r"C:\Windows\Fonts\segoeui.ttf",
        r"C:\Windows\Fonts\tahomabd.ttf",
        r"C:\Windows\Fonts\arial.ttf",
    ]
    for path in candidates:
        if os.path.isfile(path):
            try:
                return ImageFont.truetype(path, size=size)
            except Exception:
                continue
    return ImageFont.load_default()


class HybridFaceAnalysisPipeline:
    """End-to-End Real-Time Face Analysis Pipeline with Vietnamese Output & 10s Age/Gender Lock."""

    def __init__(
        self,
        scrfd_path: str = project_path("checkpoints", "scrfd_2.5g_bnkps.onnx"),
        arcface_path: str = project_path("checkpoints", "w600k_r50.onnx"),
        aam_path: str = project_path("checkpoints", "classical_aam_model.npz"),
        sdm_path: str = project_path("checkpoints", "aam_cascade_sdm_v2.npz"),
        sdm_stages: int = 3,
        embeddings_path: str = project_path("identity_embeddings.npz"),
        dan_path: str = project_path("checkpoints", "dan_rafdb_best.pth"),
        effnet_path: str = project_path("checkpoints", "effnet_b0_fairface_v3.pth"),
        use_cuda: bool = True,
        aam_fit_iters: int = 8,
        reid_interval: int = 10,
        attr_refresh_interval_sec: float = 10.0,
    ):
        self.detector = SCRFDDetector(model_path=scrfd_path, conf_thresh=0.45, use_cuda=use_cuda)
        self.embedder = ArcFaceEmbedder(model_path=arcface_path, use_cuda=use_cuda)
        self.db = RecognitionDatabase.load(embeddings_path)
        self.aam = ClassicalAAM.load(aam_path) if os.path.isfile(aam_path) else None
        # v2: HOG cascaded shape regressor (3 stages, chosen on validation) replaces Project-Out fitting
        self.sdm = CascadeSDM.load(sdm_path) if os.path.isfile(sdm_path) else None
        self.sdm_stages = int(sdm_stages)
        self.tracker = ByteTrackFaceTracker()
        self.aam_fit_iters = int(aam_fit_iters)
        self.reid_interval = int(reid_interval)
        self.attr_refresh_interval_sec = float(attr_refresh_interval_sec)
        self._frame_counter = 0
        self._clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))

        self._font_title = _load_vietnamese_font(size=16)
        self._font_sub = _load_vietnamese_font(size=14)
        self._font_title_sm = _load_vietnamese_font(size=13)
        self._font_sub_sm = _load_vietnamese_font(size=12)
        self._font_hud = _load_vietnamese_font(size=17)

        self.device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")
        self._mean_t = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        self._std_t = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)

        self.dan_model = None
        if os.path.isfile(dan_path):
            self.dan_model = DANEmotionNet(num_classes=7, num_heads=4, feat_dim=256, pretrained_backbone=False).to(self.device)
            ckpt = torch.load(dan_path, map_location=self.device, weights_only=False)
            self.dan_model.load_state_dict(ckpt["model_state_dict"])
            self.dan_model.eval()

        self.effnet_model = None
        if os.path.isfile(effnet_path):
            self.effnet_model = MultiTaskEfficientNetB0(pretrained_backbone=False).to(self.device)
            ckpt = torch.load(effnet_path, map_location=self.device, weights_only=False)
            self.effnet_model.load_state_dict(ckpt["model_state_dict"])
            self.effnet_model.eval()

    def preprocess_lighting(self, bgr: np.ndarray) -> Tuple[np.ndarray, str]:
        """Applies adaptive CLAHE luminance enhancement if the image is dim (<65) or backlit."""
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        mean_val = float(gray.mean())
        if mean_val < 65.0:
            lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            l_enh = self._clahe.apply(l)
            enh = cv2.cvtColor(cv2.merge((l_enh, a, b)), cv2.COLOR_LAB2BGR)
            return enh, f"Thiếu sáng - Đã tăng cường CLAHE ({mean_val:.0f})"
        return bgr, f"Bình thường ({mean_val:.0f})"

    def init_aam_shape_from_5kps(self, kps5: np.ndarray) -> np.ndarray:
        if self.aam is None or self.aam.mean_shape_ is None:
            raise RuntimeError("Classical AAM model is not loaded.")
        return self.aam.init_shape_from_5kps(kps5)

    def _crop_face_224_with_context(self, img_bgr: np.ndarray, bbox: np.ndarray, margin: float = 0.25) -> np.ndarray:
        """Crops a 224x224 face region with 25% head/hair context margin around bbox
        (matching FairFace padding_025 and RAF-DB framing, preserving hair & head contour).
        """
        h, w = img_bgr.shape[:2]
        x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
        bw, bh = max(1.0, x2 - x1), max(1.0, y2 - y1)
        side = max(bw, bh) * (1.0 + 2.0 * margin)
        cx, cy = 0.5 * (x1 + x2), 0.5 * (y1 + y2)

        ix0 = int(round(cx - side / 2.0))
        iy0 = int(round(cy - side / 2.0))
        ix1 = int(round(cx + side / 2.0))
        iy1 = int(round(cy + side / 2.0))

        pad = max(0, -ix0, -iy0, ix1 - w, iy1 - h)
        if pad > 0:
            padded = cv2.copyMakeBorder(img_bgr, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
            ix0 += pad
            iy0 += pad
            ix1 += pad
            iy1 += pad
            crop = padded[iy0:iy1, ix0:ix1]
        else:
            crop = img_bgr[iy0:iy1, ix0:ix1]

        if crop.size == 0:
            return cv2.resize(img_bgr, (224, 224), interpolation=cv2.INTER_AREA)
        return cv2.resize(crop, (224, 224), interpolation=cv2.INTER_AREA)

    def _prepare_tensor_224(self, face_bgr_224: np.ndarray) -> torch.Tensor:
        if face_bgr_224.shape[:2] != (224, 224):
            face_bgr_224 = cv2.resize(face_bgr_224, (224, 224), interpolation=cv2.INTER_CUBIC)
        rgb = cv2.cvtColor(face_bgr_224, cv2.COLOR_BGR2RGB)
        x = torch.from_numpy(rgb).to(self.device).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        return (x - self._mean_t) / self._std_t

    def _infer_emotion(
        self,
        aligned_bgr_112: np.ndarray,
        pts68: Optional[np.ndarray],
        matched_trk: Optional[TrackedFace] = None,
    ) -> Dict[str, Any]:
        """Runs DAN 7-Emotion model with EMA probability smoothing and Vietnamese output."""
        if self.dan_model is None:
            expr_au = analyze_expression_from_landmarks68(pts68) if pts68 is not None else {
                "predicted_expression": "Neutral",
                "confidence": 0.5,
            }
            en_lbl = expr_au["predicted_expression"]
            expr_au["predicted_expression_vn"] = VIETNAMESE_EMOTION_MAP.get(en_lbl, en_lbl)
            return expr_au

        x = self._prepare_tensor_224(aligned_bgr_112)
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.float16, enabled=(self.device.type == "cuda")):
            logits, _ = self.dan_model(x)
            probs = F.softmax(logits, dim=1)[0].detach().cpu().numpy().astype(np.float32)

        if matched_trk is not None:
            if matched_trk.ema_emotion_probs is None:
                matched_trk.ema_emotion_probs = probs
            else:
                matched_trk.ema_emotion_probs = 0.45 * probs + 0.55 * matched_trk.ema_emotion_probs
            probs = matched_trk.ema_emotion_probs

        c_idx = int(np.argmax(probs))
        en_lbl = EMOTION_7_CLASSES[c_idx]
        vn_lbl = VIETNAMESE_EMOTION_MAP.get(en_lbl, en_lbl)
        return {
            "predicted_expression": en_lbl,
            "predicted_expression_vn": vn_lbl,
            "confidence": round(float(probs[c_idx]), 4),
            "model_source": "DAN_RAFDB_40Epochs",
        }

    def _infer_age_gender_with_10s_lock(
        self,
        face_ctx_224: np.ndarray,
        matched_trk: Optional[TrackedFace],
        now_sec: float,
        use_tracking: bool,
        aligned_bgr_112: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Calculates Age (exact estimated years + age group) and Gender in Vietnamese.
        When tracking on webcam/video, locks Age & Gender for 10.0 seconds per track_id
        so numbers never flicker or jump between frames!
        """
        if self.effnet_model is None:
            return {
                "age_years": 21,
                "age_group": "20-29",
                "age_display_vn": "25 tuổi (20-29)",
                "gender_label": "Male_Label",
                "gender_vn": "Nam",
                "locked_remaining_sec": 0.0,
            }

        # Check if locked attributes are still valid within the 10-second window
        if (
            use_tracking
            and matched_trk is not None
            and matched_trk.locked_attributes is not None
            and (now_sec - matched_trk.last_attr_update_time) < self.attr_refresh_interval_sec
        ):
            locked = matched_trk.locked_attributes.copy()
            rem = max(0.0, self.attr_refresh_interval_sec - (now_sec - matched_trk.last_attr_update_time))
            locked["locked_remaining_sec"] = round(rem, 1)
            return locked

        # Otherwise (new face track OR >= 10.0 seconds elapsed), run Multi-Crop + Flip TTA inference
        x1 = self._prepare_tensor_224(face_ctx_224)
        x1_f = torch.flip(x1, dims=[3])
        if aligned_bgr_112 is not None:
            x2 = self._prepare_tensor_224(aligned_bgr_112)
            x2_f = torch.flip(x2, dims=[3])
            x_batch = torch.cat([x1, x1_f, x2, x2_f], dim=0)
        else:
            x_batch = torch.cat([x1, x1_f], dim=0)

        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.float16, enabled=(self.device.type == "cuda")):
            a_logits, g_logits = self.effnet_model(x_batch)
            a_probs = F.softmax(a_logits, dim=1).mean(dim=0).detach().cpu().numpy().astype(np.float32)
            g_probs = F.softmax(g_logits, dim=1).mean(dim=0).detach().cpu().numpy().astype(np.float32)

        if matched_trk is not None:
            if matched_trk.ema_age_probs is None:
                matched_trk.ema_age_probs = a_probs
                matched_trk.ema_gender_probs = g_probs
            else:
                matched_trk.ema_age_probs = 0.50 * a_probs + 0.50 * matched_trk.ema_age_probs
                matched_trk.ema_gender_probs = 0.50 * g_probs + 0.50 * matched_trk.ema_gender_probs
            a_probs = matched_trk.ema_age_probs
            g_probs = matched_trk.ema_gender_probs

        a_idx = int(np.argmax(a_probs))
        g_idx = int(np.argmax(g_probs))
        age_group = AGE_GROUP_CLASSES[a_idx]
        gender_en = GENDER_LABEL_CLASSES[g_idx]
        gender_vn = VIETNAMESE_GENDER_MAP.get(gender_en, "Nam")

        # Continuous expected age in years from probability distribution over bracket centers
        est_years = int(round(float(np.sum(a_probs * AGE_BRACKET_CENTERS))))

        attr_dict = {
            "age_years": est_years,
            "age_group": age_group,
            "age_display_vn": f"{est_years} tuổi ({age_group})",
            "age_confidence": round(float(a_probs[a_idx]), 4),
            "gender_label": gender_en,
            "gender_vn": gender_vn,
            "gender_confidence": round(float(g_probs[g_idx]), 4),
            "locked_remaining_sec": round(self.attr_refresh_interval_sec if use_tracking else 0.0, 1),
            "model_source": "EfficientNetB0_FairFace_35Epochs",
        }

        if matched_trk is not None:
            matched_trk.locked_attributes = attr_dict.copy()
            matched_trk.last_attr_update_time = now_sec

        return attr_dict

    @staticmethod
    def _rect_intersection_area(r1: Tuple[int, int, int, int], r2: Tuple[int, int, int, int]) -> float:
        x_left = max(r1[0], r2[0])
        y_top = max(r1[1], r2[1])
        x_right = min(r1[2], r2[2])
        y_bottom = min(r1[3], r2[3])
        if x_right <= x_left or y_bottom <= y_top:
            return 0.0
        return float((x_right - x_left) * (y_bottom - y_top))

    def _find_non_overlapping_label_pos(
        self,
        face_bbox: Tuple[int, int, int, int],
        box_w: int,
        box_h: int,
        img_w: int,
        img_h: int,
        placed_labels: List[Tuple[int, int, int, int]],
        all_face_bboxes: List[Tuple[int, int, int, int]],
        prefer_bottom: bool = False,
        gap: int = 8,
    ) -> Tuple[int, int]:
        """Finds a (lx, ly) position for a label box of size (box_w, box_h) that:
        1. Has ZERO overlap (with at least `gap` pixels clearance) with any already-placed label box.
        2. Has ZERO overlap with any detected face bounding box.
        3. Minimizes distance to its own face bounding box.
        """
        fx1, fy1, fx2, fy2 = face_bbox
        fcx = int(0.5 * (fx1 + fx2))
        fcy = int(0.5 * (fy1 + fy2))

        # Generate candidate positions in concentric rings around the face bbox
        candidates: List[Tuple[int, int]] = []
        x_offsets = [
            fcx - box_w // 2,
            fx1,
            fx2 - box_w,
            fcx - box_w,
            fcx,
            fx1 - box_w - gap,
            fx2 + gap,
            fcx - int(0.75 * box_w),
            fcx - int(0.25 * box_w),
        ]

        top_tiers = [
            fy1 - box_h - gap,
            fy1 - 2 * box_h - 2 * gap,
            fy1 - int(1.5 * box_h) - 2 * gap,
            fy1 - 3 * box_h - 3 * gap,
        ]
        bot_tiers = [
            fy2 + gap,
            fy2 + box_h + 2 * gap,
            fy2 + int(1.5 * box_h) + 2 * gap,
            fy2 + 2 * box_h + 3 * gap,
        ]

        y_tiers = (bot_tiers + top_tiers) if prefer_bottom else (top_tiers + bot_tiers)

        for ty in y_tiers:
            for tx in x_offsets:
                candidates.append((tx, ty))

        # Also add side placements (left and right of the face at several vertical offsets)
        for sy in [fy1, fcy - box_h // 2, fy2 - box_h, fy1 - box_h // 2, fy2]:
            candidates.append((fx1 - box_w - gap, sy))
            candidates.append((fx2 + gap, sy))
            candidates.append((fx1 - box_w - 2 * gap - 30, sy))
            candidates.append((fx2 + 2 * gap + 30, sy))

        min_x = 6
        min_y = 42
        max_x = max(min_x, img_w - box_w - 6)
        max_y = max(min_y, img_h - box_h - 6)

        best_pos = (int(np.clip(fcx - box_w // 2, min_x, max_x)), int(np.clip(fy1 - box_h - gap, min_y, max_y)))
        best_cost = float("inf")

        for raw_x, raw_y in candidates:
            lx = int(np.clip(raw_x, min_x, max_x))
            ly = int(np.clip(raw_y, min_y, max_y))
            # Padded rectangle for label-to-label collision check (enforces `gap` px margin between boxes)
            cand_padded = (lx - gap, ly - gap, lx + box_w + gap, ly + box_h + gap)
            cand_exact = (lx, ly, lx + box_w, ly + box_h)

            label_overlap = sum(self._rect_intersection_area(cand_padded, pl) for pl in placed_labels)
            face_overlap = sum(self._rect_intersection_area(cand_exact, fb) for fb in all_face_bboxes)

            # Distance from label center to nearest vertical anchor of own face
            lcx = lx + box_w * 0.5
            lcy = ly + box_h * 0.5
            target_y = (fy2 + box_h * 0.5 + gap) if prefer_bottom else (fy1 - box_h * 0.5 - gap)
            dist_cost = abs(lcx - fcx) * 1.1 + abs(lcy - target_y) * 1.4

            cost = label_overlap * 50000.0 + face_overlap * 15000.0 + dist_cost
            if cost < best_cost:
                best_cost = cost
                best_pos = (lx, ly)

        return best_pos

    def _draw_vietnamese_overlay(
        self,
        vis_bgr: np.ndarray,
        face_outputs: List[Dict[str, Any]],
        fps: float,
        elapsed_ms: float,
        light_state: str,
    ) -> np.ndarray:
        """Draws 3D Delaunay face mesh, 68 landmarks, bounding boxes, and collision-free
        Unicode Vietnamese info cards with leader-line pointers when multiple people stand close together.
        """
        canvas = vis_bgr.copy()
        img_h, img_w = canvas.shape[:2]

        # 1. Draw bounding boxes, 68 landmarks, and 3D Delaunay mesh in OpenCV
        for fo in face_outputs:
            b = fo["bbox"].astype(int)
            color_bgr = (0, 0, 235) if fo["is_unknown"] else (0, 210, 0)
            cv2.rectangle(canvas, (b[0], b[1]), (b[2], b[3]), color_bgr, 2)
            if fo["pts68"] is not None and self.aam is not None:
                canvas = draw_3d_face_mesh_and_pose(canvas, fo["pts68"], self.aam.triangles_, pose_dict=None)

        # 2. Render Unicode Vietnamese text blocks with PIL & Smart Collision Avoidance
        pil_img = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img, "RGBA")

        # Top HUD bar
        hud_text = (
            f"FPS: {fps:.1f} ({elapsed_ms:.1f} ms) | Số khuôn mặt: {len(face_outputs)} | "
            f"Ánh sáng: {light_state} | Chu kỳ cập nhật Tuổi/Giới tính: {int(self.attr_refresh_interval_sec)}s"
        )
        draw.rectangle([(8, 6), (min(pil_img.width - 8, 860), 36)], fill=(0, 0, 0, 175))
        draw.text((14, 10), hud_text, font=self._font_hud, fill=(80, 255, 120, 255))

        if not face_outputs:
            return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

        # Use compact font when there are >= 4 faces or small faces in the frame
        avg_fw = float(np.mean([fo["bbox"][2] - fo["bbox"][0] for fo in face_outputs]))
        use_compact = (len(face_outputs) >= 4) or (avg_fw < 110.0)
        f_title = self._font_title_sm if use_compact else self._font_title
        f_sub = self._font_sub_sm if use_compact else self._font_sub
        box_h = 50 if use_compact else 60
        line_y2 = 18 if use_compact else 22
        line_y3 = 33 if use_compact else 40

        all_face_bboxes: List[Tuple[int, int, int, int]] = [
            (int(fo["bbox"][0]) - 2, int(fo["bbox"][1]) - 2, int(fo["bbox"][2]) + 2, int(fo["bbox"][3]) + 2)
            for fo in face_outputs
        ]

        # Determine median face center Y to naturally send upper-row labels UP and lower-row labels DOWN
        # when multiple rows of people exist (or alternate left-to-right when people sit in one tight row)
        y_centers = [0.5 * (fb[1] + fb[3]) for fb in all_face_bboxes]
        y_spread = max(y_centers) - min(y_centers) if len(y_centers) > 1 else 0.0
        med_y = float(np.median(y_centers))

        # Sort faces left-to-right by x-center for deterministic staggered placement
        order = sorted(range(len(face_outputs)), key=lambda i: 0.5 * (all_face_bboxes[i][0] + all_face_bboxes[i][2]))
        placed_labels: List[Tuple[int, int, int, int]] = []

        for rank_lr, idx in enumerate(order):
            fo = face_outputs[idx]
            fb = all_face_bboxes[idx]
            b = fo["bbox"].astype(int)
            is_unk = fo["is_unknown"]
            name_vn = fo["identity_name_vn"]
            sim = fo["similarity"]
            expr_vn = fo["expression"].get("predicted_expression_vn", "Bình thường")
            attr = fo["attributes"]
            gender_vn = attr.get("gender_vn", "Nam")
            age_vn = attr.get("age_display_vn", "21 tuổi")

            line1 = f"Tên: {name_vn} ({sim:.2f})"
            line2 = f"Giới tính: {gender_vn} | Tuổi: {age_vn}"
            line3 = f"Biểu cảm: {expr_vn}"

            w1 = draw.textlength(line1, font=f_title)
            w2 = draw.textlength(line2, font=f_sub)
            w3 = draw.textlength(line3, font=f_sub)
            box_w = int(max(w1, w2, w3)) + 14

            # Decide whether to prefer placing below or above the face to separate neighbors
            fcy = 0.5 * (fb[1] + fb[3])
            if len(face_outputs) >= 5 and y_spread > 80.0:
                # Two-row group photo: back/upper row goes UP, front/lower row goes DOWN
                prefer_bottom = fcy > med_y + 15.0
            elif len(face_outputs) >= 3:
                # Single row of closely seated people: check horizontal distance to previous neighbor
                prefer_bottom = False
                if rank_lr > 0:
                    prev_fb = all_face_bboxes[order[rank_lr - 1]]
                    horiz_gap = fb[0] - prev_fb[2]
                    if horiz_gap < box_w * 0.75 and (rank_lr % 2 == 1):
                        prefer_bottom = True
            else:
                prefer_bottom = False

            lx, ly = self._find_non_overlapping_label_pos(
                face_bbox=(int(b[0]), int(b[1]), int(b[2]), int(b[3])),
                box_w=box_w,
                box_h=box_h,
                img_w=img_w,
                img_h=img_h,
                placed_labels=placed_labels,
                all_face_bboxes=all_face_bboxes,
                prefer_bottom=prefer_bottom,
                gap=8,
            )
            placed_labels.append((lx, ly, lx + box_w, ly + box_h))

            box_bg = (140, 15, 15, 205) if is_unk else (10, 85, 25, 205)
            border_col = (255, 85, 85, 255) if is_unk else (80, 255, 110, 255)

            # Draw clean leader line connecting the label box to its face bounding box
            fcx = int(0.5 * (b[0] + b[2]))
            lcx = int(np.clip(fcx, lx + 8, lx + box_w - 8))
            if ly + box_h <= b[1]:
                # Label is above face
                p_label = (lcx, ly + box_h)
                p_face = (fcx, int(b[1]))
            elif ly >= b[3]:
                # Label is below face
                p_label = (lcx, ly)
                p_face = (fcx, int(b[3]))
            elif lx + box_w <= b[0]:
                # Label is left of face
                p_label = (lx + box_w, int(0.5 * (ly + ly + box_h)))
                p_face = (int(b[0]), int(0.5 * (b[1] + b[3])))
            else:
                # Label is right of face
                p_label = (lx, int(0.5 * (ly + ly + box_h)))
                p_face = (int(b[2]), int(0.5 * (b[1] + b[3])))

            dist_line = np.hypot(p_label[0] - p_face[0], p_label[1] - p_face[1])
            if dist_line > 6.0:
                draw.line([p_label, p_face], fill=border_col, width=2)
                r_dot = 3
                draw.ellipse(
                    [(p_face[0] - r_dot, p_face[1] - r_dot), (p_face[0] + r_dot, p_face[1] + r_dot)],
                    fill=border_col,
                )

            draw.rectangle(
                [(lx, ly), (lx + box_w, ly + box_h)],
                fill=box_bg,
                outline=border_col,
                width=2,
            )
            draw.text((lx + 7, ly + 2), line1, font=f_title, fill=(255, 255, 255, 255))
            draw.text((lx + 7, ly + line_y2), line2, font=f_sub, fill=(255, 240, 130, 255))
            draw.text((lx + 7, ly + line_y3), line3, font=f_sub, fill=(180, 245, 255, 255))

        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    def process_frame(
        self,
        frame_bgr: np.ndarray,
        is_cropped_face: bool = False,
        use_tracking: bool = True,
        draw_overlay: bool = True,
        timestamp_sec: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Processes a single image or video frame end-to-end with Vietnamese outputs
        and 10-second locked Age/Gender attributes per tracked identity.
        """
        t0 = time.perf_counter()
        now_sec = time.monotonic() if timestamp_sec is None else float(timestamp_sec)
        self._frame_counter += 1
        h, w = frame_bgr.shape[:2]

        proc_img, light_state = self.preprocess_lighting(frame_bgr)
        dets = self.detector.detect(proc_img, is_cropped_face=is_cropped_face)

        # Match detections to existing tracks first so we know which track_id's 10s lock applies
        raw_dets_for_tracker = []
        aligned_crops = []

        for d in dets:
            bbox = d["bbox"]
            kps5 = d["kps5"]
            score = d["score"]

            matched_trk = None
            if use_tracking and self.tracker.tracks:
                best_iou = 0.0
                for trk in self.tracker.tracks:
                    if trk.time_since_update <= 1 and trk.embedding is not None:
                        ov = bbox_iou(trk.bbox, bbox)
                        if ov > best_iou:
                            best_iou = ov
                            matched_trk = trk
                if best_iou < 0.45:
                    matched_trk = None

            aligned_112, _ = align_face_arcface(proc_img, kps5, image_size=112)
            face_ctx_224 = self._crop_face_224_with_context(proc_img, bbox, margin=0.08)
            aligned_crops.append(face_ctx_224)

            need_arcface = (
                (not use_tracking)
                or (matched_trk is None)
                or (matched_trk.hits <= 2)
                or (self._frame_counter % self.reid_interval == 0)
            )
            if need_arcface:
                emb = self.embedder.extract(aligned_112, flip_tta=False)
                rec = self.db.identify(emb)
            else:
                emb = matched_trk.embedding
                rec = self.db.identify(emb)

            pts68 = None
            if self.aam is not None:
                init_pts68 = self.init_aam_shape_from_5kps(kps5)
                if self.sdm is not None:
                    pts68 = self.sdm.predict(proc_img, init_pts68, n_stages=self.sdm_stages)
                else:
                    pts68, _ = self.aam.fit_image(proc_img, init_pts68, num_iters=self.aam_fit_iters)

            raw_dets_for_tracker.append({
                "bbox": bbox,
                "score": score,
                "kps5": kps5,
                "embedding": emb,
                "pts68": pts68,
                "identity_id": rec["identity_id"],
                "identity_name": rec["identity_name"],
                "identity_sim": rec["similarity"],
                "aligned_112": aligned_112,
                "face_ctx_224": face_ctx_224,
            })

        face_outputs = []
        if use_tracking:
            tracks = self.tracker.update(raw_dets_for_tracker)
            for trk in tracks:
                best_crop = None
                best_aligned = None
                best_ov = -1.0
                for rd in raw_dets_for_tracker:
                    ov = bbox_iou(trk.bbox, rd["bbox"])
                    if ov > best_ov:
                        best_ov = ov
                        best_crop = rd["face_ctx_224"]
                        best_aligned = rd["aligned_112"]
                if best_crop is None and raw_dets_for_tracker:
                    best_crop = raw_dets_for_tracker[0]["face_ctx_224"]
                    best_aligned = raw_dets_for_tracker[0]["aligned_112"]

                rec_smooth = self.db.identify(trk.embedding) if trk.embedding is not None else {
                    "identity_id": -1,
                    "identity_name": trk.identity_name,
                    "similarity": trk.identity_sim,
                    "is_unknown": trk.identity_name == "Unknown",
                }
                pts68 = trk.pts68
                pose = estimate_head_pose_68(pts68, w, h) if pts68 is not None else {"yaw": 0.0, "pitch": 0.0, "roll": 0.0}

                expr_dict = self._infer_emotion(best_crop, pts68, matched_trk=trk) if best_crop is not None else {
                    "predicted_expression": "Neutral",
                    "predicted_expression_vn": "Bình thường",
                    "confidence": 0.5,
                }
                attr_dict = self._infer_age_gender_with_10s_lock(
                    best_crop, matched_trk=trk, now_sec=now_sec, use_tracking=True, aligned_bgr_112=best_aligned
                ) if best_crop is not None else {
                    "age_years": 21,
                    "age_group": "20-29",
                    "age_display_vn": "25 tuổi (20-29)",
                    "gender_vn": "Nam",
                }

                en_name = rec_smooth["identity_name"]
                vn_name = VIETNAMESE_NAME_MAP.get(en_name, en_name)

                face_outputs.append({
                    "track_id": trk.track_id,
                    "bbox": trk.bbox,
                    "det_score": trk.score,
                    "identity_id": rec_smooth["identity_id"],
                    "identity_name": en_name,
                    "identity_name_vn": vn_name,
                    "similarity": rec_smooth["similarity"],
                    "is_unknown": rec_smooth["is_unknown"],
                    "pts68": pts68,
                    "pose": pose,
                    "expression": expr_dict,
                    "attributes": attr_dict,
                })
        else:
            for idx, rd in enumerate(raw_dets_for_tracker, start=1):
                pts68 = rd["pts68"]
                pose = estimate_head_pose_68(pts68, w, h) if pts68 is not None else {"yaw": 0.0, "pitch": 0.0, "roll": 0.0}
                expr_dict = self._infer_emotion(rd["face_ctx_224"], pts68, matched_trk=None)
                attr_dict = self._infer_age_gender_with_10s_lock(
                    rd["face_ctx_224"], matched_trk=None, now_sec=now_sec, use_tracking=False, aligned_bgr_112=rd["aligned_112"]
                )
                en_name = rd["identity_name"]
                vn_name = VIETNAMESE_NAME_MAP.get(en_name, en_name)
                face_outputs.append({
                    "track_id": idx,
                    "bbox": rd["bbox"],
                    "det_score": rd["score"],
                    "identity_id": rd["identity_id"],
                    "identity_name": en_name,
                    "identity_name_vn": vn_name,
                    "similarity": rd["identity_sim"],
                    "is_unknown": en_name == "Unknown",
                    "pts68": pts68,
                    "pose": pose,
                    "expression": expr_dict,
                    "attributes": attr_dict,
                })

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        fps = 1000.0 / max(elapsed_ms, 1e-3)

        if draw_overlay:
            vis_frame = self._draw_vietnamese_overlay(frame_bgr, face_outputs, fps, elapsed_ms, light_state)
        else:
            vis_frame = frame_bgr.copy()

        return {
            "faces": face_outputs,
            "latency_ms": round(elapsed_ms, 2),
            "fps": round(fps, 2),
            "lighting_state": light_state,
            "vis_frame": vis_frame,
        }
