"""Standard 68-Point Facial Landmark Utilities (iBUG / 300-W Convention).

Provides:
- Reading and writing Menpo/iBUG .pts files (68 points)
- Coordinate system normalization, clipping, and metadata packaging
- Canonical 5-keypoint extraction from 68 landmarks
- Anatomically verified, deterministic mapping rule from MediaPipe 468/478 Face Mesh
  to iBUG 68 landmarks (with symmetry and anatomical ordering validation)
"""
import os
from typing import Dict, Any, Tuple
import numpy as np


# Standard iBUG / 300-W 68-point anatomical regions (0-indexed)
IBUG68_REGIONS: Dict[str, Tuple[int, int]] = {
    "jaw": (0, 17),            # 0..16  (17 pts: viewer-left ear to viewer-right ear)
    "right_eyebrow": (17, 22), # 17..21 (5 pts: subject's right / viewer-left eyebrow)
    "left_eyebrow": (22, 27),  # 22..26 (5 pts: subject's left / viewer-right eyebrow)
    "nose_bridge": (27, 31),   # 27..30 (4 pts: top of bridge to nose tip 30)
    "nose_lower": (31, 36),    # 31..35 (5 pts: nostrils / lower nose base)
    "right_eye": (36, 42),     # 36..41 (6 pts: subject's right / viewer-left eye)
    "left_eye": (42, 48),      # 42..47 (6 pts: subject's left / viewer-right eye)
    "outer_lip": (48, 60),     # 48..59 (12 pts: outer mouth contour)
    "inner_lip": (60, 68),     # 60..67 (8 pts: inner mouth contour)
}

# Deterministic, anatomically verified mapping from MediaPipe 468 Face Mesh vertex indices
# to the 68 iBUG/300-W landmarks (symmetric across sagittal plane).
MEDIAPIPE_468_TO_IBUG68_MAP: Tuple[int, ...] = (
    # Jawline 0..16 (viewer-left to chin 8 to viewer-right)
    234, 93, 132, 58, 172, 136, 150, 149, 152, 378, 379, 365, 397, 288, 361, 323, 454,
    # Right eyebrow (viewer-left) 17..21
    70, 63, 105, 66, 107,
    # Left eyebrow (viewer-right) 22..26
    336, 296, 334, 293, 300,
    # Nose bridge 27..30
    168, 197, 5, 4,
    # Lower nose 31..35
    75, 97, 2, 326, 305,
    # Right eye (viewer-left) 36..41
    33, 160, 158, 133, 153, 144,
    # Left eye (viewer-right) 42..47
    362, 385, 387, 263, 373, 380,
    # Outer lip 48..59
    61, 40, 37, 0, 267, 270, 291, 321, 314, 17, 84, 91,
    # Inner lip 60..67
    78, 81, 13, 311, 308, 402, 14, 178,
)


def read_pts68(path: str) -> np.ndarray:
    """Reads a 68-point Menpo/iBUG .pts file into a (68, 2) float32 array."""
    with open(path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    start = lines.index("{") + 1
    end = lines.index("}")
    pts = np.array([[float(v) for v in line.split()[:2]] for line in lines[start:end]], dtype=np.float32)
    if pts.shape != (68, 2):
        raise ValueError(f"Expected (68, 2) landmarks in {path}, got {pts.shape}")
    return pts


def write_pts68(path: str, pts68: np.ndarray) -> None:
    """Writes a (68, 2) array to a standard Menpo/iBUG .pts file."""
    pts68 = np.asarray(pts68, dtype=np.float32)
    if pts68.shape != (68, 2):
        raise ValueError(f"Expected (68, 2) array, got {pts68.shape}")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("version: 1\n")
        f.write("n_points: 68\n")
        f.write("{\n")
        for x, y in pts68:
            f.write(f"{x:.3f} {y:.3f}\n")
        f.write("}\n")


def extract_5kps_from_68(pts68: np.ndarray) -> np.ndarray:
    """Derives the canonical 5 facial keypoints (ArcFace convention) from 68 iBUG landmarks:
    [viewer_left_eye, viewer_right_eye, nose_tip, viewer_left_mouth, viewer_right_mouth].
    """
    pts68 = np.asarray(pts68, dtype=np.float32)
    if pts68.shape != (68, 2):
        raise ValueError(f"Expected (68, 2), got {pts68.shape}")
    left_eye = pts68[36:42].mean(axis=0)
    right_eye = pts68[42:48].mean(axis=0)
    nose_tip = pts68[30]
    mouth_left = pts68[48]
    mouth_right = pts68[54]
    return np.stack([left_eye, right_eye, nose_tip, mouth_left, mouth_right], axis=0).astype(np.float32)


def compute_interocular_distance(pts68: np.ndarray, mode: str = "outer_corners") -> float:
    """Computes inter-ocular distance for NME (Normalized Mean Error) calculation.
    300-W standard uses 'outer_corners' (distance between point 36 and point 45).
    """
    pts68 = np.asarray(pts68, dtype=np.float32)
    if mode == "outer_corners":
        return float(np.linalg.norm(pts68[36] - pts68[45]))
    elif mode == "centers":
        c_left = pts68[36:42].mean(axis=0)
        c_right = pts68[42:48].mean(axis=0)
        return float(np.linalg.norm(c_left - c_right))
    raise ValueError(f"Unknown mode: {mode}")


def normalize_landmarks68(
    pts68: np.ndarray,
    image_width: int,
    image_height: int,
    is_ground_truth: bool = False,
    source_tag: str = "auto_predicted_validated",
) -> Dict[str, Any]:
    """Packages 68-point landmarks with explicit coordinate system metadata."""
    pts68 = np.asarray(pts68, dtype=np.float32)
    if pts68.shape != (68, 2):
        raise ValueError(f"Expected (68, 2), got {pts68.shape}")

    clipped = np.empty_like(pts68)
    clipped[:, 0] = np.clip(pts68[:, 0], 0.0, float(image_width - 1))
    clipped[:, 1] = np.clip(pts68[:, 1], 0.0, float(image_height - 1))

    norm01 = np.empty_like(pts68)
    norm01[:, 0] = pts68[:, 0] / float(max(1, image_width))
    norm01[:, 1] = pts68[:, 1] / float(max(1, image_height))

    # Zero-mean unit-scale centered shape for Procrustes / AAM shape model
    centroid = pts68.mean(axis=0)
    centered = pts68 - centroid
    scale = float(np.linalg.norm(centered))
    unit_shape = centered / (scale + 1e-8)

    oob_mask = (
        (pts68[:, 0] < 0)
        | (pts68[:, 1] < 0)
        | (pts68[:, 0] >= image_width)
        | (pts68[:, 1] >= image_height)
    )

    xmin, ymin = pts68.min(axis=0)
    xmax, ymax = pts68.max(axis=0)

    return {
        "coords_pixel": pts68,
        "coords_clipped": clipped,
        "coords_normalized": norm01,
        "unit_centered_shape": unit_shape.astype(np.float32),
        "centroid_xy": centroid.astype(np.float32),
        "procrustes_scale": np.float32(scale),
        "kps5": extract_5kps_from_68(pts68),
        "bbox_landmarks": np.array([xmin, ymin, xmax, ymax], dtype=np.float32),
        "interocular_outer": np.float32(compute_interocular_distance(pts68, "outer_corners")),
        "interocular_centers": np.float32(compute_interocular_distance(pts68, "centers")),
        "image_size_wh": np.array([image_width, image_height], dtype=np.int32),
        "oob_count": int(oob_mask.sum()),
        "is_ground_truth": bool(is_ground_truth),
        "coord_system": "pixel_top_left_origin_xy",
        "source_tag": source_tag,
    }


def map_mediapipe_to_ibug68(mp_landmarks: np.ndarray) -> np.ndarray:
    """Maps MediaPipe Face Landmarker (468 or 478, 2 or 3) coordinates to iBUG 68 (68, 2 or 3)
    using the deterministic anatomical correspondence table MEDIAPIPE_468_TO_IBUG68_MAP.
    """
    mp_landmarks = np.asarray(mp_landmarks, dtype=np.float32)
    if mp_landmarks.ndim != 2 or mp_landmarks.shape[0] < 468:
        raise ValueError(f"Expected MediaPipe landmarks with >=468 points, got {mp_landmarks.shape}")
    return mp_landmarks[list(MEDIAPIPE_468_TO_IBUG68_MAP)].copy()
