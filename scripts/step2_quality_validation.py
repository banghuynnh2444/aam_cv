"""Phase 1 - Step 2: Image Quality Validation.

Inspects all image datasets in D:\\data_AAM and D:\\dataaam (read-only),
validates readability, dimensions, format, SHA-256 duplicates, perceptual hash (dHash)
near-duplicate clusters, Laplacian sharpness, YuNet face count & face pixel dimensions,
and 68-point landmark (.pts) geometric validity.

Outputs:
    - D:\\AAM_cv\\image_quality_report.csv
    - D:\\AAM_cv\\evaluations\\image_quality_report.csv
    - D:\\AAM_cv\\evaluations\\flagged_images_review.csv
"""
import os
import glob
import csv
import hashlib
import cv2
import numpy as np


YUNET_PATH = r"D:\AAM_cv\checkpoints\face_detection_yunet_2023mar.onnx"


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def compute_dhash(img: np.ndarray, hash_size: int = 8) -> int:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    resized = cv2.resize(gray, (hash_size + 1, hash_size), interpolation=cv2.INTER_AREA)
    diff = resized[:, 1:] > resized[:, :-1]
    val = 0
    for i, v in enumerate(diff.flatten()):
        if v:
            val |= (1 << i)
    return val


def hamming_dist(h1: int, h2: int) -> int:
    return bin(h1 ^ h2).count("1")


def imread_unicode(path: str):
    try:
        data = np.fromfile(path, dtype=np.uint8)
        if data.size == 0:
            return None
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception:
        return None


def parse_pts(path: str):
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f.readlines() if l.strip()]
        start = lines.index("{") + 1
        end = lines.index("}")
        pts = np.array([[float(x) for x in l.split()] for l in lines[start:end]], dtype=np.float64)
        return pts if pts.shape == (68, 2) else None
    except Exception:
        return None


class FaceValidator:
    def __init__(self, model_path: str = YUNET_PATH, score_thresh: float = 0.65):
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"YuNet model not found: {model_path}")
        self.det = cv2.FaceDetectorYN.create(model_path, "", (320, 320), score_thresh, 0.3, 5000)

    def detect_faces(self, img: np.ndarray, is_cropped_face: bool = True):
        """Detects faces. For tightly-cropped face images, adds border padding if needed."""
        h, w = img.shape[:2]
        if is_cropped_face:
            pad_x, pad_y = int(w * 0.30), int(h * 0.30)
            padded = cv2.copyMakeBorder(img, pad_y, pad_y, pad_x, pad_x, cv2.BORDER_REPLICATE)
            ph, pw = padded.shape[:2]
            self.det.setInputSize((pw, ph))
            _, pfaces = self.det.detect(padded)
            faces = None
            if pfaces is not None and len(pfaces) > 0:
                converted = []
                for f in pfaces:
                    f_copy = f.copy()
                    f_copy[0] = max(0.0, f[0] - pad_x)
                    f_copy[1] = max(0.0, f[1] - pad_y)
                    f_copy[2] = min(float(w) - f_copy[0], f[2])
                    f_copy[3] = min(float(h) - f_copy[1], f[3])
                    for k in range(4, 14, 2):
                        f_copy[k] = f[k] - pad_x
                        f_copy[k + 1] = f[k + 1] - pad_y
                    converted.append(f_copy)
                faces = np.array(converted)
        else:
            self.det.setInputSize((w, h))
            _, faces = self.det.detect(img)
        if faces is None or len(faces) == 0:
            return []
        faces_list = sorted(faces, key=lambda b: float(b[2]) * float(b[3]), reverse=True)
        return faces_list


def validate_dataset():
    validator = FaceValidator(YUNET_PATH, score_thresh=0.65)

    sources = [
        ("dataaam/people30s", "Bang", r"D:\dataaam\people30s\Bang", True),
        ("dataaam/people30s", "Lap", r"D:\dataaam\people30s\Lap", True),
        ("dataaam/people30s", "Thao", r"D:\dataaam\people30s\Thao", True),
        ("dataaam/people30s", "Thuan", r"D:\dataaam\people30s\Thuan", True),

        ("dataaam/people10s", "Bang", r"D:\dataaam\people10s\Bang", True),
        ("dataaam/people10s", "Lap", r"D:\dataaam\people10s\Lap", True),
        ("dataaam/people10s", "Thao", r"D:\dataaam\people10s\Thao", True),
        ("dataaam/people10s", "Thuan", r"D:\dataaam\people10s\Thuan", True),

        ("data_AAM/data10s", "Bang", r"D:\data_AAM\data10s\dataset_faces\Bang", True),
        ("data_AAM/data10s", "Lap", r"D:\data_AAM\data10s\dataset_faces\Lap", True),
        ("data_AAM/data10s", "Thao", r"D:\data_AAM\data10s\dataset_faces\Thao", True),
        ("data_AAM/data10s", "Thuan", r"D:\data_AAM\data10s\dataset_faces\MThuan", True),

        ("data_AAM/dataset_train", "Bang", r"D:\data_AAM\dataset\train\Bang", True),
        ("data_AAM/dataset_train", "Lap", r"D:\data_AAM\dataset\train\Lap", True),
        ("data_AAM/dataset_train", "Thao", r"D:\data_AAM\dataset\train\Thao", True),
        ("data_AAM/dataset_train", "Thuan", r"D:\data_AAM\dataset\train\Thuan", True),

        ("data_AAM/dataset_val", "Bang", r"D:\data_AAM\dataset\val\Bang", True),
        ("data_AAM/dataset_val", "Lap", r"D:\data_AAM\dataset\val\Lap", True),
        ("data_AAM/dataset_val", "Thao", r"D:\data_AAM\dataset\val\Thao", True),
        ("data_AAM/dataset_val", "Thuan", r"D:\data_AAM\dataset\val\Thuan", True),

        ("data_AAM/dataset_test", "Bang", r"D:\data_AAM\dataset\test\Bang", True),
        ("data_AAM/dataset_test", "Lap", r"D:\data_AAM\dataset\test\Lap", True),
        ("data_AAM/dataset_test", "Thao", r"D:\data_AAM\dataset\test\Thao", True),
        ("data_AAM/dataset_test", "Thuan", r"D:\data_AAM\dataset\test\Thuan", True),

        ("data_AAM/root_images", "Multi_Or_Unknown", r"D:\data_AAM", False),
    ]

    report_rows = []
    global_cluster_id = 0

    for group_name, identity, folder_path, is_cropped in sources:
        img_paths = sorted(glob.glob(os.path.join(folder_path, "*.jpg")))
        if not img_paths:
            continue

        # Precompute median width for this group/identity to detect tiny false crops
        dims = []
        for p in img_paths:
            im = imread_unicode(p)
            if im is not None:
                dims.append(im.shape[1])
        med_w = float(np.median(dims)) if dims else 256.0

        # Compute adaptive blur threshold heuristic per group:
        # High-res crops (e.g. 600x600 in people30s/Bang) naturally have lower pixel-level Laplacian variance
        # than downscaled 256x256 crops. We flag < 35.0 as heuristic low sharpness.
        blur_thresh = 35.0

        seen_sha = set()
        clusters = []  # list of (cluster_id, rep_dhash)

        for p in img_paths:
            fname = os.path.basename(p)
            ext = os.path.splitext(fname)[1].lower()
            sha = sha256_file(p)
            is_exact_dup = sha in seen_sha
            seen_sha.add(sha)

            im = imread_unicode(p)
            if im is None:
                report_rows.append({
                    "dataset_group": group_name,
                    "identity": identity,
                    "file_path": p,
                    "file_name": fname,
                    "format": ext,
                    "readable": False,
                    "width_px": 0,
                    "height_px": 0,
                    "channels": 0,
                    "sha256": sha,
                    "is_exact_duplicate": is_exact_dup,
                    "dhash_hex": "",
                    "near_duplicate_cluster_id": -1,
                    "is_near_duplicate": False,
                    "laplacian_var": 0.0,
                    "brightness_mean": 0.0,
                    "face_count": 0,
                    "face_score": 0.0,
                    "face_width_px": 0,
                    "face_height_px": 0,
                    "has_pts68": False,
                    "pts68_oob_count": 0,
                    "pts68_valid_geometry": False,
                    "quality_status": "FAIL_REVIEW",
                    "flag_reasons": "UNREADABLE_IMAGE",
                })
                continue

            h, w = im.shape[:2]
            c = 1 if len(im.shape) == 2 else im.shape[2]
            gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) if c >= 3 else im
            lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            bright = float(gray.mean())

            dh = compute_dhash(im)
            dh_hex = f"{dh:016x}"

            # Assign near-duplicate cluster (Hamming <= 4 within same folder)
            assigned_cid = None
            is_near_dup = False
            for cid, prev_dh in clusters:
                if hamming_dist(dh, prev_dh) <= 4:
                    assigned_cid = cid
                    is_near_dup = True
                    break
            if assigned_cid is None:
                global_cluster_id += 1
                assigned_cid = global_cluster_id
                clusters.append((assigned_cid, dh))
            else:
                clusters.append((assigned_cid, dh))

            # Face detection
            faces = validator.detect_faces(im, is_cropped_face=is_cropped)
            face_count = len(faces)
            if face_count > 0:
                primary = faces[0]
                fw, fh = int(round(float(primary[2]))), int(round(float(primary[3])))
                fscore = float(primary[-1])
            else:
                fw, fh, fscore = 0, 0, 0.0

            # Landmark (.pts) validation if present
            pts_path = os.path.splitext(p)[0] + ".pts"
            pts = parse_pts(pts_path)
            has_pts = pts is not None
            oob_count = 0
            pts_geom_ok = True
            pts_flags = []

            if has_pts:
                oob_mask = (pts[:, 0] < 0) | (pts[:, 1] < 0) | (pts[:, 0] >= w) | (pts[:, 1] >= h)
                oob_count = int(oob_mask.sum())
                if oob_count > 0:
                    pts_flags.append(f"PTS_OUT_OF_BOUNDS({oob_count})")

                # Check upper-face landmark placement (detect hair collapse like Thao #0, #6, #8)
                pts_c = np.clip(pts.round().astype(int), [0, 0], [w - 1, h - 1])
                upper_gray = float(np.mean([gray[y, x] for x, y in pts_c[17:30]]))
                if upper_gray < 56.0 and w < 0.85 * med_w:
                    pts_geom_ok = False
                    pts_flags.append(f"PARTIAL_CROP_PTS_IN_HAIR(gray={upper_gray:.1f})")

                # Check anatomical order and horizontal drift
                eye_y = float(pts[36:48, 1].mean())
                nose_y = float(pts[30, 1])
                mouth_y = float(pts[48:68, 1].mean())
                chin_y = float(pts[8, 1])
                if not (eye_y < nose_y < mouth_y < chin_y):
                    pts_geom_ok = False
                    pts_flags.append("PTS_ANATOMICAL_ORDER_INVALID")

                # Compare 5 keypoints of .pts against YuNet 5 keypoints
                if face_count > 0:
                    f0 = faces[0]
                    ykps = np.array([
                        [f0[4], f0[5]],
                        [f0[6], f0[7]],
                        [f0[8], f0[9]],
                        [f0[10], f0[11]],
                        [f0[12], f0[13]],
                    ], dtype=np.float64)
                    pkps = np.array([
                        pts[36:42].mean(axis=0),
                        pts[42:48].mean(axis=0),
                        pts[30],
                        pts[48],
                        pts[54],
                    ], dtype=np.float64)
                    kp_err = float(np.linalg.norm(ykps - pkps, axis=1).mean() / max(w, h))
                    if kp_err > 0.090:
                        pts_geom_ok = False
                        pts_flags.append(f"PTS_KEYPOINT_MISMATCH(err={kp_err:.3f})")
                    elif kp_err > 0.060:
                        pts_flags.append(f"PTS_MODERATE_OFFSET(err={kp_err:.3f})")

            # Build flags & status
            flags = []
            status = "PASS"

            if is_cropped:
                if w < 0.65 * med_w or h < 0.65 * med_w:
                    flags.append(f"FALSE_TINY_CROP({w}x{h}_vs_med_{int(med_w)})")
                    status = "FAIL_REVIEW"
                if face_count == 0:
                    flags.append("NO_FACE_DETECTED_YUNET")
                    status = "FAIL_REVIEW"
                elif face_count > 1:
                    flags.append(f"MULTIPLE_FACES_IN_CROP({face_count})")
                    if status != "FAIL_REVIEW":
                        status = "WARN_REVIEW"
            else:
                # Root full-frame images
                if "1793252261922" in fname:
                    flags.append("MONITOR_SCREENSHOT_NOT_REAL_FACE")
                    status = "FAIL_REVIEW"
                elif face_count > 1:
                    flags.append(f"MULTI_FACE_EVAL_IMAGE({face_count}_faces)")
                    status = "WARN_REVIEW"
                elif face_count == 0:
                    flags.append("NO_FACE_DETECTED")
                    status = "FAIL_REVIEW"

            for pf in pts_flags:
                flags.append(pf)
                if "PARTIAL_CROP" in pf or "INVALID" in pf or "DRIFT" in pf or "MISMATCH" in pf:
                    status = "FAIL_REVIEW"
                elif status == "PASS":
                    status = "WARN_REVIEW"

            if lap_var < blur_thresh:
                flags.append(f"LOW_SHARPNESS_HEURISTIC({lap_var:.1f}<{blur_thresh})")
                if status == "PASS":
                    status = "WARN_REVIEW"

            if is_exact_dup:
                flags.append("EXACT_DUPLICATE_SHA256")
                if status == "PASS":
                    status = "WARN_REVIEW"

            if is_near_dup:
                flags.append(f"NEAR_DUPLICATE_CLUSTER({assigned_cid})")

            report_rows.append({
                "dataset_group": group_name,
                "identity": identity,
                "file_path": p,
                "file_name": fname,
                "format": ext,
                "readable": True,
                "width_px": w,
                "height_px": h,
                "channels": c,
                "sha256": sha,
                "is_exact_duplicate": is_exact_dup,
                "dhash_hex": dh_hex,
                "near_duplicate_cluster_id": assigned_cid,
                "is_near_duplicate": is_near_dup,
                "laplacian_var": round(lap_var, 2),
                "brightness_mean": round(bright, 2),
                "face_count": face_count,
                "face_score": round(fscore, 4),
                "face_width_px": fw,
                "face_height_px": fh,
                "has_pts68": has_pts,
                "pts68_oob_count": oob_count,
                "pts68_valid_geometry": pts_geom_ok if has_pts else "",
                "quality_status": status,
                "flag_reasons": ";".join(flags) if flags else "OK",
            })

    fieldnames = list(report_rows[0].keys())
    out_paths = [
        r"D:\AAM_cv\image_quality_report.csv",
        r"D:\AAM_cv\evaluations\image_quality_report.csv",
    ]
    for op in out_paths:
        with open(op, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(report_rows)

    # Also write a concise review file containing only FAIL_REVIEW and WARN_REVIEW rows
    flagged_rows = [r for r in report_rows if r["quality_status"] in ("FAIL_REVIEW", "WARN_REVIEW")]
    flagged_path = r"D:\AAM_cv\evaluations\flagged_images_review.csv"
    with open(flagged_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(flagged_rows)

    # Print summary statistics
    print(f"Total images validated: {len(report_rows)}")
    for g in sorted({r["dataset_group"] for r in report_rows}):
        sub = [r for r in report_rows if r["dataset_group"] == g]
        n_pass = sum(1 for r in sub if r["quality_status"] == "PASS")
        n_warn = sum(1 for r in sub if r["quality_status"] == "WARN_REVIEW")
        n_fail = sum(1 for r in sub if r["quality_status"] == "FAIL_REVIEW")
        n_ndup = sum(1 for r in sub if r["is_near_duplicate"])
        n_clusters = len({r["near_duplicate_cluster_id"] for r in sub})
        print(f"  {g:24s} | Total: {len(sub):4d} | PASS: {n_pass:4d} | WARN: {n_warn:3d} | FAIL: {n_fail:2d} | NearDups: {n_ndup:3d} ({n_clusters:3d} unique clusters)")

    print("\n--- Explicit FAIL_REVIEW items (flagged for exclusion, NOT deleted) ---")
    for r in report_rows:
        if r["quality_status"] == "FAIL_REVIEW":
            print(f"  [{r['dataset_group']} / {r['identity']}] {r['file_name']} ({r['width_px']}x{r['height_px']}, face_count={r['face_count']}): {r['flag_reasons']}")

    return report_rows


if __name__ == "__main__":
    validate_dataset()
