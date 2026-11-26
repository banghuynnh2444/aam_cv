"""Phase 1 - Step 3 (Face Detection & Alignment) and Step 4 (Landmark Preparation).

Executes:
- Step 3:
  1. Runs SCRFD-2.5G-KPS (scrfd_2.5g_bnkps.onnx) on all validated personal face images
     from Session 30s (312 images) and Session 10s (181 images) = 493 images.
  2. Copies original images safely to D:\\AAM_cv\\datasets\\raw\\personal_faces\\{Identity}\\
     (leaving D:\\data_AAM and D:\\dataaam untouched).
  3. Aligns faces via 5 keypoints (Umeyama similarity transform) to 112x112 for ArcFace
     and saves to D:\\AAM_cv\\datasets\\processed\\aligned_faces\\{Identity}\\.
  4. Processes the 7 multi-face group images from D:\\data_AAM\\*.jpg with SCRFD-2.5G-KPS,
     saving numbered face bounding box visualizations and 112x112 crops to
     D:\\AAM_cv\\datasets\\processed\\aligned_faces\\multi_face_pending_verification\\
     for user identity verification (Rule Step 3.4).
- Step 4:
  1. Standardizes all 488 validated 68-point landmark annotations (312 from people30s +
     176 from people10s) into D:\\AAM_cv\\datasets\\processed\\landmarks68\\{Identity}\\
     with explicit coordinate system metadata and is_ground_truth=False.
  2. Writes manifests for aligned faces and 68-point landmarks.
"""
import os
import sys
import csv
import shutil
import cv2
import numpy as np

sys.path.insert(0, r"D:\AAM_cv")
from src.detection.scrfd import SCRFDDetector, align_face_arcface
from src.geometry.landmarks68 import (
    read_pts68,
    write_pts68,
    extract_5kps_from_68,
    normalize_landmarks68,
)


def imread_unicode(path: str):
    data = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def imwrite_unicode(path: str, img: np.ndarray, quality: int = 95):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])[1].tofile(path)


def run_step3_and_step4():
    report_csv = r"D:\AAM_cv\image_quality_report.csv"
    if not os.path.isfile(report_csv):
        raise FileNotFoundError(f"Run Step 2 first. Missing {report_csv}")

    with open(report_csv, "r", encoding="utf-8-sig") as f:
        quality_rows = list(csv.DictReader(f))

    detector = SCRFDDetector(
        model_path=r"D:\AAM_cv\checkpoints\scrfd_2.5g_bnkps.onnx",
        input_size=(640, 640),
        conf_thresh=0.45,
        nms_thresh=0.40,
        use_cuda=True,
    )

    # Collect valid personal images:
    # 1. All non-FAIL rows from dataaam/people30s (312 images with .pts)
    # 2. All non-FAIL rows from dataaam/people10s (176 images with .pts)
    # 3. The 5 extra images in data_AAM/data10s that were dropped in dataaam/people10s
    p30_rows = [
        r for r in quality_rows
        if r["dataset_group"] == "dataaam/people30s" and r["quality_status"] != "FAIL_REVIEW"
    ]
    p10_rows = [
        r for r in quality_rows
        if r["dataset_group"] == "dataaam/people10s" and r["quality_status"] != "FAIL_REVIEW"
    ]

    p10_names = {r["file_name"] for r in p10_rows}
    d10_extra_rows = []
    for r in quality_rows:
        if r["dataset_group"] == "data_AAM/data10s" and r["quality_status"] != "FAIL_REVIEW":
            stem = os.path.splitext(r["file_name"])[0]
            if not any(os.path.splitext(pn)[0].endswith(stem) for pn in p10_names):
                d10_extra_rows.append(r)

    all_single_rows = [
        ("s30", r) for r in p30_rows
    ] + [
        ("s10", r) for r in p10_rows
    ] + [
        ("s10_extra", r) for r in d10_extra_rows
    ]

    aligned_base = r"D:\AAM_cv\datasets\processed\aligned_faces"
    raw_personal_base = r"D:\AAM_cv\datasets\raw\personal_faces"
    lm68_base = r"D:\AAM_cv\datasets\processed\landmarks68"

    aligned_manifest = []
    lm68_manifest = []
    consolidated_lm68 = []

    scrfd_direct_count = 0
    pts_fallback_count = 0

    for session_tag, r in all_single_rows:
        identity = r["identity"]
        src_img_path = r["file_path"]
        fname = r["file_name"]
        stem = os.path.splitext(fname)[0]
        # Ensure clean unique sample ID prefixed with identity and session
        if stem.startswith(f"{identity}_s10") or stem.startswith(f"{identity}_s30"):
            sample_id = stem
        else:
            sample_id = f"{identity}_{session_tag}_{stem}"

        img = imread_unicode(src_img_path)
        h, w = img.shape[:2]

        # Copy original image safely to datasets/raw/personal_faces/{Identity}/
        raw_dst_path = os.path.join(raw_personal_base, identity, f"{sample_id}.jpg")
        if not os.path.exists(raw_dst_path):
            shutil.copy2(src_img_path, raw_dst_path)

        # Read 68 landmarks if available
        src_pts_path = os.path.splitext(src_img_path)[0] + ".pts"
        has_pts = os.path.isfile(src_pts_path)
        pts68 = read_pts68(src_pts_path) if has_pts else None

        # Detect face & 5 keypoints using SCRFD-2.5G-KPS
        dets = detector.detect(img, is_cropped_face=True)
        if dets:
            primary = dets[0]
            bbox = primary["bbox"]
            kps5 = primary["kps5"]
            det_score = primary["score"]
            kps_source = "scrfd_2.5g_kps"
            scrfd_direct_count += 1
        elif pts68 is not None:
            kps5 = extract_5kps_from_68(pts68)
            xmin, ymin = pts68.min(axis=0)
            xmax, ymax = pts68.max(axis=0)
            bbox = np.array([max(0, xmin), max(0, ymin), min(w, xmax), min(h, ymax)], dtype=np.float32)
            det_score = 0.85
            kps_source = "pts68_derived_5kps"
            pts_fallback_count += 1
        else:
            raise RuntimeError(f"Neither SCRFD nor .pts available for {src_img_path}")

        # Align face to 112x112 for ArcFace
        aligned_112, M = align_face_arcface(img, kps5, image_size=112)
        aligned_rel = os.path.join(identity, f"{sample_id}.jpg")
        aligned_dst = os.path.join(aligned_base, aligned_rel)
        imwrite_unicode(aligned_dst, aligned_112, quality=95)

        aligned_manifest.append({
            "sample_id": sample_id,
            "identity": identity,
            "session": "s30" if "s30" in session_tag else "s10",
            "aligned_path": aligned_rel.replace("\\", "/"),
            "raw_backup_path": os.path.join("datasets/raw/personal_faces", identity, f"{sample_id}.jpg").replace("\\", "/"),
            "original_source_path": src_img_path,
            "orig_width": w,
            "orig_height": h,
            "bbox_x1": round(float(bbox[0]), 2),
            "bbox_y1": round(float(bbox[1]), 2),
            "bbox_x2": round(float(bbox[2]), 2),
            "bbox_y2": round(float(bbox[3]), 2),
            "det_score": round(float(det_score), 4),
            "kps_source": kps_source,
            "near_duplicate_cluster_id": r["near_duplicate_cluster_id"],
            "laplacian_var": r["laplacian_var"],
            "has_landmarks68": has_pts,
        })

        # Step 4: Standardize 68 landmarks if available
        if has_pts and pts68 is not None:
            lm_pack = normalize_landmarks68(
                pts68,
                image_width=w,
                image_height=h,
                is_ground_truth=False,
                source_tag="auto_predicted_validated_with_scrfd",
            )
            pts_out_rel = os.path.join(identity, f"{sample_id}.pts")
            npz_out_rel = os.path.join(identity, f"{sample_id}.npz")
            write_pts68(os.path.join(lm68_base, pts_out_rel), lm_pack["coords_clipped"])
            np.savez_compressed(
                os.path.join(lm68_base, npz_out_rel),
                sample_id=sample_id,
                identity=identity,
                session="s30" if "s30" in session_tag else "s10",
                **lm_pack,
            )
            lm68_manifest.append({
                "sample_id": sample_id,
                "identity": identity,
                "session": "s30" if "s30" in session_tag else "s10",
                "pts_path": pts_out_rel.replace("\\", "/"),
                "npz_path": npz_out_rel.replace("\\", "/"),
                "image_path": os.path.join("datasets/raw/personal_faces", identity, f"{sample_id}.jpg").replace("\\", "/"),
                "image_width": w,
                "image_height": h,
                "interocular_outer_px": round(float(lm_pack["interocular_outer"]), 2),
                "interocular_centers_px": round(float(lm_pack["interocular_centers"]), 2),
                "orig_oob_count": lm_pack["oob_count"],
                "coord_system": lm_pack["coord_system"],
                "is_ground_truth": False,
                "source_tag": lm_pack["source_tag"],
            })
            consolidated_lm68.append((sample_id, identity, "s30" if "s30" in session_tag else "s10", lm_pack))

    # Write aligned_faces_manifest.csv
    aligned_csv_path = os.path.join(aligned_base, "aligned_faces_manifest.csv")
    with open(aligned_csv_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(aligned_manifest[0].keys()))
        w.writeheader()
        w.writerows(aligned_manifest)

    # Write landmarks68_manifest.csv and consolidated personal_landmarks68.npz
    lm68_csv_path = os.path.join(lm68_base, "landmarks68_manifest.csv")
    with open(lm68_csv_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(lm68_manifest[0].keys()))
        w.writeheader()
        w.writerows(lm68_manifest)

    np.savez_compressed(
        os.path.join(lm68_base, "personal_landmarks68_all.npz"),
        sample_ids=np.array([x[0] for x in consolidated_lm68]),
        identities=np.array([x[1] for x in consolidated_lm68]),
        sessions=np.array([x[2] for x in consolidated_lm68]),
        coords_pixel=np.stack([x[3]["coords_pixel"] for x in consolidated_lm68], axis=0),
        coords_clipped=np.stack([x[3]["coords_clipped"] for x in consolidated_lm68], axis=0),
        coords_normalized=np.stack([x[3]["coords_normalized"] for x in consolidated_lm68], axis=0),
        unit_centered_shapes=np.stack([x[3]["unit_centered_shape"] for x in consolidated_lm68], axis=0),
        image_sizes_wh=np.stack([x[3]["image_size_wh"] for x in consolidated_lm68], axis=0),
        is_ground_truth=np.zeros((len(consolidated_lm68),), dtype=bool),
        coord_system="pixel_top_left_origin_xy",
    )

    # Step 3.4: Process the 7 multi-face group images in D:\data_AAM\*.jpg for identity verification
    multi_rows = [
        r for r in quality_rows
        if r["dataset_group"] == "data_AAM/root_images" and r["quality_status"] != "FAIL_REVIEW"
    ]
    multi_dir = os.path.join(aligned_base, "multi_face_pending_verification")
    os.makedirs(multi_dir, exist_ok=True)
    multi_manifest = []

    for idx_img, r in enumerate(multi_rows, start=1):
        src_path = r["file_path"]
        img = imread_unicode(src_path)
        vis = img.copy()
        dets = detector.detect(img, is_cropped_face=False, conf_thresh=0.45)
        # Sort left-to-right by bbox x1 so numbering #0, #1, #2... is intuitive for the user!
        dets.sort(key=lambda d: float(d["bbox"][0]))

        short_id = f"group_img_{idx_img:02d}"
        for f_idx, d in enumerate(dets):
            b = d["bbox"].astype(int)
            kps5 = d["kps5"]
            score = d["score"]
            fw, fh = int(b[2] - b[0]), int(b[3] - b[1])

            aligned_crop, _ = align_face_arcface(img, kps5, image_size=112)
            crop_name = f"{short_id}_face_{f_idx:02d}.jpg"
            imwrite_unicode(os.path.join(multi_dir, crop_name), aligned_crop, quality=95)

            cv2.rectangle(vis, (b[0], b[1]), (b[2], b[3]), (0, 255, 0), 2)
            label = f"#{f_idx} ({fw}x{fh})"
            cv2.putText(
                vis,
                label,
                (max(5, b[0]), max(22, b[1] - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
            )

            multi_manifest.append({
                "group_image_id": short_id,
                "original_file_name": r["file_name"],
                "face_index_left_to_right": f_idx,
                "aligned_crop_file": crop_name,
                "bbox_x1": int(b[0]),
                "bbox_y1": int(b[1]),
                "bbox_x2": int(b[2]),
                "bbox_y2": int(b[3]),
                "face_width_px": fw,
                "face_height_px": fh,
                "scrfd_score": round(float(score), 4),
                "verified_identity": "PENDING_USER_VERIFICATION",
            })

        imwrite_unicode(os.path.join(multi_dir, f"{short_id}_annotated.jpg"), vis, quality=92)

    multi_csv = os.path.join(multi_dir, "multi_face_verification_manifest.csv")
    with open(multi_csv, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(multi_manifest[0].keys()))
        w.writeheader()
        w.writerows(multi_manifest)

    print("=== STEP 3 & STEP 4 SUMMARY ===")
    print(f"Total single-person aligned 112x112 faces: {len(aligned_manifest)}")
    print(f"  - SCRFD-2.5G-KPS direct 5-keypoint detections: {scrfd_direct_count}")
    print(f"  - .pts derived 5-keypoint fallback (extreme close crop): {pts_fallback_count}")
    for ident in ["Bang", "Lap", "Thao", "Thuan"]:
        c_s30 = sum(1 for x in aligned_manifest if x["identity"] == ident and x["session"] == "s30")
        c_s10 = sum(1 for x in aligned_manifest if x["identity"] == ident and x["session"] == "s10")
        c_lm = sum(1 for x in lm68_manifest if x["identity"] == ident)
        print(f"  Identity {ident:6s}: Aligned={c_s30 + c_s10:3d} (s30={c_s30:3d}, s10={c_s10:2d}) | Standardized 68-landmarks={c_lm:3d}")
    print(f"Multi-face group images processed: {len(multi_rows)} images -> {len(multi_manifest)} detected faces queued for verification.")
    return aligned_manifest, lm68_manifest, multi_manifest


if __name__ == "__main__":
    run_step3_and_step4()
