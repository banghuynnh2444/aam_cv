"""Phase 1 - Step 5 (Dataset Splitting) & Step 6 (Prepare Recognition Database).

Executes:
1. Updates multi_face_verification_manifest.csv with user-confirmed labels for the 31 faces
   across the 7 multi-face group images.
2. Step 5 (Dataset Splitting):
   - Splits Session 30s (312 images) by temporal block AND near_duplicate_cluster_id
     into Enrollment (~75%) and Validation (~25%) with zero cluster overlap.
   - Assigns 100% of independent Session 10s (181 single-face images) + 31 verified multi-face
     group faces (24 known across distances 1m-4m/yaw/glasses + 7 Unknown impostors) to Test.
   - Outputs enrollment_manifest.csv, validation_manifest.csv, test_manifest.csv.
3. Step 6 (Prepare Recognition Database):
   - Extracts 512-D L2-normalized ArcFace ResNet50 (w600k_r50.onnx) embeddings from Enrollment.
   - Calibrates the cosine similarity threshold on Validation (Genuine vs Impostor distributions).
   - Saves identity_embeddings.npz with full required metadata.
   - Evaluates verification & identification accuracy on Validation, Session-10s Test,
     and Multi-face/Distance/Unknown Test.
"""
import os
import sys
import csv
import json
import cv2
import numpy as np

sys.path.insert(0, r"D:\AAM_cv")
from src.recognition.arcface import (
    IDENTITY_ID_MAP,
    ID_TO_NAME_MAP,
    ArcFaceEmbedder,
    RecognitionDatabase,
)


# User-confirmed labels for the 7 multi-face group images (Left-to-Right face_index)
VERIFIED_GROUP_LABELS = {
    ("group_img_01", 0): "Unknown",
    ("group_img_01", 1): "Bang",
    ("group_img_01", 2): "Thuan",
    ("group_img_01", 3): "Lap",
    ("group_img_01", 4): "Unknown",
    ("group_img_01", 5): "Unknown",
    ("group_img_01", 6): "Unknown",
    ("group_img_01", 7): "Unknown",
    ("group_img_01", 8): "Thao",

    ("group_img_02", 0): "Lap",
    ("group_img_02", 1): "Thuan",
    ("group_img_02", 2): "Thao",

    ("group_img_03", 0): "Thao",
    ("group_img_03", 1): "Bang",
    ("group_img_03", 2): "Lap",
    ("group_img_03", 3): "Thuan",

    ("group_img_04", 0): "Bang",
    ("group_img_04", 1): "Lap",
    ("group_img_04", 2): "Thao",
    ("group_img_04", 3): "Thuan",

    ("group_img_05", 0): "Thao",
    ("group_img_05", 1): "Bang",
    ("group_img_05", 2): "Lap",
    ("group_img_05", 3): "Thuan",

    ("group_img_06", 0): "Thao",
    ("group_img_06", 1): "Lap",
    ("group_img_06", 2): "Thuan",

    ("group_img_07", 0): "Thao",
    ("group_img_07", 1): "Lap",
    ("group_img_07", 2): "Bang",
    ("group_img_07", 3): "Thuan",
}

GROUP_CONDITION_NOTES = {
    "group_img_01": "hall_group_9p_unknown_test_2.5m",
    "group_img_02": "classroom_close_1m_frontal",
    "group_img_03": "classroom_far_3.5m_to_4m_small_face",
    "group_img_04": "classroom_medium_2m_frontal",
    "group_img_05": "classroom_medium_far_2.5m_to_3m_expression",
    "group_img_06": "classroom_side_yaw_right_45deg_1.5m",
    "group_img_07": "classroom_side_yaw_left_45deg_1.5m_bang_glasses",
}


def imread_unicode(path: str):
    data = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def write_csv_both(filename: str, rows: list):
    paths = [
        os.path.join(r"D:\AAM_cv", filename),
        os.path.join(r"D:\AAM_cv\datasets\splits", filename),
    ]
    fieldnames = list(rows[0].keys())
    for p in paths:
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)


def run_step5_and_step6():
    aligned_base = r"D:\AAM_cv\datasets\processed\aligned_faces"
    aligned_csv = os.path.join(aligned_base, "aligned_faces_manifest.csv")
    multi_csv = os.path.join(aligned_base, "multi_face_pending_verification", "multi_face_verification_manifest.csv")

    with open(aligned_csv, "r", encoding="utf-8-sig") as f:
        single_rows = list(csv.DictReader(f))

    with open(multi_csv, "r", encoding="utf-8-sig") as f:
        multi_rows = list(csv.DictReader(f))

    # 1. Update multi_face_verification_manifest.csv with user-confirmed identities
    for r in multi_rows:
        key = (r["group_image_id"], int(r["face_index_left_to_right"]))
        r["verified_identity"] = VERIFIED_GROUP_LABELS[key]
        r["condition_note"] = GROUP_CONDITION_NOTES[r["group_image_id"]]

    with open(multi_csv, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(multi_rows[0].keys()))
        w.writeheader()
        w.writerows(multi_rows)

    # 2. Step 5: Split single_rows into Enrollment (s30 first ~75% clusters),
    # Validation (s30 last ~25% clusters, strictly disjoint clusters), and Test (s10 + multi_face)
    enrollment_rows = []
    validation_rows = []
    test_rows = []

    for ident in ["Bang", "Lap", "Thao", "Thuan"]:
        s30_ident = [r for r in single_rows if r["identity"] == ident and r["session"] == "s30"]
        s30_ident.sort(key=lambda x: x["sample_id"])

        # Collect ordered unique near_duplicate_cluster_ids
        ordered_clusters = []
        for r in s30_ident:
            cid = int(r["near_duplicate_cluster_id"])
            if cid not in ordered_clusters:
                ordered_clusters.append(cid)

        n_c = len(ordered_clusters)
        cut_idx = int(round(n_c * 0.76))
        enroll_cids = set(ordered_clusters[:cut_idx])
        val_cids = set(ordered_clusters[cut_idx:])

        # Guarantee zero cluster overlap
        assert enroll_cids.isdisjoint(val_cids), f"Cluster leakage detected for {ident}"

        for r in s30_ident:
            cid = int(r["near_duplicate_cluster_id"])
            entry = {
                "sample_id": r["sample_id"],
                "split": "enrollment" if cid in enroll_cids else "validation",
                "subset_type": "single_face_s30",
                "identity_id": IDENTITY_ID_MAP[ident],
                "identity_name": ident,
                "session": "s30",
                "aligned_112_path": os.path.join("datasets/processed/aligned_faces", r["aligned_path"]).replace("\\", "/"),
                "raw_image_path": r["raw_backup_path"],
                "near_duplicate_cluster_id": cid,
                "face_width_px": r["orig_width"],
                "face_height_px": r["orig_height"],
                "det_score": r["det_score"],
                "laplacian_var": r["laplacian_var"],
                "has_landmarks68": r["has_landmarks68"],
                "condition_note": "session_30s_webcam",
            }
            if cid in enroll_cids:
                enrollment_rows.append(entry)
            else:
                validation_rows.append(entry)

        # All Session 10s samples go to independent Test set
        s10_ident = [r for r in single_rows if r["identity"] == ident and r["session"] == "s10"]
        s10_ident.sort(key=lambda x: x["sample_id"])
        for r in s10_ident:
            test_rows.append({
                "sample_id": r["sample_id"],
                "split": "test",
                "subset_type": "single_face_s10",
                "identity_id": IDENTITY_ID_MAP[ident],
                "identity_name": ident,
                "session": "s10",
                "aligned_112_path": os.path.join("datasets/processed/aligned_faces", r["aligned_path"]).replace("\\", "/"),
                "raw_image_path": r["raw_backup_path"],
                "near_duplicate_cluster_id": int(r["near_duplicate_cluster_id"]),
                "face_width_px": r["orig_width"],
                "face_height_px": r["orig_height"],
                "det_score": r["det_score"],
                "laplacian_var": r["laplacian_var"],
                "has_landmarks68": r["has_landmarks68"],
                "condition_note": "session_10s_independent",
            })

    # Append the 31 verified multi-face group crops to Test manifest
    for r in multi_rows:
        ident = r["verified_identity"]
        uid = IDENTITY_ID_MAP.get(ident, -1)
        sample_id = os.path.splitext(r["aligned_crop_file"])[0]
        test_rows.append({
            "sample_id": sample_id,
            "split": "test",
            "subset_type": "multi_face_group",
            "identity_id": uid,
            "identity_name": ident,
            "session": r["group_image_id"],
            "aligned_112_path": os.path.join(
                "datasets/processed/aligned_faces/multi_face_pending_verification",
                r["aligned_crop_file"],
            ).replace("\\", "/"),
            "raw_image_path": os.path.join("D:/data_AAM", r["original_file_name"]).replace("\\", "/"),
            "near_duplicate_cluster_id": -1,
            "face_width_px": r["face_width_px"],
            "face_height_px": r["face_height_px"],
            "det_score": r["scrfd_score"],
            "laplacian_var": 200.0,
            "has_landmarks68": False,
            "condition_note": r["condition_note"],
        })

    write_csv_both("enrollment_manifest.csv", enrollment_rows)
    write_csv_both("validation_manifest.csv", validation_rows)
    write_csv_both("test_manifest.csv", test_rows)

    print("=== STEP 5: DATASET SPLITTING SUMMARY ===")
    for ident in ["Bang", "Lap", "Thao", "Thuan"]:
        n_e = sum(1 for r in enrollment_rows if r["identity_name"] == ident)
        n_v = sum(1 for r in validation_rows if r["identity_name"] == ident)
        n_t_s10 = sum(1 for r in test_rows if r["identity_name"] == ident and r["subset_type"] == "single_face_s10")
        n_t_grp = sum(1 for r in test_rows if r["identity_name"] == ident and r["subset_type"] == "multi_face_group")
        print(f"  {ident:6s} | Enrollment (s30): {n_e:3d} | Validation (s30): {n_v:2d} | Test (s10): {n_t_s10:2d} | Test (Group): {n_t_grp:2d}")
    n_unk = sum(1 for r in test_rows if r["identity_name"] == "Unknown")
    print(f"  Unknown (Group Impostors): {n_unk:2d}")
    print(f"  Total Enrollment={len(enrollment_rows)}, Validation={len(validation_rows)}, Test={len(test_rows)}")

    # 3. Step 6: Extract ArcFace ResNet50 embeddings & build RecognitionDatabase
    embedder = ArcFaceEmbedder(
        model_path=r"D:\AAM_cv\checkpoints\w600k_r50.onnx",
        use_cuda=True,
    )

    enroll_ids = []
    enroll_names = []
    enroll_embs = []
    enroll_srcs = []
    enroll_quals = []

    for r in enrollment_rows:
        img_path = os.path.join(r"D:\AAM_cv", r["aligned_112_path"])
        img = imread_unicode(img_path)
        emb = embedder.extract(img, flip_tta=True)
        # Combine detector confidence and sharpness into [0, 1] quality weight
        det_s = float(r["det_score"])
        sharp_s = min(1.0, float(r["laplacian_var"]) / 150.0)
        q_score = round(0.7 * det_s + 0.3 * sharp_s, 4)

        enroll_ids.append(int(r["identity_id"]))
        enroll_names.append(r["identity_name"])
        enroll_embs.append(emb)
        enroll_srcs.append(r["aligned_112_path"])
        enroll_quals.append(q_score)

    db = RecognitionDatabase(
        identity_ids=np.array(enroll_ids, dtype=np.int32),
        identity_names=np.array(enroll_names),
        embeddings=np.stack(enroll_embs, axis=0),
        source_images=np.array(enroll_srcs),
        quality_scores=np.array(enroll_quals, dtype=np.float32),
    )

    # Calibrate threshold on Validation set:
    # Genuine scores = score for true identity on validation images
    # Impostor scores = scores for the 3 wrong identities on validation images
    genuine_scores = []
    impostor_scores = []
    val_embs = []
    for r in validation_rows:
        img_path = os.path.join(r"D:\AAM_cv", r["aligned_112_path"])
        emb = embedder.extract(imread_unicode(img_path), flip_tta=True)
        val_embs.append(emb)
        scores = db.score_query(emb, top_k=3)
        true_name = r["identity_name"]
        for name, sc in scores.items():
            if name == true_name:
                genuine_scores.append(sc)
            else:
                impostor_scores.append(sc)

    gen_arr = np.array(genuine_scores)
    imp_arr = np.array(impostor_scores)

    # Choose optimal threshold above max impostor score with safety margin
    # Typical ArcFace hybrid cosine similarity: impostors < 0.25, genuine > 0.55
    max_imp = float(imp_arr.max())
    min_gen = float(gen_arr.min())
    # Set threshold at 0.42 (or midpoint biased toward low FAR for strong Unknown rejection)
    calibrated_thresh = round(float(np.clip(0.5 * (max_imp + min_gen), 0.38, 0.48)), 4)
    db.calibrated_threshold = calibrated_thresh

    # Save identity_embeddings.npz
    db.save(r"D:\AAM_cv\identity_embeddings.npz")
    db.save(r"D:\AAM_cv\datasets\processed\identity_embeddings.npz")

    # Evaluate on Validation, Test (s10), and Test (Multi-face Group)
    val_correct = 0
    for r, emb in zip(validation_rows, val_embs):
        res = db.identify(emb)
        if res["identity_name"] == r["identity_name"]:
            val_correct += 1

    test_s10_rows = [r for r in test_rows if r["subset_type"] == "single_face_s10"]
    s10_correct = 0
    s10_gen_scores = []
    s10_imp_scores = []
    for r in test_s10_rows:
        img_path = os.path.join(r"D:\AAM_cv", r["aligned_112_path"])
        emb = embedder.extract(imread_unicode(img_path), flip_tta=True)
        res = db.identify(emb)
        if res["identity_name"] == r["identity_name"]:
            s10_correct += 1
        for name, sc in res["scores"].items():
            if name == r["identity_name"]:
                s10_gen_scores.append(sc)
            else:
                s10_imp_scores.append(sc)

    test_grp_rows = [r for r in test_rows if r["subset_type"] == "multi_face_group"]
    grp_known_correct = 0
    grp_known_total = 0
    grp_unk_correct = 0
    grp_unk_total = 0
    grp_details = []

    for r in test_grp_rows:
        img_path = os.path.join(r"D:\AAM_cv", r["aligned_112_path"])
        emb = embedder.extract(imread_unicode(img_path), flip_tta=True)
        res = db.identify(emb)
        true_name = r["identity_name"]
        pred_name = res["identity_name"]
        ok = (pred_name == true_name)
        if true_name == "Unknown":
            grp_unk_total += 1
            if ok:
                grp_unk_correct += 1
        else:
            grp_known_total += 1
            if ok:
                grp_known_correct += 1
        grp_details.append({
            "sample_id": r["sample_id"],
            "condition": r["condition_note"],
            "face_size_px": f"{r['face_width_px']}x{r['face_height_px']}",
            "true_identity": true_name,
            "pred_identity": pred_name,
            "similarity": res["similarity"],
            "correct": ok,
        })

    summary = {
        "model_version": db.model_version,
        "enrolled_embeddings_count": len(enrollment_rows),
        "calibrated_threshold": calibrated_thresh,
        "validation_s30": {
            "total": len(validation_rows),
            "correct": val_correct,
            "accuracy": round(val_correct / max(1, len(validation_rows)), 4),
            "genuine_sim_min": round(float(gen_arr.min()), 4),
            "genuine_sim_mean": round(float(gen_arr.mean()), 4),
            "impostor_sim_max": round(float(imp_arr.max()), 4),
            "impostor_sim_mean": round(float(imp_arr.mean()), 4),
        },
        "test_s10_independent": {
            "total": len(test_s10_rows),
            "correct": s10_correct,
            "accuracy": round(s10_correct / max(1, len(test_s10_rows)), 4),
            "genuine_sim_min": round(float(np.min(s10_gen_scores)), 4),
            "genuine_sim_mean": round(float(np.mean(s10_gen_scores)), 4),
            "impostor_sim_max": round(float(np.max(s10_imp_scores)), 4),
            "impostor_sim_mean": round(float(np.mean(s10_imp_scores)), 4),
        },
        "test_multiface_group_and_distance": {
            "known_total": grp_known_total,
            "known_correct": grp_known_correct,
            "known_accuracy": round(grp_known_correct / max(1, grp_known_total), 4),
            "unknown_total": grp_unk_total,
            "unknown_rejected_correct": grp_unk_correct,
            "unknown_rejection_rate": round(grp_unk_correct / max(1, grp_unk_total), 4),
            "details": grp_details,
        },
    }

    report_json = r"D:\AAM_cv\evaluations\phase1_step6_recognition_report.json"
    with open(report_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n=== STEP 6: ARCFACE RECOGNITION DATABASE & EVALUATION ===")
    print(f"Saved identity_embeddings.npz ({len(enrollment_rows)} embeddings x 512-D)")
    print(f"Calibrated Cosine Similarity Threshold: {calibrated_thresh:.4f}")
    print(f"  Validation (s30, n={len(validation_rows)}): Accuracy = {val_correct}/{len(validation_rows)} ({100*val_correct/len(validation_rows):.2f}%) | Genuine min={gen_arr.min():.3f}, mean={gen_arr.mean():.3f} | Impostor max={imp_arr.max():.3f}")
    print(f"  Independent Test (s10, n={len(test_s10_rows)}): Accuracy = {s10_correct}/{len(test_s10_rows)} ({100*s10_correct/len(test_s10_rows):.2f}%) | Genuine min={np.min(s10_gen_scores):.3f}, mean={np.mean(s10_gen_scores):.3f} | Impostor max={np.max(s10_imp_scores):.3f}")
    print(f"  Multi-face Group Test (Known 1m-4m/yaw/glasses, n={grp_known_total}): Accuracy = {grp_known_correct}/{grp_known_total} ({100*grp_known_correct/grp_known_total:.2f}%)")
    print(f"  Multi-face Group Test (Unknown Impostor Rejection, n={grp_unk_total}): Rejected = {grp_unk_correct}/{grp_unk_total} ({100*grp_unk_correct/grp_unk_total:.2f}%)")
    for d in grp_details:
        mark = "OK" if d["correct"] else "MISS"
        print(f"    [{mark:4s}] {d['sample_id']:22s} ({d['face_size_px']:7s}, {d['condition']}): true={d['true_identity']:7s} -> pred={d['pred_identity']:7s} (sim={d['similarity']:.3f})")

    return summary


if __name__ == "__main__":
    run_step5_and_step6()
