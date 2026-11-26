import os
import sys
import csv
import unittest
import cv2
import numpy as np

sys.path.insert(0, r"D:\AAM_cv")
from src.detection.scrfd import (
    SCRFDDetector,
    align_face_arcface,
    umeyama_similarity_transform,
    ARCFACE_REF_5PTS,
)
from src.geometry.landmarks68 import (
    IBUG68_REGIONS,
    MEDIAPIPE_468_TO_IBUG68_MAP,
    read_pts68,
    write_pts68,
    extract_5kps_from_68,
    compute_interocular_distance,
    normalize_landmarks68,
    map_mediapipe_to_ibug68,
)


class TestStep3AndStep4(unittest.TestCase):
    def test_umeyama_identity_and_alignment_112(self):
        M = umeyama_similarity_transform(ARCFACE_REF_5PTS, ARCFACE_REF_5PTS)
        expected = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
        np.testing.assert_allclose(M, expected, atol=1e-4)

        dummy_img = np.full((256, 256, 3), 128, dtype=np.uint8)
        aligned, M2 = align_face_arcface(dummy_img, ARCFACE_REF_5PTS * 2.0, image_size=112)
        self.assertEqual(aligned.shape, (112, 112, 3))
        self.assertEqual(M2.shape, (2, 3))

    def test_mediapipe_to_ibug68_mapping_rules(self):
        # Must map exactly 68 unique, non-random anatomical indices in [0, 467]
        self.assertEqual(len(MEDIAPIPE_468_TO_IBUG68_MAP), 68)
        self.assertEqual(len(set(MEDIAPIPE_468_TO_IBUG68_MAP)), 68)
        self.assertTrue(all(0 <= idx < 468 for idx in MEDIAPIPE_468_TO_IBUG68_MAP))

        fake_mp = np.arange(478 * 3, dtype=np.float32).reshape(478, 3)
        mapped68 = map_mediapipe_to_ibug68(fake_mp)
        self.assertEqual(mapped68.shape, (68, 3))

    def test_aligned_faces_outputs(self):
        aligned_csv = r"D:\AAM_cv\datasets\processed\aligned_faces\aligned_faces_manifest.csv"
        self.assertTrue(os.path.isfile(aligned_csv))
        with open(aligned_csv, "r", encoding="utf-8-sig") as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(len(rows), 493)
        # Spot-check first 5 aligned images are 112x112x3
        for r in rows[:5]:
            p = os.path.join(r"D:\AAM_cv\datasets\processed\aligned_faces", r["aligned_path"])
            self.assertTrue(os.path.isfile(p))
            im = cv2.imdecode(np.fromfile(p, dtype=np.uint8), cv2.IMREAD_COLOR)
            self.assertEqual(im.shape, (112, 112, 3))

    def test_landmarks68_outputs_and_metadata(self):
        lm_csv = r"D:\AAM_cv\datasets\processed\landmarks68\landmarks68_manifest.csv"
        all_npz = r"D:\AAM_cv\datasets\processed\landmarks68\personal_landmarks68_all.npz"
        self.assertTrue(os.path.isfile(lm_csv))
        self.assertTrue(os.path.isfile(all_npz))
        with open(lm_csv, "r", encoding="utf-8-sig") as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(len(rows), 488)
        # Verify auto-predicted landmarks are NOT marked as ground truth
        self.assertTrue(all(r["is_ground_truth"] == "False" for r in rows))

        data = np.load(all_npz)
        self.assertEqual(data["coords_pixel"].shape, (488, 68, 2))
        self.assertEqual(data["coords_clipped"].shape, (488, 68, 2))
        self.assertFalse(bool(data["is_ground_truth"].any()))


if __name__ == "__main__":
    unittest.main()
