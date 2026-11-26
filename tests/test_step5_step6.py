import os
import sys
import csv
import json
import unittest
import numpy as np

sys.path.insert(0, r"D:\AAM_cv")
from src.recognition.arcface import (
    IDENTITY_ID_MAP,
    RecognitionDatabase,
    l2_normalize,
)


class TestStep5AndStep6(unittest.TestCase):
    def test_manifests_exist_and_zero_cluster_leakage(self):
        e_path = r"D:\AAM_cv\enrollment_manifest.csv"
        v_path = r"D:\AAM_cv\validation_manifest.csv"
        t_path = r"D:\AAM_cv\test_manifest.csv"
        for p in [e_path, v_path, t_path]:
            self.assertTrue(os.path.isfile(p), f"Missing {p}")

        with open(e_path, "r", encoding="utf-8-sig") as f:
            e_rows = list(csv.DictReader(f))
        with open(v_path, "r", encoding="utf-8-sig") as f:
            v_rows = list(csv.DictReader(f))
        with open(t_path, "r", encoding="utf-8-sig") as f:
            t_rows = list(csv.DictReader(f))

        # Check counts: 312 total in s30 split between enrollment and validation
        self.assertEqual(len(e_rows) + len(v_rows), 312)
        # 181 (s10) + 31 (multi-face group) = 212 total in test
        self.assertEqual(len(t_rows), 212)

        # Verify zero near-duplicate cluster overlap between Enrollment and Validation
        e_clusters = {int(r["near_duplicate_cluster_id"]) for r in e_rows}
        v_clusters = {int(r["near_duplicate_cluster_id"]) for r in v_rows}
        self.assertTrue(e_clusters.isdisjoint(v_clusters), "Cluster leakage between Enrollment and Validation!")

        # Verify session separation: s30 in enroll/val, s10 + group in test
        self.assertTrue(all(r["session"] == "s30" for r in e_rows))
        self.assertTrue(all(r["session"] == "s30" for r in v_rows))
        self.assertTrue(all(r["session"] != "s30" for r in t_rows))

    def test_identity_embeddings_npz_structure_and_recognition(self):
        npz_path = r"D:\AAM_cv\identity_embeddings.npz"
        self.assertTrue(os.path.isfile(npz_path))
        db = RecognitionDatabase.load(npz_path)

        self.assertEqual(db.embeddings.shape[1], 512)
        self.assertEqual(db.centroids.shape, (4, 512))
        # Verify unit L2 norm
        norms = np.linalg.norm(db.embeddings, axis=1)
        np.testing.assert_allclose(norms, np.ones_like(norms), atol=1e-4)

        # Querying with each identity's centroid must return that exact identity
        for uid, name in [(0, "Bang"), (1, "Lap"), (2, "Thao"), (3, "Thuan")]:
            res = db.identify(db.centroids[uid])
            self.assertEqual(res["identity_id"], uid)
            self.assertEqual(res["identity_name"], name)
            self.assertFalse(res["is_unknown"])

        # Querying with an orthogonal/random noise vector must return Unknown
        rng = np.random.default_rng(42)
        random_vec = l2_normalize(rng.standard_normal(512).astype(np.float32))
        res_unk = db.identify(random_vec)
        self.assertEqual(res_unk["identity_id"], -1)
        self.assertEqual(res_unk["identity_name"], "Unknown")
        self.assertTrue(res_unk["is_unknown"])


if __name__ == "__main__":
    unittest.main()
