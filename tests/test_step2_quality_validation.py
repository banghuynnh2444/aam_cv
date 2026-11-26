import os
import sys
import csv
import unittest

sys.path.insert(0, r"D:\AAM_cv\scripts")
from step2_quality_validation import (
    compute_dhash,
    hamming_dist,
    parse_pts,
    validate_dataset,
)


class TestStep2QualityValidation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = validate_dataset()

    def test_report_files_exist_and_non_empty(self):
        for p in [
            r"D:\AAM_cv\dataset_inventory.csv",
            r"D:\AAM_cv\image_quality_report.csv",
            r"D:\AAM_cv\evaluations\image_quality_report.csv",
            r"D:\AAM_cv\evaluations\flagged_images_review.csv",
        ]:
            self.assertTrue(os.path.isfile(p), f"Missing expected output: {p}")
            self.assertGreater(os.path.getsize(p), 100, f"Empty output file: {p}")

    def test_total_image_counts(self):
        # 323 (people30s) + 176 (people10s) + 181 (data10s) + 777 (dataset) + 8 (root) = 1465
        self.assertEqual(len(self.rows), 1465)
        self.assertTrue(all(r["readable"] for r in self.rows))

    def test_known_corrupted_crops_and_misaligned_pts_flagged(self):
        fail_names = {
            r["file_name"]
            for r in self.rows
            if r["dataset_group"] == "dataaam/people30s" and r["quality_status"] == "FAIL_REVIEW"
        }
        expected_fails = {
            "Lap_s30_000270.jpg",
            "Lap_s30_000280.jpg",
            "Thao_s30_000010.jpg",
            "Thao_s30_000100.jpg",
            "Thao_s30_000110.jpg",
            "Thao_s30_000140.jpg",
            "Thao_s30_000650.jpg",
            "Thuan_s30_000130.jpg",
            "Thuan_s30_000170.jpg",
            "Thuan_s30_000410.jpg",
            "Thuan_s30_000600.jpg",
        }
        self.assertEqual(fail_names, expected_fails)

    def test_original_files_untouched(self):
        # Verify none of the flagged original files were deleted
        for fn in ["Lap_s30_000270.jpg", "Thao_s30_000010.jpg", "Thuan_s30_000410.jpg"]:
            matches = [r["file_path"] for r in self.rows if r["file_name"] == fn]
            self.assertEqual(len(matches), 1)
            self.assertTrue(os.path.isfile(matches[0]))


if __name__ == "__main__":
    unittest.main()
