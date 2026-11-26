"""Comprehensive Unit Test Suite for All Hybrid AAM + Deep Learning Modules."""
import os
import sys
import unittest
import numpy as np
import torch

sys.path.insert(0, r"D:\AAM_cv")
from src.aam.classical_aam import ClassicalAAM, compute_nme_68
from src.detection.scrfd import SCRFDDetector, align_face_arcface, ARCFACE_REF_5PTS
from src.geometry.landmarks68 import extract_5kps_from_68, normalize_landmarks68, map_mediapipe_to_ibug68
from src.geometry.head_pose_3d import estimate_head_pose_68
from src.tracking.bytetrack_face import ByteTrackFaceTracker
from src.recognition.arcface import ArcFaceEmbedder, RecognitionDatabase
from src.emotion.emotion_dan import DANEmotionNet, DANTrainerRTX5050, analyze_expression_from_landmarks68
from src.attributes.age_gender_effnet import (
    AGE_GROUP_CLASSES,
    GENDER_LABEL_CLASSES,
    MultiTaskEfficientNetB0,
    AttributeTrainerRTX5050,
    compute_classification_metrics,
)


class TestAllSystemModules(unittest.TestCase):
    def test_classical_aam_model_loaded_and_reconstruction(self):
        ckpt = r"D:\AAM_cv\checkpoints\classical_aam_model.npz"
        self.assertTrue(os.path.isfile(ckpt))
        aam = ClassicalAAM.load(ckpt)
        self.assertEqual(aam.mean_shape_.shape, (68, 2))
        self.assertEqual(aam.texture_size, 128)
        self.assertGreater(len(aam.triangles_), 80)

        # Reconstructing the mean shape scaled to 256x256 must have near-zero NME
        test_s = aam.mean_shape_ * 120.0 + 128.0
        recon_s, b_s = aam.reconstruct_shape(test_s)
        nme = compute_nme_68(recon_s, test_s)
        self.assertLess(nme, 1e-4)

    def test_head_pose_3d_and_expression_au(self):
        ckpt = r"D:\AAM_cv\checkpoints\classical_aam_model.npz"
        aam = ClassicalAAM.load(ckpt)
        frontal_pts = aam.mean_shape_ * 180.0 + 150.0
        pose = estimate_head_pose_68(frontal_pts, 300, 300)
        self.assertIn("yaw", pose)
        self.assertIn("pitch", pose)
        self.assertIn("roll", pose)
        self.assertLess(abs(pose["yaw"]), 20.0)

        expr = analyze_expression_from_landmarks68(frontal_pts)
        self.assertIn(expr["predicted_expression"], ("Neutral", "Happy", "Sad", "Surprise", "Fear", "Disgust", "Anger"))
        self.assertEqual(expr["ground_truth_status"], "UNLABELED_NO_GROUND_TRUTH")

    def test_bytetrack_face_tracker_continuity_and_reid(self):
        tracker = ByteTrackFaceTracker()
        emb1 = np.ones((512,), dtype=np.float32)
        emb1 /= np.linalg.norm(emb1)

        # Frame 1: High confidence detection
        t1 = tracker.update([{"bbox": np.array([100, 100, 200, 220], dtype=np.float32), "score": 0.90, "embedding": emb1}])
        self.assertEqual(len(t1), 1)
        tid = t1[0].track_id

        # Frame 2: Low confidence detection (motion blur recovery via Stage 2 ByteTrack)
        t2 = tracker.update([{"bbox": np.array([104, 102, 204, 222], dtype=np.float32), "score": 0.35, "embedding": emb1}])
        self.assertEqual(len(t2), 1)
        self.assertEqual(t2[0].track_id, tid)

        # Frame 3: Brief miss (0 detections), Frame 4: Re-appearance -> must keep same Track ID!
        tracker.update([])
        t4 = tracker.update([{"bbox": np.array([115, 105, 215, 225], dtype=np.float32), "score": 0.88, "embedding": emb1}])
        self.assertEqual(len(t4), 1)
        self.assertEqual(t4[0].track_id, tid)

    def test_dan_emotion_and_effnet_attributes_amp_fp16(self):
        dan = DANEmotionNet(num_classes=7, num_heads=4, pretrained_backbone=False)
        trainer_dan = DANTrainerRTX5050(dan)
        x = torch.randn(2, 3, 224, 224)
        y = torch.tensor([0, 1], dtype=torch.long)
        res_dan = trainer_dan.train_step(x, y)
        self.assertIn("total_loss", res_dan)

        eff = MultiTaskEfficientNetB0(pretrained_backbone=False)
        trainer_eff = AttributeTrainerRTX5050(eff)
        y_age = torch.tensor([2, 3], dtype=torch.long)
        y_gen = torch.tensor([0, 1], dtype=torch.long)
        res_eff = trainer_eff.train_step(x, y_age, y_gen)
        self.assertIn("total_loss", res_eff)

        m = compute_classification_metrics(np.array([0, 1, 1, 0]), np.array([0, 1, 1, 0]), GENDER_LABEL_CLASSES)
        self.assertEqual(m["macro_f1"], 1.0)
        self.assertEqual(m["balanced_accuracy"], 1.0)

    def test_vietnamese_labels_and_10s_age_gender_lock(self):
        import cv2
        from src.pipeline.realtime_pipeline import HybridFaceAnalysisPipeline

        pipeline = HybridFaceAnalysisPipeline(aam_fit_iters=4, attr_refresh_interval_sec=10.0)
        img_path = r"D:\AAM_cv\datasets\raw\personal_faces\Bang\Bang_s30_000000.jpg"
        im = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)

        # Frame at t = 100.0s
        out1 = pipeline.process_frame(im, is_cropped_face=True, use_tracking=True, draw_overlay=True, timestamp_sec=100.0)
        self.assertEqual(len(out1["faces"]), 1)
        f1 = out1["faces"][0]
        self.assertEqual(f1["identity_name_vn"], "Bằng")
        self.assertIn(f1["attributes"]["gender_vn"], ("Nam", "Nữ"))
        self.assertIn("tuổi", f1["attributes"]["age_display_vn"])
        self.assertIn(
            f1["expression"]["predicted_expression_vn"],
            ("Bình thường", "Vui vẻ", "Buồn", "Ngạc nhiên", "Sợ hãi", "Ghê tởm", "Tức giận"),
        )

        # Modify image brightness slightly and process at t = 104.5s (within 10s lock window)
        im_shifted = np.clip(im.astype(np.int32) + 25, 0, 255).astype(np.uint8)
        out2 = pipeline.process_frame(im_shifted, is_cropped_face=True, use_tracking=True, draw_overlay=False, timestamp_sec=104.5)
        f2 = out2["faces"][0]
        # Age & Gender MUST remain 100% locked and identical, with remaining lock time ~5.5s
        self.assertEqual(f2["attributes"]["age_years"], f1["attributes"]["age_years"])
        self.assertEqual(f2["attributes"]["gender_vn"], f1["attributes"]["gender_vn"])
        self.assertAlmostEqual(f2["attributes"]["locked_remaining_sec"], 5.5, places=1)


if __name__ == "__main__":
    unittest.main()
