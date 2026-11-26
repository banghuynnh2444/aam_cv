"""ByteTrack + ArcFace Embedding Multi-Face Tracker Module.

Implements:
- Constant-velocity Kalman state prediction for 2D face bounding boxes
- Two-stage ByteTrack association (High-score IoU + ArcFace embedding similarity,
  followed by Low-score IoU recovery for motion blur / partial occlusion)
- Embedding-based Re-Identification (ReID) for lost tracks to prevent ID switches
- Exponential Moving Average (EMA) temporal smoothing for embeddings and 68 landmarks
"""
from typing import List, Dict, Any, Optional, Tuple
import numpy as np


def bbox_iou(boxA: np.ndarray, boxB: np.ndarray) -> float:
    xA = max(float(boxA[0]), float(boxB[0]))
    yA = max(float(boxA[1]), float(boxB[1]))
    xB = min(float(boxA[2]), float(boxB[2]))
    yB = min(float(boxA[3]), float(boxB[3]))
    inter = max(0.0, xB - xA) * max(0.0, yB - yA)
    areaA = max(0.0, float(boxA[2] - boxA[0])) * max(0.0, float(boxA[3] - boxA[1]))
    areaB = max(0.0, float(boxB[2] - boxB[0])) * max(0.0, float(boxB[3] - boxB[1]))
    return float(inter / (areaA + areaB - inter + 1e-9))


class TrackedFace:
    """Represents a single tracked face trajectory with Kalman motion + EMA embedding."""

    def __init__(
        self,
        track_id: int,
        bbox: np.ndarray,
        score: float,
        embedding: Optional[np.ndarray] = None,
        kps5: Optional[np.ndarray] = None,
        pts68: Optional[np.ndarray] = None,
        identity_name: str = "Unknown",
        identity_sim: float = 0.0,
    ):
        self.track_id = int(track_id)
        self.bbox = np.asarray(bbox, dtype=np.float32).copy()
        self.velocity = np.zeros((4,), dtype=np.float32)
        self.score = float(score)
        self.embedding = self._norm(embedding) if embedding is not None else None
        self.kps5 = np.asarray(kps5, dtype=np.float32).copy() if kps5 is not None else None
        self.pts68 = np.asarray(pts68, dtype=np.float32).copy() if pts68 is not None else None
        self.identity_name = identity_name
        self.identity_sim = float(identity_sim)
        self.hits = 1
        self.age = 1
        self.time_since_update = 0

        # 10-second locked Age & Gender attributes and EMA-smoothed Emotion probabilities
        self.locked_attributes: Optional[Dict[str, Any]] = None
        self.last_attr_update_time: float = -999.0
        self.ema_age_probs: Optional[np.ndarray] = None
        self.ema_gender_probs: Optional[np.ndarray] = None
        self.ema_emotion_probs: Optional[np.ndarray] = None
        self.cached_expression: Optional[Dict[str, Any]] = None

    @staticmethod
    def _norm(v: np.ndarray) -> np.ndarray:
        v = np.asarray(v, dtype=np.float32).reshape(-1)
        return v / max(float(np.linalg.norm(v)), 1e-12)

    def predict(self) -> np.ndarray:
        self.age += 1
        self.time_since_update += 1
        self.bbox = self.bbox + 0.7 * self.velocity
        self.velocity *= 0.85
        return self.bbox

    def update(
        self,
        bbox: np.ndarray,
        score: float,
        embedding: Optional[np.ndarray] = None,
        kps5: Optional[np.ndarray] = None,
        pts68: Optional[np.ndarray] = None,
        identity_name: Optional[str] = None,
        identity_sim: Optional[float] = None,
        ema_alpha: float = 0.75,
    ) -> None:
        new_bbox = np.asarray(bbox, dtype=np.float32)
        self.velocity = 0.5 * self.velocity + 0.5 * (new_bbox - self.bbox)
        self.bbox = ema_alpha * new_bbox + (1.0 - ema_alpha) * self.bbox
        self.score = float(score)
        self.hits += 1
        self.time_since_update = 0

        if embedding is not None:
            norm_e = self._norm(embedding)
            if self.embedding is None:
                self.embedding = norm_e
            else:
                self.embedding = self._norm(0.35 * norm_e + 0.65 * self.embedding)

        if kps5 is not None:
            new_k = np.asarray(kps5, dtype=np.float32)
            self.kps5 = new_k if self.kps5 is None else (ema_alpha * new_k + (1.0 - ema_alpha) * self.kps5)

        if pts68 is not None:
            new_p = np.asarray(pts68, dtype=np.float32)
            self.pts68 = new_p if self.pts68 is None else (ema_alpha * new_p + (1.0 - ema_alpha) * self.pts68)

        if identity_name is not None:
            self.identity_name = identity_name
        if identity_sim is not None:
            self.identity_sim = float(identity_sim)


class ByteTrackFaceTracker:
    """Two-stage ByteTrack + Face Embedding Re-ID Multi-Face Tracker."""

    def __init__(
        self,
        high_thresh: float = 0.55,
        low_thresh: float = 0.25,
        iou_thresh: float = 0.25,
        reid_sim_thresh: float = 0.58,
        max_time_lost: int = 45,
    ):
        self.high_thresh = float(high_thresh)
        self.low_thresh = float(low_thresh)
        self.iou_thresh = float(iou_thresh)
        self.reid_sim_thresh = float(reid_sim_thresh)
        self.max_time_lost = int(max_time_lost)

        self.tracks: List[TrackedFace] = []
        self._next_id = 1

    def reset(self) -> None:
        self.tracks = []
        self._next_id = 1

    def _greedy_match(
        self,
        track_indices: List[int],
        dets: List[Dict[str, Any]],
        use_embedding: bool = True,
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        if not track_indices or not dets:
            return [], list(track_indices), list(range(len(dets)))

        score_mat = np.full((len(track_indices), len(dets)), -1.0, dtype=np.float32)
        for r, t_idx in enumerate(track_indices):
            trk = self.tracks[t_idx]
            for c, d in enumerate(dets):
                iou = bbox_iou(trk.bbox, d["bbox"])
                emb_sim = 0.0
                if use_embedding and trk.embedding is not None and d.get("embedding") is not None:
                    d_emb = TrackedFace._norm(d["embedding"])
                    emb_sim = float(np.dot(trk.embedding, d_emb))

                # Allow match if IoU >= iou_thresh OR strong embedding Re-ID match
                if iou >= self.iou_thresh or (use_embedding and emb_sim >= self.reid_sim_thresh):
                    score_mat[r, c] = 0.55 * iou + 0.45 * max(0.0, emb_sim)

        matches = []
        unmatched_trks = set(range(len(track_indices)))
        unmatched_dets = set(range(len(dets)))

        while unmatched_trks and unmatched_dets:
            best_val = -1.0
            best_pair = None
            for r in unmatched_trks:
                for c in unmatched_dets:
                    if score_mat[r, c] > best_val:
                        best_val = float(score_mat[r, c])
                        best_pair = (r, c)
            if best_pair is None or best_val < 0.12:
                break
            r, c = best_pair
            matches.append((track_indices[r], c))
            unmatched_trks.remove(r)
            unmatched_dets.remove(c)

        rem_trks = [track_indices[r] for r in sorted(unmatched_trks)]
        rem_dets = sorted(unmatched_dets)
        return matches, rem_trks, rem_dets

    def update(self, detections: List[Dict[str, Any]]) -> List[TrackedFace]:
        """Updates tracker with current frame detections.
        Each detection dict contains: 'bbox', 'score', and optional 'embedding', 'kps5', 'pts68', 'identity_name', 'identity_sim'.
        """
        for trk in self.tracks:
            trk.predict()

        high_dets = [d for d in detections if d["score"] >= self.high_thresh]
        low_dets = [d for d in detections if self.low_thresh <= d["score"] < self.high_thresh]

        active_indices = [i for i, t in enumerate(self.tracks) if t.time_since_update == 1]
        lost_indices = [i for i, t in enumerate(self.tracks) if 1 < t.time_since_update <= self.max_time_lost]

        # Stage 1: Match high-score detections with active tracks (IoU + Embedding)
        m1, rem_active, rem_high_idx = self._greedy_match(active_indices, high_dets, use_embedding=True)
        for t_idx, d_idx in m1:
            d = high_dets[d_idx]
            self.tracks[t_idx].update(
                d["bbox"], d["score"], d.get("embedding"), d.get("kps5"), d.get("pts68"),
                d.get("identity_name"), d.get("identity_sim"),
            )

        # Stage 2: Match low-score detections with remaining active tracks (IoU only)
        m2, rem_active2, _ = self._greedy_match(rem_active, low_dets, use_embedding=False)
        for t_idx, d_idx in m2:
            d = low_dets[d_idx]
            self.tracks[t_idx].update(
                d["bbox"], d["score"], d.get("embedding"), d.get("kps5"), d.get("pts68"),
                d.get("identity_name"), d.get("identity_sim"),
            )

        # Stage 3: Re-identify remaining high-score detections against recently lost tracks
        rem_high_dets = [high_dets[i] for i in rem_high_idx]
        m3, _, unm_high_idx = self._greedy_match(lost_indices, rem_high_dets, use_embedding=True)
        for t_idx, d_idx in m3:
            d = rem_high_dets[d_idx]
            self.tracks[t_idx].update(
                d["bbox"], d["score"], d.get("embedding"), d.get("kps5"), d.get("pts68"),
                d.get("identity_name"), d.get("identity_sim"),
            )

        # Spawn new tracks for unmatched high-confidence detections
        for d_idx in unm_high_idx:
            d = rem_high_dets[d_idx]
            new_trk = TrackedFace(
                track_id=self._next_id,
                bbox=d["bbox"],
                score=d["score"],
                embedding=d.get("embedding"),
                kps5=d.get("kps5"),
                pts68=d.get("pts68"),
                identity_name=d.get("identity_name", "Unknown"),
                identity_sim=d.get("identity_sim", 0.0),
            )
            self._next_id += 1
            self.tracks.append(new_trk)

        # Prune expired tracks
        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_time_lost]

        # Return currently visible active tracks
        return [t for t in self.tracks if t.time_since_update == 0]
