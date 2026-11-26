"""3D Head Pose Estimation (Yaw, Pitch, Roll) & 3D Face Mesh Rendering Module.

Implements:
- 3D-to-2D Perspective-n-Point (EPnP / Iterative PnP) head pose estimation from 68 landmarks
  producing Euler angles (Yaw, Pitch, Roll in degrees) and 3D rotation/translation vectors.
- 3D Anthropometric 68-vertex mesh model + Delaunay 3D wireframe overlay & 3D coordinate axes drawing.
"""
from typing import Dict, Any, Tuple, Optional
import cv2
import numpy as np


# Canonical 3D anthropometric reference coordinates (in mm) for 14 key iBUG-68 landmarks:
# [17, 21, 22, 26] eyebrows, [36, 39, 42, 45] eye corners, [30, 31, 35] nose, [48, 54] mouth corners, [8] chin
PNP_LANDMARK_INDICES = [17, 21, 22, 26, 36, 39, 42, 45, 30, 31, 35, 48, 54, 8]

CANONICAL_3D_14PTS = np.array([
    [-68.0,  65.0, -20.0],  # 17: Right eyebrow outer
    [-20.0,  60.0,  -5.0],  # 21: Right eyebrow inner
    [ 20.0,  60.0,  -5.0],  # 22: Left eyebrow inner
    [ 68.0,  65.0, -20.0],  # 26: Left eyebrow outer
    [-48.0,  38.0, -22.0],  # 36: Right eye outer corner
    [-18.0,  36.0, -18.0],  # 39: Right eye inner corner
    [ 18.0,  36.0, -18.0],  # 42: Left eye inner corner
    [ 48.0,  38.0, -22.0],  # 45: Left eye outer corner
    [  0.0,   0.0,   0.0],  # 30: Nose tip
    [-19.0, -12.0, -15.0],  # 31: Right nostril
    [ 19.0, -12.0, -15.0],  # 35: Left nostril
    [-31.0, -42.0, -20.0],  # 48: Right mouth corner
    [ 31.0, -42.0, -20.0],  # 54: Left mouth corner
    [  0.0, -82.0, -25.0],  # 8:  Chin bottom
], dtype=np.float64)


def estimate_head_pose_68(
    pts68: np.ndarray,
    image_width: int,
    image_height: int,
) -> Dict[str, Any]:
    """Estimates 3D Head Pose (yaw, pitch, roll in degrees) from 68 2D landmarks using solvePnP."""
    pts68 = np.asarray(pts68, dtype=np.float64).reshape(68, 2)
    image_pts = pts68[PNP_LANDMARK_INDICES]

    focal_length = float(max(image_width, image_height))
    center = (image_width / 2.0, image_height / 2.0)
    camera_matrix = np.array([
        [focal_length, 0.0, center[0]],
        [0.0, focal_length, center[1]],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)
    dist_coeffs = np.zeros((4, 1), dtype=np.float64)

    ok, rvec, tvec = cv2.solvePnP(
        CANONICAL_3D_14PTS,
        image_pts,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok:
        return {"yaw": 0.0, "pitch": 0.0, "roll": 0.0, "pose_bin": "frontal"}

    rmat, _ = cv2.Rodrigues(rvec)
    proj_matrix = np.hstack((rmat, tvec))
    _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(proj_matrix)

    pitch = float(euler_angles[0, 0])
    yaw = float(euler_angles[1, 0])
    roll = float(euler_angles[2, 0])

    # Normalize angles into [-90, 90] range
    if pitch > 90:
        pitch -= 180
    elif pitch < -90:
        pitch += 180
    if roll > 90:
        roll -= 180
    elif roll < -90:
        roll += 180
    yaw = float(np.clip(yaw, -90.0, 90.0))
    pitch = float(np.clip(pitch, -60.0, 60.0))
    roll = float(np.clip(roll, -60.0, 60.0))

    abs_yaw = abs(yaw)
    if abs_yaw < 15.0:
        pose_bin = "frontal_0_15"
    elif abs_yaw < 35.0:
        pose_bin = "moderate_yaw_15_35"
    elif abs_yaw < 55.0:
        pose_bin = "large_yaw_35_55"
    else:
        pose_bin = "extreme_yaw_55_plus"

    return {
        "yaw": round(yaw, 2),
        "pitch": round(pitch, 2),
        "roll": round(roll, 2),
        "pose_bin": pose_bin,
        "rvec": rvec,
        "tvec": tvec,
        "camera_matrix": camera_matrix,
    }


def draw_3d_face_mesh_and_pose(
    img: np.ndarray,
    pts68: np.ndarray,
    triangles: np.ndarray,
    pose_dict: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    """Renders the 68 landmarks, 3D Delaunay triangular mesh wireframe, and 3D head-pose axes."""
    canvas = img.copy()
    pts = np.asarray(pts68, dtype=np.float32).reshape(68, 2)

    # 1. Draw Delaunay triangular mesh wireframe
    for i0, i1, i2 in triangles:
        p0 = tuple(np.round(pts[i0]).astype(int))
        p1 = tuple(np.round(pts[i1]).astype(int))
        p2 = tuple(np.round(pts[i2]).astype(int))
        cv2.line(canvas, p0, p1, (255, 200, 0), 1, cv2.LINE_AA)
        cv2.line(canvas, p1, p2, (255, 200, 0), 1, cv2.LINE_AA)
        cv2.line(canvas, p2, p0, (255, 200, 0), 1, cv2.LINE_AA)

    # 2. Draw 68 landmark vertices
    for x, y in pts:
        cv2.circle(canvas, (int(round(x)), int(round(y))), 2, (0, 255, 0), -1, cv2.LINE_AA)

    # 3. Draw 3D coordinate axes at nose tip if pose_dict is provided
    if pose_dict is not None and "rvec" in pose_dict:
        axis_3d = np.array([
            [50.0, 0.0, 0.0],
            [0.0, 50.0, 0.0],
            [0.0, 0.0, 50.0],
        ], dtype=np.float64)
        proj_pts, _ = cv2.projectPoints(
            axis_3d,
            pose_dict["rvec"],
            pose_dict["tvec"],
            pose_dict["camera_matrix"],
            np.zeros((4, 1)),
        )
        nose_tip = tuple(np.round(pts[30]).astype(int))
        p_x = tuple(np.round(proj_pts[0].ravel()).astype(int))
        p_y = tuple(np.round(proj_pts[1].ravel()).astype(int))
        p_z = tuple(np.round(proj_pts[2].ravel()).astype(int))
        cv2.line(canvas, nose_tip, p_x, (0, 0, 255), 2, cv2.LINE_AA)   # X: Red
        cv2.line(canvas, nose_tip, p_y, (0, 255, 0), 2, cv2.LINE_AA)   # Y: Green
        cv2.line(canvas, nose_tip, p_z, (255, 0, 0), 2, cv2.LINE_AA)   # Z: Blue

    return canvas
