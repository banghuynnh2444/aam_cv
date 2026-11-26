"""Real-Time Webcam / Video / Image Demo Entry Point for Hybrid AAM + Deep Learning System
(Hiển thị đầy đủ tiếng Việt: Tên, Giới tính, Tuổi, Biểu cảm, Góc quay đầu;
Tuổi và Giới tính được chốt cố định mỗi 10 giây/lần để chống nhảy số trên Webcam).

Usage:
    python demo/run_demo.py --source 0                                  # Live Webcam
    python demo/run_demo.py --source "D:/data_AAM/data30s/lập.mp4"      # Video File
    python demo/run_demo.py --source "D:/data_AAM/1793868205371_...jpg" # Static Image
"""
import os
import sys
import argparse
from pathlib import Path
import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.pipeline.realtime_pipeline import HybridFaceAnalysisPipeline


def main():
    parser = argparse.ArgumentParser(description="Hybrid AAM + Deep Learning Real-Time Face Analysis Demo (Tiếng Việt)")
    parser.add_argument("--source", default="0", help="Webcam index (0) hoặc đường dẫn video/ảnh")
    parser.add_argument("--out", default="", help="Đường dẫn lưu ảnh/video kết quả")
    parser.add_argument("--attr-interval", type=float, default=10.0, help="Chu kỳ tính lại Tuổi & Giới tính (giây, mặc định 10s)")
    parser.add_argument("--max-frames", type=int, default=0, help="Số frame tối đa (0 = không giới hạn)")
    parser.add_argument("--no-display", action="store_true", help="Chạy ẩn không mở cửa sổ cv2.imshow")
    parser.add_argument("--cpu", action="store_true", help="Run all models on CPU")
    args = parser.parse_args()

    pipeline = HybridFaceAnalysisPipeline(
        attr_refresh_interval_sec=args.attr_interval, use_cuda=not args.cpu
    )

    if os.path.isfile(args.source) and args.source.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
        img = cv2.imdecode(np.fromfile(args.source, dtype=np.uint8), cv2.IMREAD_COLOR)
        res = pipeline.process_frame(img, use_tracking=False, draw_overlay=True)
        print(f"Đã phân tích ảnh {args.source}: {len(res['faces'])} khuôn mặt ({res['latency_ms']:.1f} ms)")
        for f in res["faces"]:
            attr = f["attributes"]
            expr = f["expression"]
            print(
                f"  [#{f['track_id']}] Tên: {f['identity_name_vn']} (Độ tin cậy: {f['similarity']:.3f}) | "
                f"Giới tính: {attr['gender_vn']} | Tuổi: {attr['age_display_vn']} | "
                f"Biểu cảm: {expr['predicted_expression_vn']}"
            )
        if args.out:
            os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
            cv2.imencode(".jpg", res["vis_frame"], [cv2.IMWRITE_JPEG_QUALITY, 94])[1].tofile(args.out)
            print("Đã lưu ảnh kết quả tại:", args.out)
        if not args.no_display:
            cv2.imshow("Hybrid AAM + Deep Learning Face Analysis (Tieng Viet)", res["vis_frame"])
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        return

    src = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise RuntimeError(f"Không thể mở nguồn video/webcam: {args.source}")

    writer = None
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        res = pipeline.process_frame(frame, use_tracking=True, draw_overlay=True)
        vis = res["vis_frame"]

        if args.out and writer is None:
            os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
            h, w = vis.shape[:2]
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        if writer is not None:
            writer.write(vis)

        if not args.no_display:
            cv2.imshow("Hybrid AAM + Deep Learning Face Analysis (Tieng Viet)", vis)
            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                break

        frame_idx += 1
        if args.max_frames > 0 and frame_idx >= args.max_frames:
            break

    cap.release()
    if writer is not None:
        writer.release()
    if not args.no_display:
        cv2.destroyAllWindows()
    print(f"Đã xử lý xong {frame_idx} frames.")


if __name__ == "__main__":
    main()
