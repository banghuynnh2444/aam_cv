# Hybrid AAM + Deep Learning

Đồ án phân tích khuôn mặt: SCRFD, Classical AAM / SDM, ArcFace, ByteTrack,
DAN (biểu cảm) và EfficientNet-B0 (nhóm tuổi, giới tính).

## Cài trên máy khác (Windows, Python 3.12)

1. Clone repo hoặc chọn **Code → Download ZIP** trên GitHub rồi giải nén.
2. Mở PowerShell trong thư mục có `requirements.txt`:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

3. Nhận thêm file **AAM_cv_runtime.zip** từ chủ dự án và giải nén ngay tại
   thư mục repo, để có cấu trúc:

```text
checkpoints/scrfd_2.5g_bnkps.onnx
checkpoints/w600k_r50.onnx
checkpoints/classical_aam_model.npz
checkpoints/aam_cascade_sdm_v2.npz
checkpoints/dan_rafdb_best.pth
checkpoints/effnet_b0_fairface_v3.pth
identity_embeddings.npz
```

4. Chạy webcam (bấm **q** để thoát):

```powershell
.\.venv\Scripts\python.exe demo/run_demo.py --source 0 --cpu
```

Chạy ảnh hoặc video của bạn:

```powershell
.\.venv\Scripts\python.exe demo/run_demo.py --source "C:\Videos\test.mp4" --cpu
```

Có thể lưu kết quả bằng `--out outputs/result.mp4`. CPU dùng được nhưng tốc độ
phụ thuộc cấu hình máy. Khi đã cài PyTorch hỗ trợ GPU NVIDIA của máy, bỏ `--cpu`
để dùng CUDA cho DAN/EfficientNet. `onnxruntime` trong requirements dùng CPU cho
SCRFD/ArcFace.

## Tạo gói model trên máy chủ dự án

```powershell
python scripts/pack_runtime.py
```

File được tạo ở `exports/AAM_cv_runtime.zip`, gồm đúng các model dùng trong demo
và embeddings nhận diện Bằng, Lập, Thảo, Thuận. Gửi riêng cho thành viên nhóm.
Repo không chứa ảnh khuôn mặt, embeddings, dataset, model hoặc kết quả chạy.
Không cần chép bộ dữ liệu huấn luyện để chạy demo.

## Phạm vi hỗ trợ

Demo và các module trong `src/` tự tìm đường dẫn theo vị trí repo, nên không cần
đặt repo ở `D:\AAM_cv`. Các script huấn luyện, xử lý dữ liệu và test cũ vẫn có
đường dẫn của máy gốc (`D:\AAM_cv`, `D:\data_AAM`, `D:\dataaam`); muốn chạy lại
cần chuẩn bị dữ liệu và chỉnh các đường dẫn đó. Xem `PROJECT_FULL_REPORT.md`
để đọc báo cáo hiện có; các số liệu trong đó là kết quả từ máy huấn luyện gốc.
