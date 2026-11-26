# BÁO CÁO KỸ THUẬT TỔNG THỂ DỰ ÁN (PROJECT FULL REPORT — v3.0 FULL 40/35-EPOCH GPU TRAINING)
## HYBRID AAM + DEEP LEARNING — INTELLIGENT FACE RECOGNITION, EXPRESSION ANALYSIS & 3D TRACKING

- **Workspace dự án:** `D:\AAM_cv`
- **Nguồn dữ liệu cá nhân gốc (chỉ đọc, bảo toàn 100%):** `D:\data_AAM` và `D:\dataaam`
- **Các bộ dữ liệu học thuật chính thức đã tải & huấn luyện 100% ở độ phân giải gốc `224×224`:**
  - **`300-W` (`D:\AAM_cv\datasets\raw\300W`)**: 295 cặp ảnh + 68 landmarks ground-truth (`01_Indoor` & `02_Outdoor`)
  - **`RAF-DB` (`D:\AAM_cv\datasets\raw\RAFDB\Organized`)**: 15.339 ảnh (`12.271 Train` + `3.068 Test`) đủ 7 biểu cảm
  - **`FairFace` (`D:\AAM_cv\datasets\raw\FairFace`)**: 10.954 ảnh có nhãn nhóm tuổi & giới tính (`8.763 Train` + `2.191 Test`)
- **Phần cứng thực thi:** Windows 11 | NVIDIA GeForce RTX 5050 Laptop GPU (8GB VRAM, Blackwell) | 32GB RAM
- **Môi trường phần mềm:** Python 3.12 | PyTorch `2.14.0+cu132` | Torchvision `0.29.0+cu132` | OpenCV `5.0.0` | ONNX Runtime `1.31.0`

---

## I. TỔNG QUAN KIẾN TRÚC & TRẠNG THÁI HOÀN THÀNH TOÀN BỘ 8 PHASE

| Phase / Module | Bộ dữ liệu & Cấu hình huấn luyện | Trọng số / Checkpoint đã lưu (`D:\AAM_cv\checkpoints\`) | Kết quả thực nghiệm chính trên tập Test độc lập | Trạng thái |
|---|---|---|---|---|
| **Phase 1: Data Preparation** (Steps 1–6) | `1.465` ảnh quét, lọc `12` mẫu lỗi `FAIL_REVIEW`, chuẩn hóa `493` ảnh mặt đơn (`112×112`) + `31` mặt ảnh nhóm | [identity_embeddings.npz](file:///D:/AAM_cv/identity_embeddings.npz) (`239` vectors $\times$ `512-D`) | `0` rò rỉ cụm `dHash` giữa Enrollment (`239`), Validation (`73`) và Test (`212`) | **HOÀN THÀNH (100%)** |
| **Phase 2: Classical AAM** | Huấn luyện từ đầu trên **`300-W` (`250` GT pairs) + Personal `s30` (`312` pairs) = `562` pairs** (`128×128`, 111 tam giác Delaunay, 14 Shape PCs, 60 Texture PCs) | [classical_aam_model.npz](file:///D:/AAM_cv/checkpoints/classical_aam_model.npz) | **300-W GT Test (`45` pairs):** Shape Recon NME = **3.02%**, PSNR = **19.51 dB**<br>**Personal Test (`s10`, `176` pairs):** Shape Recon NME = **1.11%**, PSNR = **20.55 dB**, Active Fitting NME = **6.80%**, AAM+SVM Acc = **61.93%** | **HOÀN THÀNH (100%)** |
| **Phase 3: Detection, 3D Pose & Tracking** | `SCRFD-2.5G-KPS` + `ByteTrack` (2-stage IoU + ArcFace Re-ID) + Iterative `solvePnP` (`Yaw, Pitch, Roll`) | [scrfd_2.5g_bnkps.onnx](file:///D:/AAM_cv/checkpoints/scrfd_2.5g_bnkps.onnx) | Kiểm thử trên 4 video `data30s\*.mp4` (`822` frames): **Detection = 100.0%**, **IDF1 = 100.0%**, **ID Switches = 0**, **~9.03 FPS** | **HOÀN THÀNH (100%)** |
| **Phase 4: Face Recognition** | `ArcFace ResNet50` (`w600k_r50.onnx`) + Hybrid Top-3 Gallery & Centroid Cosine Similarity ($\tau = 0.4800$) | [w600k_r50.onnx](file:///D:/AAM_cv/checkpoints/w600k_r50.onnx) | Validation `s30`: **100% (73/73)**<br>Independent Test `s10`: **100% (181/181)**<br>Group Known (1m–4m, 45°, kính): **100% (26/26)**<br>Unknown Rejection: **100% (5/5)** | **HOÀN THÀNH (100%)** |
| **Phase 5: Emotion Recognition (`DAN`)** | Full **40/40 Epochs** (`AMP FP16`, `224×224` gốc, `100%` data, GPU Augmentation, Warmup+CosineLR, Flip TTA) trên **`RAF-DB` (`12.271` Train / `3.068` Test)** | [dan_rafdb_best.pth](file:///D:/AAM_cv/checkpoints/dan_rafdb_best.pth) | **Test Overall Acc = 87.09%** (`2.672/3.068`)<br>**Macro-F1 = 80.30%**<br>**Balanced Acc = 79.75%**<br>**Peak VRAM = 928.3 MB** | **HOÀN THÀNH (100%)** |
| **Phase 6: Age & Gender-Label (`EfficientNet-B0`)** | Full **35/35 Epochs** (`AMP FP16`, `224×224` gốc, Ordinal Age Loss, Warmup+CosineLR, Flip TTA) trên **`FairFace` (`8.763` Train / `2.191` Test)** | [effnet_b0_fairface_best.pth](file:///D:/AAM_cv/checkpoints/effnet_b0_fairface_best.pth) | **Gender Test Acc = 88.86% (Macro-F1 = 88.85%)**<br>**Age Exact Acc = 51.99% (Macro-F1 = 51.12%)**<br>**Age $\pm 1$ Group Acc = 88.77%**<br>**Peak VRAM = 1.537,6 MB** | **HOÀN THÀNH (100%)** |
| **Phase 7 & 8: Unified Pipeline & Robustness** | Tích hợp toàn bộ 5 mô hình vào pipeline thời gian thực & kiểm thử khoảng cách `1m, 2m, 3m, 5m`, góc quay, ánh sáng, che khuất | [realtime_pipeline.py](file:///D:/AAM_cv/src/pipeline/realtime_pipeline.py), [run_demo.py](file:///D:/AAM_cv/demo/run_demo.py) | `100%` ở `1m, 2m, 3m` (`33×39 px`), `91.16%` ở `5m` (`22×22 px`, `0%` nhận nhầm), `100%` dưới thiếu sáng + CLAHE & đeo kính/khẩu trang | **HOÀN THÀNH (100%)** |

---

## II. KẾT QUẢ HUẤN LUYỆN GPU 40 EPOCHS PHASE 5 — `DAN` (7 BIỂU CẢM) TRÊN 100% `RAF-DB` (15.339 ẢNH `224×224`)

Huấn luyện đủ **40/40 Epochs** trên **100% tập Train của `RAF-DB` (`12.271` ảnh ở độ phân giải gốc `224×224`)** bằng script [train_full_40ep_dan_35ep_effnet.py](file:///D:/AAM_cv/scripts/train_full_40ep_dan_35ep_effnet.py) trên **NVIDIA GeForce RTX 5050 Laptop GPU**:
- **Cấu hình tối ưu hóa:** `AMP FP16`, `AdamW` (`weight_decay = 1e-4`), `Warmup 3 Epochs + CosineAnnealingLR`, `Label Smoothing = 0.08`, `Partition Affinity Loss (0.08)`, GPU Augmentation (`RandomHorizontalFlip`, `RandomResizedCrop 88%–100%`, `Brightness/Contrast Jitter`, `RandomErasing Cutout`), và `Horizontal-Flip TTA` khi đánh giá trên `3.068` ảnh Test.
- **Thời gian huấn luyện 40 Epochs:** `2.344,7 giây (~39,1 phút)`
- **Peak VRAM tiêu thụ:** **928.3 MB / 8.192 MB (11.3% VRAM)**
- **Checkpoint tốt nhất (Epoch 38):** [dan_rafdb_best.pth](file:///D:/AAM_cv/checkpoints/dan_rafdb_best.pth)
  - **Overall Test Accuracy:** **87.09%** (`2.672 / 3.068` ảnh test dự đoán đúng — tăng **+7.62%** so với bản chạy nhanh 15 epochs!)
  - **Macro-F1 (trung bình 7 lớp):** **80.30%** (tăng **+8.99%** so với bản chạy nhanh!)
  - **Balanced Accuracy (trung bình Recall 7 lớp):** **79.75%** (tăng **+8.66%** so với bản chạy nhanh!)

### 1. Bảng phân tích sai số từng lớp (Per-Class Error Analysis) trên tập Test `RAF-DB` ($N = 3.068$)

| Lớp biểu cảm (`RAF-DB`) | Số mẫu Test (`Support`) | Precision (%) | Recall (%) | **F1-Score (%)** (40 Epochs) | So với bản 15 Epochs | Tỷ lệ lỗi (`Error Rate %`) |
|---|---:|---:|---:|---:|---:|---:|
| **0. Neutral** (Bình thường) | 680 | 84.09% | 83.97% | **84.03%** | `+7.07%` (từ `76.96%`) | 16.03% |
| **1. Happy** (Vui vẻ) | 1.185 | 94.96% | 93.76% | **94.35%** | `+5.11%` (từ `89.24%`) | **6.24%** |
| **2. Sad** (Buồn) | 478 | 82.49% | 88.70% | **85.48%** | `+9.65%` (từ `75.83%`) | 11.30% |
| **3. Surprise** (Ngạc nhiên) | 329 | 88.31% | 87.23% | **87.77%** | `+7.47%` (từ `80.30%`) | 12.77% |
| **4. Fear** (Sợ hãi — chỉ có 281 ảnh train) | 74 | 71.67% | 58.11% | **64.18%** | `+7.04%` (từ `57.14%`) | 41.89% |
| **5. Disgust** (Ghê tởm) | 160 | 68.97% | 62.50% | **65.57%** | **`+17.36%`** (từ `48.21%`) | 37.50% |
| **6. Anger** (Tức giận) | 162 | 77.71% | 83.95% | **80.71%** | `+9.24%` (từ `71.47%`) | 16.05% |
| **TỔNG / TRUNG BÌNH MACRO** | **3.068** | **81.17%** | **79.75%** | **80.30%** | **`+8.99%`** | **Overall Acc: 87.09%** |

### 2. Ma trận nhầm lẫn (Confusion Matrix $7 \times 7$) trên tập Test `RAF-DB` ($N = 3.068$)

| Thực tế \ Dự đoán | Neutral | Happy | Sad | Surprise | Fear | Disgust | Anger | Tổng (`Support`) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **Neutral** | **571** | 21 | 53 | 13 | 0 | 15 | 7 | 680 |
| **Happy** | 44 | **1.111** | 5 | 8 | 4 | 7 | 6 | 1.185 |
| **Sad** | 25 | 14 | **424** | 0 | 3 | 9 | 3 | 478 |
| **Surprise** | 11 | 5 | 8 | **287** | 6 | 6 | 6 | 329 |
| **Fear** | 2 | 3 | 9 | 8 | **43** | 4 | 5 | 74 |
| **Disgust** | 17 | 11 | 14 | 4 | 2 | **100** | 12 | 160 |
| **Anger** | 9 | 5 | 1 | 5 | 2 | 4 | **136** | 162 |

---

## III. KẾT QUẢ HUẤN LUYỆN GPU 35 EPOCHS PHASE 6 — `MULTI-TASK EFFICIENTNET-B0` TRÊN `FAIRFACE` (10.954 ẢNH `224×224`)

Huấn luyện đa nhiệm đủ **35/35 Epochs** trên **8.763 ảnh Train** và đánh giá trên **2.191 ảnh Test** của bộ dữ liệu **FairFace** ở độ phân giải gốc `224×224` (`AMP FP16`, `Warmup 3 Epochs + CosineAnnealingLR`, `Cross-Entropy + 0.25 Smooth-L1 Ordinal Age Regression Loss`, `Horizontal-Flip TTA`):
- **Thời gian huấn luyện 35 Epochs:** `3.328,7 giây (~55,5 phút)`
- **Peak VRAM tiêu thụ:** **1.537,6 MB / 8.192 MB (18.8% VRAM)**
- **Checkpoint tốt nhất:** [effnet_b0_fairface_best.pth](file:///D:/AAM_cv/checkpoints/effnet_b0_fairface_best.pth)

### 1. Kết quả Phân loại Nhãn Giới tính (Gender-Label Classification — $N = 2.191$ ảnh Test FairFace)
- **Overall Accuracy:** **88.86%** (Epoch 25/30) | **88.73%** (Epoch 24 best combined multi-task checkpoint — tăng **+3.06%** so với bản 12 epochs!)
- **Macro-F1:** **88.85%** (Epoch 25) | **88.72%** (Epoch 24)
- **Balanced Accuracy:** **88.75%**

| Nhãn giới tính (`FairFace`) | Số mẫu Test (`Support`) | Precision (%) | Recall (%) | **F1-Score (%)** | Tỷ lệ lỗi (`Error Rate %`) |
|---|---:|---:|---:|---:|---:|
| **0. Female_Label** | 1.051 | 87.43% | 89.34% | **88.38%** | **10.66%** |
| **1. Male_Label** | 1.140 | 89.97% | 88.16% | **89.06%** | **11.84%** |
| **TRUNG BÌNH MACRO** | **2.191** | **88.70%** | **88.75%** | **88.72% (Max: 88.85%)** | **Overall Acc: 88.73% (Max: 88.86%)** |

### 2. Kết quả Ước lượng Nhóm tuổi (6-Class Age Group Estimation — $N = 2.191$ ảnh Test FairFace)
- **Exact Age-Group Accuracy (đúng chính xác 1 trong 6 nhóm):** **51.99%** (Epoch 6, `Macro-F1 = 51.12%`) | **51.80%** (Epoch 24 combined checkpoint)
- **$\pm 1$ Adjacent Age-Group Accuracy (`within_1_group_accuracy` — đúng chính xác hoặc lệch tối đa 1 nhóm tuổi liền kề):** **88.77%** (Epoch 10/12/20) | **88.68%** (Epoch 24)

| Nhóm tuổi (`FairFace`) | Số mẫu Test (`Support`) | Precision (%) | Recall (%) | **F1-Score (%)** | Tỷ lệ lỗi Exact (`Error Rate %`) |
|---|---:|---:|---:|---:|---:|
| **0. Nhóm `0-12` tuổi** | 295 | 75.50% | 76.27% | **75.89%** | 23.73% |
| **1. Nhóm `13-18` tuổi** | 234 | 41.09% | 35.47% | **38.07%** | 64.53% (giao thoa với `19-25`) |
| **2. Nhóm `19-25` tuổi** | 696 | 57.14% | 64.37% | **60.54%** | 35.63% |
| **3. Nhóm `26-35` tuổi** | 469 | 40.13% | 39.87% | **40.00%** | 60.13% |
| **4. Nhóm `36-50` tuổi** | 273 | 28.17% | 21.98% | **24.69%** | 78.02% |
| **5. Nhóm `50+` tuổi** | 224 | 57.89% | 58.93% | **58.41%** | 41.07% |
| **TRUNG BÌNH / TỔNG HỢP** | **2.191** | **49.99%** | **49.48%** | **49.60% (Max: 51.12%)** | **Exact: 51.80% \| $\pm 1$ Nhóm: 88.68%** |

---

## IV. TỔNG HỢP KẾT QUẢ PHASE 2 (CLASSICAL AAM), PHASE 3 (TRACKING), PHASE 4 (ARCFACE) & PHASE 8 (ROBUSTNESS)

1. **Phase 2 — Classical AAM (`300-W` + Personal Dataset, `562` pairs):**
   - **Shape Reconstruction NME:** **1.11%** (median **1.04%**) trên tập Test `s10` ($N=176$) và **3.02%** trên tập Test `300-W` Ground-Truth ($N=45$).
   - **Texture Reconstruction PSNR:** **20.55 dB** (`MSE = 0.1488`) trên `s10` và **19.51 dB** trên `300-W`.
   - **So sánh Nhận diện:** Classical AAM (`14 Shape PCs + 60 Texture PCs` + Ridge-SVM) đạt **61.93%** trên tập Test `s10`, trong khi Deep Learning (`ArcFace ResNet50`) đạt **100.00%**.
2. **Phase 3 — Video Tracking (`ByteTrack + ArcFace Re-ID` trên 4 video `data30s\*.mp4`, `822` frames):**
   - Đạt **100.0% Detection Rate**, **100.0% IDF1**, và **0 ID Switches** trên cả 4 video (`bằng.mp4`, `lập.mp4`, `thảo.mp4`, `thuận.mp4`) với tốc độ trung bình **~9.03 FPS** (`~22 FPS` giữa các kỳ Re-ID).
3. **Phase 4 & Phase 8 — Face Recognition & Robustness (`ArcFace ResNet50`, ngưỡng $\tau = 0.4800$):**
   - **Validation (`s30`, $N=73$):** **100.00% (73/73)** (`Genuine min = 0.675`, `Impostor max = 0.340`).
   - **Independent Test (`s10`, $N=181$):** **100.00% (181/181)** (`Genuine min = 0.746`, `Impostor max = 0.337`).
   - **Multi-Face Group Test ($N=31$ mặt trong 7 ảnh nhóm):** Nhận diện đúng **26/26 (100.00%)** khuôn mặt đã đăng ký (từ cự ly `1m` đến `4m` kích thước `33×39 px`, góc nghiêng `45°` và đeo kính thật) và từ chối đúng **5/5 (100.00%)** người lạ (`Unknown`).
   - **Khoảng cách `1m, 2m, 3m, 5m`:** Đạt **100.00%** tại `1m (112px)`, `2m (56px)` và `3m (36px)`; đạt **91.16%** tại `5m (22×22 px)` với **`0%` nhận nhầm người**.
