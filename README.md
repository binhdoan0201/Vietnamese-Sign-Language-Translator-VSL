# 🤟 Vietnamese Sign Language Translator (VSL)
### Hệ thống nhận diện & phiên dịch Ngôn ngữ Ký hiệu Tiếng Việt thời gian thực

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-orange.svg)](https://tensorflow.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-Holistic-green.svg)](https://developers.google.com/mediapipe)
[![Flask](https://img.shields.io/badge/Flask-SocketIO-lightgrey.svg)](https://flask.palletsprojects.com/)

---

## 📌 1. Giới thiệu

Ngôn ngữ ký hiệu là phương tiện giao tiếp chính của cộng đồng người khiếm thính, nhưng phần lớn người nghe không biết ngôn ngữ này, tạo ra một rào cản giao tiếp lớn trong học tập, công việc và đời sống.

**VSL Translator** nhận diện ký hiệu tiếng Việt từ **webcam thông thường** (không cần găng tay cảm biến hay camera chiều sâu), chuyển thành **văn bản tiếng Việt có dấu** và đọc to bằng giọng nói.

Hướng tiếp cận: trích xuất khung xương bằng **MediaPipe Holistic**, chuẩn hóa đặc trưng, rồi phân loại chuỗi **30 khung hình** bằng mạng **LSTM**.

---

## ✨ 2. Tính năng

* 🎯 **Nhận diện chuỗi chuyển động:** MediaPipe Holistic (tay, thân trên) + LSTM trên cửa sổ 30 khung hình (~3.2 giây).
* 📏 **Chuẩn hóa theo độ rộng vai:** giảm ảnh hưởng của khoảng cách người ký tới camera. *(Góc camera lệch và ánh sáng lạ vẫn là điểm yếu, xem mục 6.)*
* 🌐 **Giao diện Web:** Flask + SocketIO, hiển thị khung xương và độ tin cậy theo thời gian thực.
* 🔊 **Phát âm tiếng Việt:** Text-to-Speech (gTTS) khi nhận diện được một ký hiệu.
* 🚦 **Cổng chuyển động (Motion Gate):** loại bỏ nhãn ảo khi tay chưa vào khung hình hoặc đang nghỉ (xem mục 4.3).
* 🛠️ **Bộ công cụ đầy đủ:** thu thập dữ liệu theo session, kiểm tra dữ liệu, đánh giá Leave-One-Session-Out (LOSO), đánh giá trên web, kiểm thử tự động.

---

## 📖 3. Từ vựng hỗ trợ (43 lớp)

| Nhóm | Các lớp |
| :--- | :--- |
| **Bảng chữ cái** | `a`, `aw (ă)`, `aa (â)`, `b`, `c`, `d`, `e`, `ee (ê)`, `f`, `g`, `h`, `i`, `j`, `k`, `l`, `m`, `n`, `o`, `oo (ô)`, `p`, `qq (q)`, `r`, `s`, `u`, `v`, `x`, `y`, `z` |
| **Dấu thanh** | `dau_sac`, `dau_huyen`, `dau_hoi`, `dau_nga`, `dau_nang` |
| **Xưng hô gia đình** | `bo_me`, `bo_duong`, `anh_em`, `anh_hai`, `anh_ho`, `anh_re`, `anh_ruot` |
| **Giao tiếp & khác** | `xin_chao`, `xin_loi`, `dao_dien` |

Dữ liệu: **5.166 mẫu** (43 lớp × 4 session × ~30 mẫu/session), mỗi mẫu là 30 khung hình × 1662 giá trị.

## 📦 3.1 Dữ liệu mở (Open Dataset - 258 Landmarks)

Nhằm phục vụ cộng đồng nghiên cứu và phát triển thêm, dự án phát hành bộ dữ liệu mở **VSL 258-Landmark**:
* **Quyền riêng tư & Kích thước đặc trưng:** Dữ liệu chỉ gồm **258 chiều** (tư thế thân trên 132 + tay trái 63 + tay phải 63), **loại bỏ hoàn toàn 1.404 điểm khuôn mặt (Face Mesh)** để bảo vệ danh tính người tham gia ký và loại bỏ nhiễu chiều dữ liệu.
* **Quy mô:** **5.166 mẫu** trên 43 lớp ký hiệu × 4 session thu thập độc lập.
* **Liên kết tải dữ liệu:** [Tải bộ dữ liệu vsl_landmarks_258_v1.zip](<CHƯA CÓ>) *(Lưu ý: URL tải sẽ do tác giả cung cấp khi xuất bản)*.
* **Kiểm tra tính toàn vẹn dữ liệu tải về:**
  ```powershell
  python tools/verify_dataset.py release_assets/dataset_public
  ```
* **Huấn luyện trực tiếp từ dữ liệu công khai:**
  ```powershell
  python pipeline/preprocess_data.py --input-dir release_assets/dataset_public --mode session --test-session 4
  ```

---

## ⚡ 4. Cài đặt & chạy nhanh

Dự án có sẵn mô hình đã huấn luyện `models/sign_language_model.h5` (huấn luyện trên toàn bộ 5.166 mẫu, 40 epoch), có thể chạy ngay.

### 4.1 Yêu cầu
* Python 3.12 (đã kiểm thử trên 3.12.8)
* Webcam
* Cài đúng phiên bản trong `requirements.txt`

### 4.2 Cài đặt

```powershell
# Windows (PowerShell)
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

```bash
# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

> Nếu lệnh `python` không dùng đúng môi trường ảo (hay gặp khi đã copy thư mục dự án), hãy gọi trực tiếp `.\.venv\Scripts\python.exe`.

**Chạy giao diện Web (khuyên dùng khi demo):**

```powershell
.\.venv\Scripts\python.exe app.py
```

Mở trình duyệt tại `http://localhost:5000` và cho phép truy cập webcam. Kiểm tra cấu hình đang chạy tại `http://localhost:5000/api/config`.

**Chạy bản Desktop (OpenCV + phát âm thanh trực tiếp):**

```powershell
.\.venv\Scripts\python.exe realtime_test.py
```

Nhấn `Q` để thoát.

### 4.3 Cấu hình suy luận (đầu file `app.py`)

| Hằng số | Mặc định | Ý nghĩa |
| :--- | :---: | :--- |
| `TARGET_DURATION` | 3.2 | Độ dài cửa sổ (giây), khớp với thời lượng một mẫu lúc thu thập |
| `MIN_FILL_RATIO` | 0.9 | Cần gom đủ 90% cửa sổ mới bắt đầu dự đoán |
| `THRESHOLD` | 0.6 | Độ tự tin tối thiểu |
| `SMOOTH_WINDOW` | 3 | Số lần dự đoán giống nhau liên tiếp để xác nhận một từ |
| `COOLDOWN_SEC` | 1.0 | Thời gian nghỉ sau mỗi lần nhận diện |
| `GATE_ENABLED` | `True` | Bật/tắt cổng chuyển động |
| `GATE_MIN_HAND_RATIO` | 0.50 | Tay phải có mặt ít nhất 50% số frame trong cửa sổ |
| `GATE_MOTION_STAT` | `"max"` | Thống kê chuyển động cổ tay (`max` hoặc `median`) |

**Cổng chuyển động:** khi bật, một ứng viên chỉ được chấp nhận nếu tay xuất hiện đủ tỉ lệ trong cửa sổ (và đủ chuyển động đối với các lớp động). Nhật ký khi chạy web, nối thêm mỗi lần chạy (nên đổi tên sau mỗi phiên đo):

* `logs/web_session_log.csv`: mọi lần dự đoán
* `logs/web_events.csv`: các từ được web phát ra
* `logs/web_gate_blocked.csv`: các ứng viên bị cổng chặn

---

## 🛠️ 5. Pipeline huấn luyện lại

```text
[Thu thập] ─► [Kiểm tra] ─► [Tiền xử lý] ─► [Đánh giá LOSO] ─► [Huấn luyện mô hình cuối]
data_collector  dataset_validator  preprocess_data  train_model --eval loso  train_model --final
```

### 5.1 Thu thập dữ liệu
```powershell
.\.venv\Scripts\python.exe pipeline/data_collector.py
```
Menu chọn session (4 kịch bản), đếm ngược tự động. Gõ một từ (ví dụ `xin_chao`) để quay theo block, hoặc cặp từ (ví dụ `a,aw`) để quay xen kẽ.

Bốn session ứng với bốn điều kiện quay khác nhau (xem `SCENARIOS` trong `data_collector.py`): chính diện, lệch trái ánh sáng ban ngày, gần camera, lệch phải ngược sáng. Điều kiện này là cơ sở của phép đánh giá theo session.

### 5.2 Kiểm tra dữ liệu
```powershell
.\.venv\Scripts\python.exe tools/dataset_validator.py
.\.venv\Scripts\python.exe tools/replay_sample.py dataset/raw/xin_chao/session_01/sample_001.npy
```

### 5.3 Tiền xử lý
```powershell
.\.venv\Scripts\python.exe pipeline/preprocess_data.py --mode session --test-session 4
```
Tạo `X_all.npy` (N×30×384), `X_raw258_all.npy` (N×30×258), `y_all.npy`, `session_all.npy` trong `dataset/processed/`. Chế độ `--mode random` giữ lại cách chia ngẫu nhiên cũ để so sánh.

### 5.4 Đánh giá (LOSO, protocol P2)
```powershell
.\.venv\Scripts\python.exe pipeline/train_model.py --eval loso --fixed-epochs 40 --seed 42 --out-json loso_results_v3_P2_seed42.json
```
Lặp lại với `--seed 7` và `--seed 123`. Kết quả JSON và ma trận nhầm lẫn lưu trong `models/eval/`.

### 5.5 Huấn luyện mô hình cuối
```powershell
.\.venv\Scripts\python.exe pipeline/train_model.py --final --fixed-epochs 40
```
Huấn luyện trên toàn bộ dữ liệu và lưu `models/sign_language_model.h5`.

### 5.6 Kiểm thử
```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py"
.\.venv\Scripts\python.exe tools/smoke_test_web.py
```

---

## 📊 6. Kết quả đánh giá

### 6.1 Cách đánh giá: vì sao chia theo session

Mỗi session gồm các mẫu quay liên tiếp trong cùng một điều kiện. Chia ngẫu nhiên khiến các mẫu gần giống nhau rơi vào cả tập train lẫn test (rò rỉ dữ liệu) và cho độ chính xác ảo (**~98.6%** trên dữ liệu cũ). Dự án đánh giá bằng **Leave-One-Session-Out (LOSO)**: lần lượt giữ một session làm test (điều kiện chưa từng thấy), session kế tiếp làm validation, hai session còn lại để train.

**Protocol P2** (protocol chính): huấn luyện cố định **40 epoch**, không dùng early stopping, đánh giá ở epoch cuối. Lý do: chọn epoch bằng một session validation lệch điều kiện làm kết quả dao động mạnh giữa các lần chạy.

### 6.2 Kết quả LOSO P2 (3 seed: 42, 7, 123; trung bình ± độ lệch chuẩn mẫu)

| Session test | Độ chính xác |
| :--- | :---: |
| S1 (chính diện) | 63.90 ± 4.32% |
| S2 (lệch trái, ánh sáng ngày) | 29.29 ± 0.86% |
| S3 (gần camera) | 71.00 ± 4.62% |
| S4 (lệch phải, ngược sáng) | 28.81 ± 4.90% |
| **Trung bình** | **48.25 ± 2.30%** (macro-F1 0.435 ± 0.026) |

Nhận xét: mô hình nhận tốt ở điều kiện gần dữ liệu train (S1, S3) nhưng **chỉ ~29% khi gặp góc camera và ánh sáng lạ (S2, S4)**. Các chữ cái tĩnh dễ nhầm lẫn nhau (ví dụ `d`/`o`, `m`/`n`, `v`/`u`, `h`/`k`), recall thấp nhất ở `a`, `b`, `aa`, `dau_hoi`.

### 6.3 Thử nghiệm đã làm và bị loại

* **Cắt đoạn hoạt động trước khi đưa vào model** (hand/motion): kết quả kém hơn, vì cử chỉ chiếm gần hết cửa sổ 3.2 giây.
* **Data augmentation** (xoay quanh trục Y/Z, scale, dịch, nhiễu, time-warp): khi so khớp seed (42 và 7), mức tăng khoảng +1.9 (`rot_y`) đến +3.1 điểm (`full`), chưa vượt ngưỡng đặt trước (≈ +4.6 điểm) nên **giữ nguyên mô hình hiện tại**.

### 6.4 Đánh giá trên web: tác dụng của cổng chuyển động

Thử trên webcam, 19 từ theo danh sách cố định (`expected_words.txt`), mỗi từ ký một lần, hai lượt đảo thứ tự (nhật ký và báo cáo chi tiết tại `results/ab_web/run1/` và `results/ab_web/run2/`). Người ký cùng điều kiện với dữ liệu train.

| Lượt | Cổng bật: đúng | Cổng tắt: đúng | Cổng bật: sai + thừa | Cổng tắt: sai + thừa | Nguồn log |
| :--- | :---: | :---: | :---: | :---: | :--- |
| 1 (tắt → bật) | 14/19 | 10/19 | 4 (sai 4, thừa 0) | 31 (sai 9, thừa 22) | `results/ab_web/run1/` |
| 2 (bật → tắt) | 15/19 | 6/19 | 3 (sai 3, thừa 0) | 12 (sai 12, thừa 0) | `results/ab_web/run2/` |
| **Tổng** | **29/38 (76%)** | **16/38 (42%)** | **7 (thừa 0)** | **43 (thừa 22)** | |

Trong 34 lần cổng chặn ở lượt 2 (`results/ab_web/run2/web_gate_blocked.csv`), 100% đều do tay chưa có mặt đủ trong cửa sổ (`closed_no_hand`). Đây là thử nghiệm nhỏ (một người, một lượt mỗi điều kiện), nên chỉ mang tính tham khảo.

### 6.5 Hiệu năng và độ trễ

* Dự đoán ≈ 5 ms mỗi lần (đã tối ưu bằng `fast_predict`).
* FPS xử lý trên web khoảng 7-10, khớp với nhịp thu thập (~9.5 FPS), vì vậy cửa sổ 30 khung hình trên web tương ứng ~3.2 giây như lúc huấn luyện.
* Độ trễ mỗi từ **~4 giây ở trường hợp tốt nhất** (phải gom đủ cửa sổ), thực tế đo được trung vị khoảng 7-8 giây tính từ lúc tay xuất hiện.

### 6.6 Hạn chế

* Độ chính xác giảm mạnh ở góc camera lệch và ánh sáng khác điều kiện huấn luyện.
* Độ trễ vài giây mỗi từ do thiết kế cửa sổ 3.2 giây; muốn giảm phải thu lại dữ liệu theo nhịp nhanh hơn.
* Các chữ cái tĩnh có hình dạng tay giống nhau dễ bị nhầm.
* Chưa có lớp "nghỉ" (không ký); hiện dùng cổng chuyển động ở tầng ứng dụng.
* Dữ liệu từ số ít người ký, chưa kiểm tra trên người dùng mới.

---

## 🧠 7. Kiến trúc kỹ thuật

1. **Trích xuất khung xương (MediaPipe Holistic):** 33 điểm pose, 468 điểm mặt, 21 điểm mỗi tay, tổng 1662 giá trị mỗi frame. Cấu hình lúc thu thập và lúc chạy web đã đồng bộ (độ phân giải 640×480, `model_complexity=1`).
2. **Đặc trưng (384 chiều):** bỏ khuôn mặt (còn 258 chiều thô: pose 132 + tay trái 63 + tay phải 63), chuẩn hóa theo độ rộng vai, thêm đặc trưng cục bộ của ngón tay so với cổ tay (+126 chiều).
3. **Chuẩn hóa thời gian khi chạy web:** buffer frame theo dấu thời gian, nội suy tuyến tính về đúng 30 điểm trên dữ liệu thô 258 chiều trước khi tính đặc trưng.
4. **Mô hình LSTM:** `LSTM(64, return_sequences=True)` → `Dropout(0.2)` → `LSTM(128)` → `Dropout(0.2)` → `Dense(64, relu)` → `Dense(43, softmax)`.
5. **Hậu xử lý:** ngưỡng tin cậy, xác nhận 3 lần liên tiếp, thời gian nghỉ, cổng chuyển động.

---

## 📂 8. Cấu trúc thư mục

```text
SignLanguageTranslator/
├── README.md
├── requirements.txt
├── requirements-dev.txt           # Công cụ phát triển & kiểm tra mã tĩnh (pyflakes)
├── dictionary.json                # nhãn → tiếng Việt có dấu
├── expected_words.txt             # danh sách từ dùng cho đánh giá trên web
├── CLEANUP_PLAN.md                # kế hoạch dọn dẹp và phân loại file
├── cleanup_moves.csv              # nhật ký truy vết các file đã di chuyển
│
├── app.py                         # Web demo (Flask + SocketIO)
├── realtime_test.py               # Bản Desktop (OpenCV + Pygame)
├── feature_utils.py               # Trích xuất & chuẩn hóa đặc trưng
├── temporal_utils.py              # Nội suy theo thời gian, làm mượt dự đoán
│
├── pipeline/
│   ├── data_collector.py
│   ├── preprocess_data.py
│   ├── train_model.py
│   ├── augment_utils.py
│   ├── segment_utils.py
│   └── web_logging.py
│
├── tools/                         # Kiểm tra, đánh giá, phân tích, undo_cleanup.ps1
├── tests/                         # Kiểm thử tự động (18 unit tests)
│
├── dataset/
│   ├── raw/                       # dữ liệu thô 43 từ x 4 session (5.166 mẫu)
│   └── processed/                 # dữ liệu đã tiền xử lý (tái tạo được)
│
├── models/
│   ├── sign_language_model.h5     # mô hình LSTM chính thức
│   └── eval/                      # kết quả LOSO (21 JSON) và ma trận nhầm lẫn P2
│
├── results/                       # bằng chứng cho báo cáo (xem results/INDEX.md)
│   ├── ab_web/                    # nhật ký & đánh giá A/B cổng chuyển động
│   ├── analysis/                  # phân tích FPS, suy luận, landmark, độ dài cử chỉ
│   └── training_logs/             # log huấn luyện LOSO và tiền xử lý
├── logs/                          # nhật ký khi chạy web (giữ .gitkeep)
├── audio/  static/  templates/
└── archive/                       # công cụ và file cũ, bản sao lưu
    ├── models_backup/             # bản sao lưu model qua các vòng
    ├── models_eval_extra/         # ảnh biểu đồ của các cấu hình thử nghiệm phụ
    ├── review_packages/          # các gói zip đánh giá vòng trước
    ├── tools/                     # các script một lần, đã hoàn thành nhiệm vụ
    └── README_old.md              # bản sao lưu README trước dọn dẹp
```

---

## 🔁 9. Tái lập kết quả

* Seed: 42, 7, 123. Protocol P2 (40 epoch). Dữ liệu 5.166 mẫu.
* Số liệu LOSO nằm trong `models/eval/`. Bảng tổng hợp có thể tạo lại bằng `tools/evaluate_loso_seeds.py` và `tools/evaluate_augment_seeds.py`.
* Dữ liệu và nhật ký đánh giá trên web, các phân tích khác: `results/INDEX.md`.

---

## 🎯 10. Baseline cần vượt (Baseline to Beat)

Mọi đóng góp cải tiến kiến trúc hoặc mô hình cần so sánh với kết quả **Protocol P2 (LOSO 4 folds, 40 epochs, 3 seeds 42/7/123)**:
* **Độ chính xác trung bình (Mean Accuracy):** **48.25 ± 2.30%**
* **Macro-F1:** **0.4350 ± 0.0256**
* **Chi tiết từng Fold:** S1 (63.90%), S2 (29.29%), S3 (71.00%), S4 (28.81%)
* Quy tắc chấp nhận mô hình mới: Không làm suy giảm độ chính xác của bất kỳ fold nào quá 3% so với baseline và tăng tổng thể tối thiểu +4.59%.

---

## 📄 11. Giấy phép (License)

* **Mã nguồn:** Phát hành theo giấy phép [MIT License](LICENSE) (© 2026 Đoàn Duy Bình).
* **Bộ dữ liệu:** Phát hành theo giấy phép [Creative Commons Attribution 4.0 International (CC BY 4.0)](release_assets/dataset_public/LICENSE-DATA).

---

## 👥 Đóng góp & phát triển

Vui lòng đọc hướng dẫn chi tiết tại [CONTRIBUTING.md](CONTRIBUTING.md).
Hướng phát triển: thu thêm dữ liệu cho các chữ yếu ở điều kiện góc nghiêng/ngược sáng, thêm một session với người ký mới, thêm lớp "nghỉ", rút ngắn cửa sổ thu thập để giảm độ trễ, thử các kiến trúc khác (Transformer, ST-GCN) hoặc TFLite cho thiết bị di động.
