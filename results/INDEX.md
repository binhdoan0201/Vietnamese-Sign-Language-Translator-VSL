# CHỈ MỤC BẰNG CHỨNG THỰC NGHIỆM VÀ LOGS (results/INDEX.md)
*Cập nhật: 12/11/2026 - Dự án SignLanguageTranslator*

Thư mục `results/` chứa toàn bộ dữ liệu thực nghiệm, log huấn luyện, kết quả kiểm thử và phân tích của dự án qua các vòng phát triển và đánh giá. Tài liệu này cung cấp mục lục tra cứu nguồn gốc số liệu được trích dẫn trong `README.md` và các báo cáo tổng kết.

---

## 1. Phân mục A/B Web và Cổng chuyển động (`results/ab_web/`)

Khu vực lưu trữ dữ liệu thực nghiệm kiểm thử A/B cổng chuyển động (`Motion Gate`) trên giao diện web thời gian thực với kịch bản chuẩn 19 từ cử chỉ tiếng Việt (`expected_words.txt`). Thực nghiệm gồm 2 lượt đo độc lập:
- **Lượt 1 (run1):** Thứ tự chạy TẮT cổng trước $\to$ BẬT cổng sau.
- **Lượt 2 (run2):** Thứ tự chạy BẬT cổng trước $\to$ TẮT cổng sau.

### 1.1 Lượt 1 (`results/ab_web/run1/`)
| Tên file | Nội dung & Mục đích | Lệnh / Nguồn tạo ra | Con số chính | Vị trí trong README / Báo cáo |
|---|---|---|---|---|
| `web_events_gate_off.csv` | Log sự kiện web Lượt 1 khi TẮT cổng (`gate_enabled=False`) | Web demo `app.py` (lượt 1) | 41 sự kiện: 10 đúng, 9 sai, 0 sót, 22 thừa; trễ trung vị 7.01s (P90: 27.33s) | README mục 6.4 |
| `web_events_gate_on.csv` | Log sự kiện web Lượt 1 khi BẬT cổng (`gate_enabled=True`) | Web demo `app.py` (lượt 1) | 18 sự kiện: 14 đúng, 4 sai, 1 sót, 0 thừa; trễ trung vị 7.23s (P90: 17.44s) | README mục 6.4 |
| `eval_gate_off.txt` | Báo cáo đánh giá tự động Lượt 1 khi tắt cổng | `python tools/eval_web_events.py results/ab_web/run1/web_events_gate_off.csv expected_words.txt` | Đúng: 10/19 (52.6%), Sai: 9, Sót: 0, Dư thừa: 22 nhãn rác (`anh_em` 5, `f` 4, `m` 3...) | README mục 6.4 |
| `eval_gate_on.txt` | Báo cáo đánh giá tự động Lượt 1 khi bật cổng | `python tools/eval_web_events.py results/ab_web/run1/web_events_gate_on.csv expected_words.txt` | Đúng: 14/19 (73.7%), Sai: 4, Sót: 1, Dư thừa: 0 | README mục 6.4 |

### 1.2 Lượt 2 (`results/ab_web/run2/`)
| Tên file | Nội dung & Mục đích | Lệnh / Nguồn tạo ra | Con số chính | Vị trí trong README / Báo cáo |
|---|---|---|---|---|
| `web_events_gate_on.csv` | Log sự kiện web Lượt 2 khi BẬT cổng (`gate_enabled=True`) | Web demo `app.py` (lượt 2) | 18 sự kiện: 15 đúng, 3 sai, 1 sót, 0 thừa; trễ trung vị 7.96s (P90: 18.15s) | README mục 6.4 |
| `web_events_gate_off.csv` | Log sự kiện web Lượt 2 khi TẮT cổng (`gate_enabled=False`) | Web demo `app.py` (lượt 2) | 18 sự kiện: 6 đúng, 12 sai, 1 sót, 0 thừa; trễ trung vị 7.07s (P90: 20.17s) | README mục 6.4 |
| `eval_gate_on.txt` | Báo cáo đánh giá tự động Lượt 2 khi bật cổng | `python tools/eval_web_events.py results/ab_web/run2/web_events_gate_on.csv expected_words.txt` | Đúng: 15/19 (78.9%), Sai: 3, Sót: 1, Dư thừa: 0 | README mục 6.4 |
| `eval_gate_off.txt` | Báo cáo đánh giá tự động Lượt 2 khi tắt cổng | `python tools/eval_web_events.py results/ab_web/run2/web_events_gate_off.csv expected_words.txt` | Đúng: 6/19 (31.6%), Sai: 12, Sót: 1, Dư thừa: 0 | README mục 6.4 |
| `web_gate_blocked.csv` | Chi tiết các khung hình bị cổng chuyển động chặn lại ở lượt 2 | Hook ghi nhận trong `app.py` | 34 dòng, 100% `closed_no_hand` (tay chưa có mặt đủ 50% cửa sổ) | README mục 6.4 |
| `web_session_log.csv` | Log chi tiết chu trình xử lý khung hình thời gian thực trên web ở lượt 2 | `app.py` | 420 dòng log, FPS trung bình 7.54 (trung vị 7.21), predict time trung bình 7.55 ms/lần | README mục 6.5 |

### 1.3 Tổng hợp kết quả A/B qua 2 lượt (38 lần thử)
- **Cổng BẬT (cộng 2 lượt):** Đúng **29/38 (76.3%)** (Lượt 1: 14/19; Lượt 2: 15/19); Dư thừa: **0 nhãn thừa** (cả hai lượt đều không phát sinh nhãn rác).
- **Cổng TẮT (cộng 2 lượt):** Đúng **16/38 (42.1%)** (Lượt 1: 10/19; Lượt 2: 6/19); Dư thừa: **22 nhãn thừa** (toàn bộ xuất hiện ở Lượt 1 khi hạ tay ở trạng thái nghỉ).

---

## 2. Phân mục Phân tích & Benchmark (`results/analysis/`)

Lưu trữ các phân tích kỹ thuật về độ trễ, FPS, MediaPipe complexity, tính toàn vẹn dataset và kiểm thử chất lượng code.

| Tên file | Nội dung & Mục đích | Lệnh / Nguồn tạo ra | Con số chính | Vị trí trong README / Báo cáo |
|---|---|---|---|---|
| `benchmark_fps.txt` | Đo đạc FPS thực tế của webcam, MediaPipe Holistic và vòng lặp Flask-SocketIO | `python tools/benchmark_fps.py` | Holistic lúc quay: ~9.5 FPS (1 sample ~3.18s); Holistic 640x480: 40.24 ms/frame (~24.8 FPS); predict cũ: ~90.86 ms | README mục 6.1 |
| `benchmark_inference.txt` | So sánh thời gian suy luận giữa `model.predict()` và `@tf.function` `fast_predict()` | `python tools/benchmark_inference.py` | model.predict: mean 59.32 ms, P95 69.62 ms; fast_predict: mean **1.98 ms**, P95 **2.27 ms** (giảm 96.7% trễ suy luận) | README mục 6.1 & 6.5 |
| `compare_mediapipe.txt` | So sánh phân phối tọa độ landmark giữa `model_complexity=0` và `complexity=1` | `python tools/compare_mediapipe_settings.py` | Complexity 0 làm lệch phân phối landmark so với data train $\to$ Quyết định giữ complexity 1 | Báo cáo Vòng 2 |
| `dataset_validator.txt` | Kiểm tra tính toàn vẹn 43 lớp x 4 session của bộ dữ liệu gốc | `python tools/dataset_validator.py` | 5.166/5.166 mẫu hợp lệ, shape chuẩn `(30, 1662)`, 0 lỗi/cảnh báo | README mục 3 & 5.2 |
| `analyze_gesture_span.txt` | Phân tích phân bố thời lượng cử chỉ trong cửa sổ 30 khung hình | `python tools/analyze_gesture_span.py` | Cử chỉ chiếm 20-28 frame (~2.1 - 2.9s) trong cửa sổ 3.2s $\to$ Không thể cắt ngắn cửa sổ mà không mất thông tin | README mục 6.3 |
| `analyze_weak_classes.txt` | Thống kê và phân tích các lớp nhận diện kém trong kiểm thử LOSO P2 | `python tools/analyze_weak_classes.py` | Các lớp yếu: h (29.4%), i (27.7%), x (37.2%), dau_hoi (19.4%), a (14.2%), b (14.7%), aa (16.1%) | README mục 6.2 |
| `diagnose_dau_hoi.txt` | Chẩn đoán nguyên nhân nhầm lẫn cụ thể của lớp `dau_hoi` | `python tools/diagnose_dau_hoi.py` | Nhầm với các cử chỉ có quỹ đạo uốn lượn tương đồng | Báo cáo Vòng 2 |
| `offline_replay_test.txt` | Log kiểm tra mô phỏng phát lại chuỗi landmark ngoại tuyến | `tools/offline_replay_test.py` (vòng 1) | Ghi nhận tính khả thi của pipeline ngoại tuyến | Báo cáo Vòng 1 |
| `app_config_check.txt` | Kiểm tra phản hồi cấu hình từ endpoint REST `/api/config` | HTTP GET `/api/config` | Trả về JSON cấu hình: `send_width=640`, `mp_complexity=1`, `gate_enabled=True` | Báo cáo Vòng 2 |
| `smoke_test_web.py` log / `smoke_test_web.txt` | Kiểm thử tích hợp khởi động headless và pipeline dự đoán | `python tools/smoke_test_web.py` | Kết nối thành công, nhận frame và ghi event CSV đạt chuẩn 100% | README mục 5.6 |
| `run_all_tests.txt` | Log chạy toàn bộ test tự động của hệ thống | `python -m unittest discover tests` | Toàn bộ 18 unit tests đều PASS | README mục 5.6 |
| `pyflakes_before.txt` / `pyflakes_after.txt` | Kiểm tra lỗi cú pháp và biến thừa bằng pyflakes | `python -m pyflakes ...` | Kết quả sau dọn dẹp sạch hoàn toàn (0 lỗi/cảnh báo) | Báo cáo Cleanup |
| `test_temporal_utils.txt` | Kết quả kiểm thử các hàm nội suy thời gian 258 chiều | `tests/test_temporal_utils.py` | Đảm bảo tính toán nội suy chính xác, bảo toàn frame mất tay | Báo cáo Vòng 1 |
| `test_encoding.txt` | Kiểm tra hỗ trợ UTF-8 cho từ điển tiếng Việt | Unit test encoding | Không bị lỗi font hoặc sai lệch chuỗi Unicode tiếng Việt có dấu | Báo cáo Vòng 1 |

---

## 3. Phân mục Nhật ký Huấn luyện (`results/training_logs/`)

Lưu trữ log chi tiết của các lượt huấn luyện và tiền xử lý qua các vòng.

| Tên file | Nội dung & Protocol | Thông số huấn luyện & Kết quả | Vị trí trong README / Báo cáo |
|---|---|---|---|
| `preprocess_random.txt` | Tiền xử lý dữ liệu chia ngẫu nhiên (Baseline cũ trước vòng 1) | 80% train / 20% test ngẫu nhiên | Báo cáo Vòng 1 |
| `preprocess_session.txt` | Tiền xử lý dữ liệu chia theo Session (Vòng 1, 4 session) | Lưu trữ mảng `session_all.npy`, `X_raw258_all.npy` | Báo cáo Vòng 1 |
| `preprocess_session_r2.txt` | Tiền xử lý dữ liệu cập nhật Vòng 2 | Kiểm tra định dạng dữ liệu đầy đủ | Báo cáo Vòng 2 |
| `train_random_baseline.txt` | Huấn luyện baseline chia ngẫu nhiên | Test Accuracy: ~96.5% (Overfitting do rò rỉ session) | Báo cáo Vòng 1 |
| `train_loso.txt` | Huấn luyện LOSO 4 fold đầu tiên (Vòng 1, 3816 mẫu) | Mean Acc: 46.14% ± 20.09% | Báo cáo Vòng 1 |
| `train_loso_full.txt` | Huấn luyện LOSO trên bộ dữ liệu đủ 5166 mẫu | Mean Acc: ~40.75% (chạy với Protocol P1) | Báo cáo Vòng 3 |
| `train_loso_v3_P1_seed*.txt` | Huấn luyện LOSO theo Protocol P1 (Early stopping theo val session, seed 7, 42, 123) | Dao động mạnh giữa các seed do val session lệch phân phối | Báo cáo Vòng 3 |
| `train_loso_v3_P2_seed*.txt` | **Huấn luyện LOSO theo Protocol P2 CHUẨN** (Cố định 40 epoch, không early stopping, seed 42, 7, 123) | **Mean Acc: 48.25% ± 2.30%**; S1: 63.90%, S2: 29.29%, S3: 71.00%, S4: 28.81% | **Mục 6.2 Bảng 1 README** |
| `train_loso_v3_segment_*.txt` | Thử nghiệm cắt đoạn cử động tay (`segment_hand`, `segment_motion`) | Độ chính xác giảm xuống ~34-36% $\to$ Đã loại bỏ | README mục 6.3 |
| `train_loso_v4_rot_*.txt` | Thử nghiệm tăng cường dữ liệu xoay trục Y/Z dưới Protocol P2 | Rotate Y (seed 42, 7): ~51.16% (không đạt ngưỡng B4 +4.59%, rủi ro biến dạng landmark) | README mục 6.3 |
| `train_final.txt` | Huấn luyện mô hình triển khai thực tế trên toàn bộ 4 session (Vòng 1) | Huấn luyện 40 epoch | Báo cáo Vòng 1 |
| `train_final_v3.txt` | **Huấn luyện mô hình triển khai thực tế chính thức** (`models/sign_language_model.h5`, 5166 mẫu, 4 session) | Huấn luyện 40 epoch, loss hội tụ 0.08 | **Mục 6.2 README** |

---

## 4. Tóm tắt Đối chiếu Nguồn dữ liệu

Khi người kiểm tra hoặc người dùng đọc các bảng số liệu trong `README.md`, có thể đối chiếu trực tiếp với các file nguồn:
1. **Bảng 1 (LOSO Protocol P2 - 4 Folds):**
   - File JSON: `models/eval/loso_results_v3_P2_seed42.json`, `loso_results_v3_P2_seed7.json`, `loso_results_v3_P2_seed123.json`
   - File log chi tiết: `results/training_logs/train_loso_v3_P2_seed*.txt`
2. **Bảng 2 (Thử nghiệm Tăng cường Dữ liệu Vòng 4):**
   - File JSON: `models/eval/loso_results_v4_rot_y_seed*.json`, `models/eval/loso_results_v4_rot_yz_seed*.json`
   - File log chi tiết: `results/training_logs/train_loso_v4_*.txt`
3. **Bảng 3 (Thực nghiệm A/B Cổng chuyển động trên Web):**
   - Lượt 1: `results/ab_web/run1/web_events_gate_off.csv`, `results/ab_web/run1/web_events_gate_on.csv`, `results/ab_web/run1/eval_gate_off.txt`, `results/ab_web/run1/eval_gate_on.txt`
   - Lượt 2: `results/ab_web/run2/web_events_gate_off.csv`, `results/ab_web/run2/web_events_gate_on.csv`, `results/ab_web/run2/eval_gate_off.txt`, `results/ab_web/run2/eval_gate_on.txt`
   - Khung hình bị chặn (Lượt 2): `results/ab_web/run2/web_gate_blocked.csv`
