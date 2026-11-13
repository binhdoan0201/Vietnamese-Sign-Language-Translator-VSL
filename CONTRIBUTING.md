# 🤝 Hướng Dẫn Đóng Góp (Contributing Guidelines)

Cảm ơn bạn đã quan tâm đóng góp cho dự án **SignLanguageTranslator** (Nhận diện Ngôn ngữ Ký hiệu Tiếng Việt)!

---

## 1. Thiết lập Môi trường Phát triển & Kiểm thử

1. **Khởi tạo môi trường ảo (Python 3.12 khuyến nghị):**
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\activate
   pip install -r requirements.txt
   pip install -r requirements-dev.txt
   ```

2. **Chạy toàn bộ kiểm thử tự động:**
   ```powershell
   python -m unittest discover -s tests -p "test_*.py"
   ```

3. **Kiểm tra cú pháp và chất lượng mã nguồn:**
   ```powershell
   python -m pyflakes app.py realtime_test.py feature_utils.py temporal_utils.py pipeline tests tools
   ```

---

## 2. Giao thức Đánh giá Bắt buộc khi So sánh Kết quả (Evaluation Protocol)

Để đảm bảo tính khoa học và có thể so sánh công bằng với các mốc cơ sở:
1. **Chia dữ liệu:** Phải dùng **Leave-One-Session-Out (LOSO)** 4 fold theo 4 session thực tế. Tuyệt đối không dùng chia ngẫu nhiên (random split) vì sẽ gây rò rỉ dữ liệu giữa các khung hình liên tiếp.
2. **Protocol P2 (Chuẩn):**
   - Huấn luyện cố định đúng **40 epochs**, không sử dụng early stopping.
   - Đánh giá trên epoch cuối cùng (epoch 40).
3. **Đa Seed (Multi-seed):**
   - Bắt buộc thực nghiệm trên 3 seed cố định: `42`, `7`, `123`.
   - Báo cáo kết quả dưới dạng: `Mean ± Sample Std` (ddof=1) cho cả Accuracy tổng thể và Macro-F1.
4. **Mốc Baseline hiện tại cần vượt:**
   - **Mean Accuracy: 48.25 ± 2.30%** (S1: 63.90%, S2: 29.29%, S3: 71.00%, S4: 28.81%)
   - **Macro-F1: 0.4350 ± 0.0256**

---

## 3. Quy định Đóng góp Dữ liệu Mới (Data Contribution)

1. **Bảo vệ quyền riêng tư:** Dữ liệu đóng góp chỉ được chứa tọa độ khung xương tư thế và hai bàn tay (**258 chiều / frame**, shape `(30, 258)`), tuyệt đối không gửi ảnh/video hoặc tọa độ khuôn mặt.
2. **Nguyên tắc phân chia Session:**
   - Khi thu thập dữ liệu mới, hãy tạo **session mới** (ví dụ `session_05`, `session_06` với kịch bản ghi rõ: người ký, góc quay, ánh sáng).
   - **Không trộn dữ liệu mới vào các session cũ** (`session_01` đến `session_04`) để duy trì tính độc lập phục vụ đánh giá tổng quát hóa.
3. **Đồng thuận của người ký:** Người đóng góp phải đảm bảo người tham gia ký đã đồng ý công bố dữ liệu dưới giấy phép **CC BY 4.0** và dữ liệu được ẩn danh hóa.

---

## 4. Các Hướng Phát triển Ưu tiên (Roadmap)

- [ ] Thu thập bổ sung mẫu cho các chữ cái tĩnh yếu (`h`, `i`, `x`, `k`, `n`, `u`, `m`, `l`, `a`, `b`, `c`, `aa`, `dau_hoi`) trong điều kiện góc nghiêng và ánh sáng lạ.
- [ ] Bổ sung session hoàn toàn độc lập với người ký mới để đánh giá độ tổng quát hóa liên người ký (cross-signer evaluation).
- [ ] Huấn luyện lớp "nghỉ" (idle/rest class) tự nhiên trong mô hình để giảm phụ thuộc vào cổng chuyển động ở tầng ứng dụng.
- [ ] Rút ngắn cửa sổ thời gian (ví dụ từ 3.2s xuống 1.5s - 2.0s) bằng cách thu thập cử chỉ theo nhịp nhanh hơn.
- [ ] Thử nghiệm các kiến trúc học sâu không-thời gian tiên tiến hơn (ST-GCN, 1D-CNN + BiLSTM, Transformers) hoặc chuyển đổi mô hình sang TFLite để chạy tối ưu trên thiết bị di động.

---

## 5. Lưu ý về Công cụ Nội bộ Dành cho Tác giả (Author-only Tools)

Trong thư mục `tools/`, một số tập lệnh được thiết kế phục vụ quy trình chuẩn bị dữ liệu gốc của tác giả (yêu cầu dữ liệu thô 1662 chiều đầy đủ):
- `tools/export_public_dataset.py`: Trích xuất và xuất gói dữ liệu mở 258 chiều từ dữ liệu thô ban đầu.
- `tools/prove_equivalence.py`: Thực nghiệm chứng minh tính tương đương số học giữa pipeline từ dữ liệu thô và dữ liệu mở.

Người dùng và cộng đồng nghiên cứu chỉ cần sử dụng bộ dữ liệu mở 258 chiều đã giải nén vào `dataset/raw` cùng các tập lệnh chính (`tools/verify_dataset.py`, `pipeline/preprocess_data.py`, `pipeline/train_model.py`, `app.py`).

