# 📦 Dữ liệu Ngôn ngữ Ký hiệu (VSL Dataset)

Dữ liệu thô và các tệp mảng NumPy (`.npy`) **không nằm trong kho mã nguồn này** để đảm bảo kích thước repository tinh gọn.

## 🚀 Hướng dẫn nhanh

1. **Tải dữ liệu:** Tải gói dữ liệu mở `vsl_landmarks_258_v1.zip` tại mục [Releases](https://github.com/binhdoan0201/Vietnamese-Sign-Language-Translator-VSL/releases/tag/v1.0) của dự án.
2. **Giải nén vào `dataset/raw`:**
   - **Windows (Command Prompt):**
     ```cmd
     mkdir dataset\raw
     tar -xf vsl_landmarks_258_v1.zip -C dataset\raw
     ```
   - **Windows (PowerShell):**
     ```powershell
     Expand-Archive vsl_landmarks_258_v1.zip -DestinationPath dataset\raw
     ```
   - **Linux / macOS:**
     ```bash
     mkdir -p dataset/raw && unzip vsl_landmarks_258_v1.zip -d dataset/raw
     ```
3. **Kiểm tra tính toàn vẹn:**
   ```powershell
   python tools/verify_dataset.py dataset/raw
   ```
4. **Dùng cho huấn luyện lại hoặc nghiên cứu:**
   ```powershell
   python pipeline/preprocess_data.py --mode session --test-session 4
   python pipeline/train_model.py
   ```

Chi tiết cấu trúc đặc trưng, kịch bản thu thập và giấy phép:
- Xem thẻ mô tả dữ liệu: [DATASET_CARD.md](DATASET_CARD.md)
- Xem giấy phép dữ liệu: [LICENSE-DATA](LICENSE-DATA) (CC BY 4.0)
