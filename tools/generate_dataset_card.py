"""
tools/generate_dataset_card.py
==============================
Tự động tính toán số mẫu theo (class x session) từ release_assets/dataset_public
và tạo file DATASET_CARD.md đầy đủ tiêu chuẩn khoa học.
"""
import os
import glob
import pandas as pd

DATASET_DIR = "release_assets/dataset_public"
OUT_CARD = os.path.join(DATASET_DIR, "DATASET_CARD.md")

def main():
    print("Đang quét dữ liệu trong release_assets/dataset_public để tạo DATASET_CARD.md...")
    classes = sorted([d for d in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, d))])
    
    rows = []
    total_by_session = {"session_01": 0, "session_02": 0, "session_03": 0, "session_04": 0}
    
    for c in classes:
        row = {"class": c}
        c_total = 0
        for s in ["session_01", "session_02", "session_03", "session_04"]:
            s_dir = os.path.join(DATASET_DIR, c, s)
            n_files = len(glob.glob(os.path.join(s_dir, "*.npy"))) if os.path.isdir(s_dir) else 0
            row[s] = n_files
            total_by_session[s] += n_files
            c_total += n_files
        row["total"] = c_total
        rows.append(row)
        
    df = pd.DataFrame(rows)
    total_samples = df["total"].sum()
    
    # Tạo bảng markdown
    table_lines = [
        "| STT | Lớp ký hiệu (Class) | Session 01 | Session 02 | Session 03 | Session 04 | Tổng cộng |",
        "|---|---|:---:|:---:|:---:|:---:|:---:|"
    ]
    for idx, r in df.iterrows():
        table_lines.append(f"| {idx+1} | `{r['class']}` | {r['session_01']} | {r['session_02']} | {r['session_03']} | {r['session_04']} | **{r['total']}** |")
    table_lines.append(f"| | **TỔNG CỘNG** | **{total_by_session['session_01']}** | **{total_by_session['session_02']}** | **{total_by_session['session_03']}** | **{total_by_session['session_04']}** | **{total_samples}** |")
    table_md = "\n".join(table_lines)
    
    card_content = f"""# 📋 Dataset Card: Vietnamese Sign Language (VSL) 258-Landmark Dataset

## 1. Tổng quan bộ dữ liệu (Dataset Summary)
Bộ dữ liệu **VSL 258-Landmark** là tập dữ liệu mở phục vụ nghiên cứu và phát triển các mô hình nhận diện Ngôn ngữ Ký hiệu Tiếng Việt (Vietnamese Sign Language). Dữ liệu được trích xuất từ webcam thông thường qua MediaPipe Holistic, sau đó **loại bỏ hoàn toàn 1.404 điểm khuôn mặt** để bảo vệ quyền riêng tư cá nhân và tập trung vào đặc trưng tư thế thân trên cùng chuyển động của hai bàn tay.

- **Tổng số mẫu:** {total_samples:,} mẫu
- **Số lớp ký hiệu:** 43 lớp (chữ cái, dấu thanh, danh từ xưng hô gia đình, giao tiếp cơ bản)
- **Số session thu thập:** 4 session độc lập với các điều kiện góc quay, khoảng cách và ánh sáng khác nhau
- **Định dạng dữ liệu:** Mảng NumPy (`.npy`), kiểu dữ liệu `float32`, kích thước chuẩn `(30, 258)`

---

## 2. Cấu trúc và Định dạng dữ liệu (Data Format)
Mỗi mẫu cử chỉ là một tệp `.npy` chứa chuỗi thời gian gồm đúng **30 khung hình**, mỗi khung hình biểu diễn bởi **258 giá trị tọa độ thực** (chuẩn hóa theo không gian khung hình [0, 1]).

Thứ tự 258 chiều đặc trưng được trích xuất chuẩn hóa từ `feature_utils.slice_features`:
1. **Tư thế thân trên (Pose landmarks):** 132 giá trị đầu tiên (33 khớp xương × 4 giá trị `[x, y, z, visibility]`). Bao gồm tọa độ mũi, mắt, tai, vai, khuỷu tay, cổ tay, hông.
2. **Bàn tay trái (Left Hand landmarks):** 63 giá trị tiếp theo (21 khớp bàn tay × 3 giá trị `[x, y, z]`).
3. **Bàn tay phải (Right Hand landmarks):** 63 giá trị cuối cùng (21 khớp bàn tay × 3 giá trị `[x, y, z]`).

> [!NOTE]
> - Điểm gốc `[0, 0, 0]` biểu thị bàn tay vắng mặt trong khung hình (không phát hiện được).
> - Dữ liệu **HOÀN TOÀN KHÔNG CHỨA** hình ảnh video gốc, không chứa điểm lưới khuôn mặt (Face Mesh 1404 chiều đã bị lược bỏ).

---

## 3. Điều kiện thu thập và Phân chia Session (Collection Scenarios)
Tốc độ thu thập thực tế đo được qua benchmark hệ thống là **~9.5 FPS**, tương ứng mỗi chuỗi 30 khung hình biểu diễn khoảng **~3.2 giây** cử động tự nhiên.

Dữ liệu được tổ chức theo 4 session với kịch bản thu thập định sẵn:
- **`session_01` (Chính diện, điều kiện chuẩn):** Camera đặt góc 0° ngang tầm ngực, ánh sáng đèn phòng trong nhà ổn định, khoảng cách ~1.0m - 1.2m.
- **`session_02` (Góc lệch trái, ánh sáng ban ngày):** Camera lệch góc 15° - 20° về bên trái, nguồn sáng tự nhiên từ cửa sổ chiếu xiên.
- **`session_03` (Khoảng cách gần, đa người ký):** Camera chính diện ở khoảng cách gần (~0.6m - 0.8m), ánh sáng phòng bình thường.
- **`session_04` (Góc lệch phải, ngược sáng nhẹ):** Camera lệch góc 15° - 20° về bên phải, điều kiện ánh sáng yếu / ngược sáng.

---

## 4. Thông tin người ký và Tuyên bố đồng thuận (Signers & Consent)
- **Số lượng người ký:** 3 người (mã hóa: `signer_A`, `signer_B`, `signer_C`).
  - `signer_A` (tác giả Đoàn Duy Bình): Ký toàn bộ `session_01`, `session_02`, `session_04` và phần lớn `session_03` (~94% tổng dữ liệu).
  - `signer_B`: Ký khoảng 240 mẫu (~8 lớp) trong `session_03`.
  - `signer_C`: Ký khoảng 90 mẫu (~3 lớp) trong `session_03`.
- **Tình trạng gán nhãn người ký:** Mỗi người ký thực hiện trọn vẹn từng lớp trong `session_03`. Do bản đồ phân công chi tiết chưa được xác nhận chính thức (`SIGNER_MAP_CONFIRMED = False`), trong `metadata.csv`, các session 01, 02, 04 được ghi `signer_A`, còn `session_03` được định danh là `multiple_unlabeled`.
- **Đồng thuận (Consent):** **Đã xác nhận (CONSENT_CONFIRMED = True)**. Dữ liệu được công bố công khai với sự đồng ý đầy đủ của tất cả những người tham gia ký. Để bảo vệ quyền riêng tư, danh tính của người tham gia được ẩn danh hóa và chỉ dùng mã số.

---

## 5. Thống kê chi tiết số mẫu (Data Distribution)
Bảng thống kê số lượng mẫu thực tế theo từng lớp và từng session (tính toán tự động từ tệp tin):

{table_md}

---

## 6. Giao thức Đánh giá Khuyến nghị & Kết quả Cơ sở (Evaluation & Baseline)

### Giao thức khuyến nghị (Recommended Protocol)
Để đo lường chính xác năng lực tổng quát hóa và tránh hiện tượng rò rỉ dữ liệu giữa các khung hình liên tiếp:
- **Phương pháp:** **Leave-One-Session-Out (LOSO) Cross-Validation** (4 folds). Ở mỗi fold, giữ 1 session làm tập Test (môi trường chưa từng thấy), session kế tiếp làm Validation, 2 session còn lại làm tập Train.
- **Protocol P2 (Chuẩn):** Huấn luyện mạng LSTM cố định **40 epochs**, không sử dụng early stopping (nhằm tránh nhiễu do phân phối tập validation bị lệch). Đánh giá trung bình qua **3 seeds: 42, 7, 123**.

### Kết quả Baseline chính thức (Protocol P2)
Được huấn luyện trên mô hình LSTM 2 lớp (64 -> 128 units):
- **Độ chính xác trung bình (Mean Accuracy):** **48.25 ± 2.30%** (Macro-F1: 0.4350 ± 0.0256)
- **Độ chính xác từng Fold:**
  - Fold 1 (Test trên Session 01): **63.90 ± 4.32%**
  - Fold 2 (Test trên Session 02 - Lệch trái): **29.29 ± 0.86%**
  - Fold 3 (Test trên Session 03 - Gần): **71.00 ± 4.62%**
  - Fold 4 (Test trên Session 04 - Lệch phải): **28.81 ± 4.90%**

---

## 7. Các hạn chế đã biết (Known Limitations)
1. **Thiên lệch người ký (Signer Bias):** Khoảng ~94% tổng số mẫu do một người (`signer_A`) thực hiện. Mặc dù có thêm 2 người tham gia ở `session_03`, do mỗi người ký các lớp ký hiệu khác nhau, **không thể tách rời ảnh hưởng của người ký khỏi đặc trưng của lớp ký hiệu** (không thể so sánh độ chính xác giữa các người ký để rút ra kết luận).
2. **Nhạy cảm với góc camera và ánh sáng:** Độ chính xác sụt giảm mạnh khi kiểm thử trên các session có góc lệch (S2, S4 đạt ~29%).
3. **Các ký hiệu tĩnh dễ nhầm lẫn:** Các chữ cái có cấu trúc ngón tay tương đồng (ví dụ: `h`, `i`, `x`, `k`, `n`, `u`, `m`, `l`, `a`, `b`, `c`, `aa`, `dau_hoi`) có recall thấp do phụ thuộc vào độ sắc nét của ngón tay.
4. **Độ trễ cửa sổ:** Cử chỉ ký hiệu chiếm đa số khung hình trong cửa sổ 3.2 giây, dẫn đến độ trễ phản hồi thời gian thực từ 4 - 7 giây.

---

## 8. Hướng dẫn Kiểm tra Tính toàn vẹn (Integrity Verification)
Để kiểm tra tính toàn vẹn của dữ liệu sau khi tải về, sử dụng tệp `checksums.sha256`:

```powershell
# Windows PowerShell
Get-FileHash -Algorithm SHA256 (Get-ChildItem -Recurse -File | Where-Object {{$_.Name -ne 'checksums.sha256'}})
```

Hoặc chạy công cụ kiểm tra tự động đi kèm mã nguồn:
```powershell
python tools/verify_dataset.py dataset/raw
```

---

## 9. Giấy phép & Trích dẫn (License & Citation)

### Giấy phép (License)
Bộ dữ liệu được phát hành theo giấy phép **Creative Commons Attribution 4.0 International (CC BY 4.0)**. Bạn được tự do sao chép, phân phối và phát triển dựa trên dữ liệu này với điều kiện ghi nhận công trạng tác giả phù hợp.

### Trích dẫn (BibTeX)
```bibtex
@dataset{{vsl_landmarks_258_2026,
  author    = {{Đoàn Duy Bình}},
  title     = {{Vietnamese Sign Language (VSL) 258-Landmark Dataset}},
  year      = {{2026}},
  publisher = {{GitHub}},
  license   = {{CC BY 4.0}},
  note      = {{MediaPipe Holistic landmarks for 43 VSL classes across 4 distinct recording sessions}}
}}
```
"""

    with open(OUT_CARD, "w", encoding="utf-8") as f:
        f.write(card_content)
    print(f"✅ Đã tạo thành công DATASET_CARD.md tại: {OUT_CARD}")

if __name__ == '__main__':
    main()
