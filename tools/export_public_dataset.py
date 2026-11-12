"""
tools/export_public_dataset.py
==============================
Xuất bộ dữ liệu công khai 258 chiều (loại bỏ hoàn toàn điểm khuôn mặt).
Đọc từ dataset/raw/ (30, 1662), cắt bằng feature_utils.slice_features() -> (30, 258),
lưu vào release_assets/dataset_public/<class>/session_0X/sample_XXX.npy dạng float32.

Hỗ trợ:
  --input-dir         : Đường dẫn dataset gốc (mặc định: dataset/raw)
  --output-dir        : Thư mục xuất (mặc định: release_assets/dataset_public)
  --exclude-signers   : Danh sách mã người ký cần loại bỏ (ví dụ: signer_B,signer_C).
                        Chỉ hoạt động khi có file incoming/signer_map.csv xác nhận.
"""
import os
import sys
import glob
import json
import hashlib
import argparse
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from feature_utils import slice_features, get_actions

DEFAULT_INPUT_DIR = os.path.join(PROJECT_ROOT, "dataset", "raw")
DEFAULT_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "release_assets", "dataset_public")
SIGNER_MAP_PATH = os.path.join(PROJECT_ROOT, "incoming", "signer_map.csv")


def compute_sha256(filepath):
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def export_dataset(input_dir=DEFAULT_INPUT_DIR, output_dir=DEFAULT_OUTPUT_DIR, exclude_signers=None):
    print("=" * 80)
    print("BẮT ĐẦU XUẤT BỘ DỮ LIỆU CÔNG KHAI (258 CHIỀU, KHÔNG CÓ ĐIỂM KHUÔN MẶT)")
    print(f"Thư mục nguồn : {input_dir}")
    print(f"Thư mục đích  : {output_dir}")
    print("=" * 80)

    # Kiểm tra exclude-signers
    signer_mapping = {}
    if exclude_signers:
        if not os.path.exists(SIGNER_MAP_PATH):
            print(f"[LỖI] Cờ --exclude-signers yêu cầu bản đồ người ký đã xác nhận tại '{SIGNER_MAP_PATH}', nhưng file không tồn tại!")
            sys.exit(1)
        # Đọc signer_map
        df_map = pd.read_csv(SIGNER_MAP_PATH)
        for _, row in df_map.iterrows():
            signer_mapping[(row['class'], row['session'])] = row['signer']
        print(f"[CẤU HÌNH] Đã nạp nhãn người ký. Sẽ loại bỏ các signer: {exclude_signers}")

    classes = get_actions(input_dir)
    print(f"Phát hiện {len(classes)} lớp hành động.")

    os.makedirs(output_dir, exist_ok=True)

    metadata_rows = []
    total_processed = 0
    total_skipped = 0
    errors = []

    for c in classes:
        class_dir = os.path.join(input_dir, c)
        if not os.path.isdir(class_dir):
            continue

        sessions = sorted([s for s in os.listdir(class_dir) if os.path.isdir(os.path.join(class_dir, s))])
        for sess in sessions:
            sess_dir = os.path.join(class_dir, sess)
            out_sess_dir = os.path.join(output_dir, c, sess)
            os.makedirs(out_sess_dir, exist_ok=True)

            # Xác định người ký cho session/class này
            if sess in ["session_01", "session_02", "session_04"]:
                signer = "signer_A"
            else: # session_03
                if (c, sess) in signer_mapping:
                    signer = signer_mapping[(c, sess)]
                else:
                    # Chưa có nhãn xác nhận
                    signer = "multiple_unlabeled"

            # Kiểm tra nếu cần loại bỏ
            if exclude_signers and signer in exclude_signers:
                print(f"[LOẠI BỎ] Bỏ qua {c}/{sess} vì thuộc {signer}")
                continue

            files = sorted(glob.glob(os.path.join(sess_dir, "*.npy")))
            for f in files:
                fname = os.path.basename(f)
                out_path = os.path.join(out_sess_dir, fname)

                try:
                    data = np.load(f)
                    if data.shape != (30, 1662):
                        errors.append((f, f"Shape sai: {data.shape}, kỳ vọng (30, 1662)"))
                        total_skipped += 1
                        continue

                    # Cắt đặc trưng dùng slice_features (pose 132 + lh 63 + rh 63 = 258)
                    data_258 = slice_features(data)
                    data_258 = np.asarray(data_258, dtype=np.float32)

                    if data_258.shape != (30, 258):
                        errors.append((f, f"Shape sau cắt sai: {data_258.shape}"))
                        total_skipped += 1
                        continue

                    # Lưu file 258 chiều
                    np.save(out_path, data_258)
                    total_processed += 1

                    sample_name = os.path.splitext(fname)[0]
                    rel_path = os.path.relpath(out_path, output_dir).replace("\\", "/")
                    metadata_rows.append({
                        "class": c,
                        "session": sess,
                        "sample": sample_name,
                        "file": rel_path,
                        "signer": signer
                    })

                except Exception as e:
                    errors.append((f, str(e)))
                    total_skipped += 1

    print("\n[KẾT QUẢ XUẤT] Hoàn tất xuất dữ liệu:")
    print(f"  - Thành công : {total_processed} files (shape (30, 258), float32)")
    print(f"  - Bỏ qua/Lỗi : {total_skipped} files")
    if errors:
        print("[DANH SÁCH LỖI]:")
        for ef, err in errors[:10]:
            print(f"    {ef}: {err}")

    # Ghi metadata.csv
    meta_df = pd.DataFrame(metadata_rows)
    meta_path = os.path.join(output_dir, "metadata.csv")
    meta_df.to_csv(meta_path, index=False, encoding="utf-8")
    print(f"[METADATA] Đã ghi metadata.csv ({len(meta_df)} dòng) tại {meta_path}")

    # Ghi classes.json
    classes_path = os.path.join(output_dir, "classes.json")
    with open(classes_path, "w", encoding="utf-8") as f:
        json.dump(classes, f, indent=2, ensure_ascii=False)
    print(f"[CLASSES] Đã ghi classes.json ({len(classes)} classes) tại {classes_path}")

    # Ghi sessions.json
    sessions_desc = {
        "session_01": {
            "name": "Chính diện, điều kiện chuẩn",
            "camera_angle": "0 độ (chính diện ngang tầm ngực/vai)",
            "lighting": "Ánh sáng đèn phòng trong nhà ổn định",
            "distance": "Khoảng cách trung bình (~1.0m - 1.2m)",
            "signers": ["signer_A"]
        },
        "session_02": {
            "name": "Góc lệch trái, ánh sáng ban ngày",
            "camera_angle": "Lệch trái khoảng 15-20 độ",
            "lighting": "Ánh sáng tự nhiên ban ngày từ cửa sổ",
            "distance": "Khoảng cách trung bình (~1.0m - 1.2m)",
            "signers": ["signer_A"]
        },
        "session_03": {
            "name": "Khoảng cách gần, đa người ký",
            "camera_angle": "Chính diện",
            "lighting": "Ánh sáng phòng bình thường",
            "distance": "Khoảng cách gần camera (~0.6m - 0.8m)",
            "signers": ["signer_A", "signer_B", "signer_C"],
            "note": "Session thu thập với 3 người ký (signer_A, signer_B ~240 mẫu, signer_C ~90 mẫu). Mỗi người ký trọn từng class."
        },
        "session_04": {
            "name": "Góc lệch phải, ánh sáng ngược",
            "camera_angle": "Lệch phải khoảng 15-20 độ",
            "lighting": "Ánh sáng yếu / ngược sáng nhẹ",
            "distance": "Khoảng cách trung bình (~1.0m - 1.2m)",
            "signers": ["signer_A"]
        }
    }
    sess_path = os.path.join(output_dir, "sessions.json")
    with open(sess_path, "w", encoding="utf-8") as f:
        json.dump(sessions_desc, f, indent=2, ensure_ascii=False)
    print(f"[SESSIONS] Đã ghi sessions.json tại {sess_path}")

    # Ghi LICENSE-DATA (CC BY 4.0)
    lic_path = os.path.join(output_dir, "LICENSE-DATA")
    with open(lic_path, "w", encoding="utf-8") as f:
        f.write("Creative Commons Attribution 4.0 International (CC BY 4.0)\n\n"
                "Tác giả dữ liệu: Đoàn Duy Bình (2026)\n"
                "Bạn được tự do chia sẻ và điều chỉnh dữ liệu này cho bất kỳ mục đích nào,\n"
                "kể cả thương mại, với điều kiện ghi công tác giả thích hợp và dẫn liên kết đến giấy phép.\n")
    print(f"[LICENSE-DATA] Đã tạo LICENSE-DATA (CC BY 4.0) tại {lic_path}")

    # Sinh checksums.sha256 cho toàn bộ file trong output_dir (trừ checksums.sha256 chính nó)
    print("\n[CHECKSUM] Đang tính SHA-256 cho tất cả các file công khai...")
    checksum_lines = []
    for root, _, files in os.walk(output_dir):
        for f in sorted(files):
            if f == "checksums.sha256":
                continue
            fp = os.path.join(root, f)
            rel = os.path.relpath(fp, output_dir).replace("\\", "/")
            sha = compute_sha256(fp)
            checksum_lines.append(f"{sha}  {rel}")

    checksum_path = os.path.join(output_dir, "checksums.sha256")
    with open(checksum_path, "w", encoding="utf-8") as f:
        f.write("\n".join(checksum_lines) + "\n")
    print(f"[CHECKSUM] Đã tạo checksums.sha256 với {len(checksum_lines)} mục tại {checksum_path}")

    return total_processed, total_skipped


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Xuất bộ dữ liệu công khai 258 chiều")
    parser.add_argument("--input-dir", default=DEFAULT_INPUT_DIR, help="Đường dẫn dataset gốc")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Thư mục xuất dữ liệu")
    parser.add_argument("--exclude-signers", default=None, help="Danh sách signer cần loại, phân tách bởi dấu phẩy")
    args = parser.parse_args()

    exclude_list = [s.strip() for s in args.exclude_signers.split(",")] if args.exclude_signers else None
    export_dataset(args.input_dir, args.output_dir, exclude_list)
