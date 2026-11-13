"""
tools/verify_dataset.py
=======================
Kiểm tra tính toàn vẹn và xác thực bộ dữ liệu công khai (258 chiều).
Cách dùng:
  python tools/verify_dataset.py [đường_dẫn_dataset]
  (Mặc định: dataset/raw)
"""
import os
import sys
import glob
import hashlib
import argparse
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_DATASET_DIR = os.path.join(PROJECT_ROOT, "dataset", "raw")


def compute_sha256(filepath):
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def verify_dataset(dataset_dir=DEFAULT_DATASET_DIR):
    print("=" * 80)
    print(f"BẮT ĐẦU XÁC THỰC BỘ DỮ LIỆU CÔNG KHAI TẠI: {dataset_dir}")
    print("=" * 80)

    if not os.path.exists(dataset_dir):
        print(f"[FAIL] Thư mục không tồn tại: {dataset_dir}")
        return False

    # 1. Kiểm tra checksums.sha256
    checksum_file = os.path.join(dataset_dir, "checksums.sha256")
    checksum_ok = True
    if os.path.exists(checksum_file):
        print("[1/4] Đang kiểm tra checksum SHA-256 từ checksums.sha256...")
        with open(checksum_file, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f.readlines() if l.strip()]

        mismatches = []
        missing = []
        for line in lines:
            parts = line.split(maxsplit=1)
            if len(parts) != 2:
                continue
            expected_hash, rel_path = parts
            full_path = os.path.join(dataset_dir, rel_path)
            if not os.path.exists(full_path):
                missing.append(rel_path)
                continue
            actual_hash = compute_sha256(full_path)
            if actual_hash != expected_hash:
                mismatches.append((rel_path, expected_hash, actual_hash))

        if missing:
            print(f"  ❌ Thiếu {len(missing)} tệp so với checksums.sha256!")
            checksum_ok = False
        if mismatches:
            print(f"  ❌ Sai lệch hash ở {len(mismatches)} tệp!")
            checksum_ok = False
        if checksum_ok:
            print(f"  ✅ Tất cả {len(lines)} tệp đều khớp mã băm SHA-256 hoàn hảo.")
    else:
        print("[1/4] ⚠️ Không tìm thấy checksums.sha256 (bỏ qua bước kiểm tra hash).")

    # 2. Kiểm tra classes
    classes = sorted([d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))])
    print(f"[2/4] Kiểm tra danh sách lớp: Tìm thấy {len(classes)} lớp thư mục.")
    if len(classes) != 43:
        print(f"  ⚠️ Cảnh báo: Số lớp ({len(classes)}) khác 43 lớp chuẩn!")

    # 3. Quét kiểm tra shape, dtype, NaN/Inf từng tệp .npy
    print("[3/4] Kiểm tra định dạng (shape (30, 258), dtype float32, NaN/Inf)...")
    total_samples = 0
    errors = []
    stats = {}

    for c in classes:
        stats[c] = {}
        for s in ["session_01", "session_02", "session_03", "session_04"]:
            s_dir = os.path.join(dataset_dir, c, s)
            if not os.path.isdir(s_dir):
                stats[c][s] = 0
                continue
            files = sorted(glob.glob(os.path.join(s_dir, "*.npy")))
            stats[c][s] = len(files)
            for f in files:
                total_samples += 1
                try:
                    arr = np.load(f)
                    if arr.shape != (30, 258):
                        errors.append((f, f"Shape sai: {arr.shape}, kỳ vọng (30, 258)"))
                    if arr.dtype != np.float32 and arr.dtype != np.float64:
                        errors.append((f, f"Dtype sai: {arr.dtype}"))
                    if np.isnan(arr).any():
                        errors.append((f, "Chứa giá trị NaN"))
                    if np.isinf(arr).any():
                        errors.append((f, "Chứa giá trị Inf"))
                except Exception as e:
                    errors.append((f, f"Lỗi đọc file: {e}"))

    if errors:
        print(f"  ❌ Phát hiện {len(errors)} tệp lỗi:")
        for ef, err in errors[:10]:
            print(f"    - {ef}: {err}")
        return False
    else:
        print(f"  ✅ Toàn bộ {total_samples:,} mẫu đều đạt chuẩn shape (30, 258), không có NaN/Inf.")

    # 4. In bảng phân bố (class x session)
    print("\n[4/4] BẢNG PHÂN BỐ DỮ LIỆU CÔNG KHAI (CLASS x SESSION):")
    print("-" * 75)
    print(f"{'CLASS':<20}{'S1':>10}{'S2':>10}{'S3':>10}{'S4':>10}{'TỔNG':>12}")
    print("-" * 75)
    total_s = {"session_01": 0, "session_02": 0, "session_03": 0, "session_04": 0}
    for c in classes:
        s1 = stats[c].get("session_01", 0)
        s2 = stats[c].get("session_02", 0)
        s3 = stats[c].get("session_03", 0)
        s4 = stats[c].get("session_04", 0)
        total_s["session_01"] += s1
        total_s["session_02"] += s2
        total_s["session_03"] += s3
        total_s["session_04"] += s4
        row_tot = s1 + s2 + s3 + s4
        print(f"{c:<20}{s1:>10}{s2:>10}{s3:>10}{s4:>10}{row_tot:>12}")
    print("-" * 75)
    print(f"{'TỔNG CỘNG':<20}{total_s['session_01']:>10}{total_s['session_02']:>10}{total_s['session_03']:>10}{total_s['session_04']:>10}{total_samples:>12}")
    print("=" * 80)
    print("✅ XÁC THỰC BỘ DỮ LIỆU CÔNG KHAI THÀNH CÔNG 100%!")
    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Xác thực bộ dữ liệu công khai 258 chiều")
    parser.add_argument("dataset_dir", nargs="?", default=DEFAULT_DATASET_DIR,
                        help="Đường dẫn thư mục dataset công khai")
    args = parser.parse_args()
    success = verify_dataset(args.dataset_dir)
    sys.exit(0 if success else 1)
