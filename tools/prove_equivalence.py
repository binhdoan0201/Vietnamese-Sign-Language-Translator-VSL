"""
tools/prove_equivalence.py
==========================
Công cụ nội bộ dành cho tác giả (cần dữ liệu thô 1662 chiều và gói công khai).
Chứng minh tính tương đương số học 100% giữa việc tiền xử lý từ
dataset/raw (1662 chiều) và dữ liệu công khai (258 chiều).
"""
import os
import sys
import shutil
import numpy as np
import subprocess

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PYTHON_EXE = sys.executable

PROCESSED_ORIGINAL = os.path.join(PROJECT_ROOT, "dataset", "processed")
TEMP_PROCESSED = os.path.join(PROJECT_ROOT, "temp_processed_verify")
PUBLIC_DATASET = os.path.join(PROJECT_ROOT, "release_assets", "dataset_public")


def main():
    print("=" * 80)
    print("BẮT ĐẦU CHỨNG MINH TƯƠNG ĐƯƠNG GIỮA DATASET RAW VÀ DATASET CÔNG KHAI")
    print("=" * 80)

    # 1. Chạy preprocess_data.py trên release_assets/dataset_public
    os.makedirs(TEMP_PROCESSED, exist_ok=True)
    cmd = [
        PYTHON_EXE,
        os.path.join(PROJECT_ROOT, "pipeline", "preprocess_data.py"),
        "--input-dir", PUBLIC_DATASET,
        "--output-dir", TEMP_PROCESSED,
        "--mode", "session",
        "--test-session", "4",
        "--seed", "42"
    ]
    print(f"Đang thực thi: {' '.join(cmd)}...")
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8")
    if res.returncode != 0:
        print("[FAIL] preprocess_data.py thất bại:")
        print(res.stderr)
        sys.exit(1)
    print("✓ preprocess_data.py trên dataset công khai hoàn tất thành công.")

    # 2. So sánh từng mảng
    arrays_to_check = [
        ("X_raw258_all.npy", "exact"),
        ("X_all.npy", "allclose"),
        ("y_all.npy", "exact"),
        ("session_all.npy", "exact"),
        ("X_train.npy", "allclose"),
        ("X_test.npy", "allclose"),
        ("y_train.npy", "exact"),
        ("y_test.npy", "exact")
    ]

    all_matched = True
    print("\n--- KẾT QUẢ SO SÁNH TỪNG MẢNG ---")
    for fname, check_type in arrays_to_check:
        orig_p = os.path.join(PROCESSED_ORIGINAL, fname)
        temp_p = os.path.join(TEMP_PROCESSED, fname)

        if not os.path.exists(orig_p):
            print(f"  ❌ Không tìm thấy file gốc: {orig_p}")
            all_matched = False
            continue
        if not os.path.exists(temp_p):
            print(f"  ❌ Không tìm thấy file sinh ra: {temp_p}")
            all_matched = False
            continue

        arr_orig = np.load(orig_p, allow_pickle=True)
        arr_temp = np.load(temp_p, allow_pickle=True)

        if arr_orig.shape != arr_temp.shape:
            print(f"  ❌ {fname:<20}: Lệch shape! Gốc={arr_orig.shape}, Mới={arr_temp.shape}")
            all_matched = False
            continue

        if check_type == "exact":
            match = np.array_equal(arr_orig, arr_temp)
            if match:
                print(f"  ✅ {fname:<20}: np.array_equal = True (Khớp chính xác 100%, shape {arr_orig.shape})")
            else:
                print(f"  ❌ {fname:<20}: np.array_equal = False!")
                all_matched = False
        else: # allclose atol=1e-6
            max_diff = np.max(np.abs(arr_orig - arr_temp))
            match = np.allclose(arr_orig, arr_temp, atol=1e-6)
            if match:
                print(f"  ✅ {fname:<20}: np.allclose = True (Max diff = {max_diff:.2e} <= 1e-6, shape {arr_orig.shape})")
            else:
                print(f"  ❌ {fname:<20}: np.allclose = False! Max diff = {max_diff:.2e}")
                all_matched = False

    # Dọn dẹp thư mục tạm
    shutil.rmtree(TEMP_PROCESSED, ignore_errors=True)

    print("=" * 80)
    if all_matched:
        print("🎉 KẾT LUẬN: TƯƠNG ĐƯƠNG HOÀN TOÀN 100%! BỘ DỮ LIỆU CÔNG KHAI 258 CHIỀU SỬ DỤNG HOÀN HẢO.")
    else:
        print("❌ KẾT LUẬN: PHÁT HIỆN SAI LỆCH! DỪNG VÀ BÁO CÁO LỖI.")
        sys.exit(1)


if __name__ == '__main__':
    main()
