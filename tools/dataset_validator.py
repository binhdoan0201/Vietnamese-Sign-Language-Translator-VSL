"""
tools/dataset_validator.py
==========================
Kiểm tra tính toàn vẹn và chất lượng của dataset thô (dataset/raw).
Hỗ trợ quét đệ quy các thư mục session_XX.

CÁCH DÙNG:
  python tools/dataset_validator.py
"""

import os
import sys
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_DATASET_PATH = os.path.join(PROJECT_ROOT, "dataset", "raw")
EXPECTED_SHAPE = (30, 1662)


def validate_dataset(dataset_path=None):
    if dataset_path is None:
        dataset_path = DEFAULT_DATASET_PATH

    print(f"--- BẮT ĐẦU KIỂM TRA DATASET TẠI: {dataset_path} ---")

    if not os.path.exists(dataset_path):
        print(f"[!] Lỗi: Không tìm thấy thư mục {dataset_path}")
        return False

    classes = sorted([d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))])

    if not classes:
        print("[!] Thư mục raw hiện đang trống.")
        return False

    total_valid = 0
    total_errors = 0
    class_stats = {}

    for sign_class in classes:
        class_path = os.path.join(dataset_path, sign_class)

        # Quét đệ quy tất cả file .npy
        npy_files = []
        for root, dirs, files in os.walk(class_path):
            for f in files:
                if f.endswith('.npy'):
                    npy_files.append(os.path.join(root, f))
        npy_files.sort()

        class_valid = 0
        class_errors = 0

        for sample_path in npy_files:
            rel_name = os.path.relpath(sample_path, class_path)
            try:
                data = np.load(sample_path)

                # 1. Kiểm tra shape (30, 1662)
                if data.shape != EXPECTED_SHAPE:
                    print(f"  [X] Sai Shape - {sign_class}/{rel_name}: {data.shape} (Kỳ vọng: {EXPECTED_SHAPE})")
                    class_errors += 1
                    continue

                # 2. Kiểm tra NaN / Inf
                if np.isnan(data).any() or np.isinf(data).any():
                    print(f"  [X] Lỗi NaN/Inf - {sign_class}/{rel_name}")
                    class_errors += 1
                    continue

                # 3. Kiểm tra sequence rỗng (không bắt được điểm nào)
                if np.count_nonzero(data) == 0:
                    print(f"  [!] Cảnh báo mẫu rỗng (toàn 0) - {sign_class}/{rel_name}")
                    class_errors += 1
                    continue

                class_valid += 1

            except Exception as e:
                print(f"  [X] Không thể đọc file {sign_class}/{rel_name}: {e}")
                class_errors += 1

        total_valid += class_valid
        total_errors += class_errors
        class_stats[sign_class] = (class_valid, len(npy_files))

    print("\n" + "=" * 60)
    print(f"{'TỪ VỰNG':<25}{'HỢP LỆ / TỔNG SỐ':<20}{'TRẠNG THÁI'}")
    print("-" * 60)
    for c, (v, total) in class_stats.items():
        status = "✅ OK" if (v == total and total > 0) else "⚠️ Cần kiểm tra"
        print(f"{c:<25}{f'{v}/{total}':<20}{status}")

    print("=" * 60)
    print(f"Tổng số sample hợp lệ : {total_valid}")
    print(f"Tổng số lỗi / cảnh báo: {total_errors}")

    if total_errors == 0 and total_valid > 0:
        print("\n✅ Tuyệt vời! Toàn bộ dataset đều hợp lệ và đúng chuẩn (30, 1662).")
        return True
    else:
        print("\n⚠️ Có lỗi trong dataset. Hãy xem xét quay lại hoặc xóa các sample lỗi ở trên.")
        return False


if __name__ == "__main__":
    target_path = sys.argv[1] if len(sys.argv) > 1 else None
    validate_dataset(target_path)
