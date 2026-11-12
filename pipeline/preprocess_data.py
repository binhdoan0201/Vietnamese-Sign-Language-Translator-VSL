"""
pipeline/preprocess_data.py
===========================
Đọc file .npy thô (30, 1662) -> cắt face -> chuẩn hóa -> đóng gói (N, 30, 384)
Hỗ trợ chia dữ liệu theo Session (mặc định) hoặc Ngẫu nhiên (random stratified).

Chạy lại file này MỖI KHI:
  - Thu thêm data mới
  - Sửa logic trong feature_utils.py

CÁCH DÙNG:
  # Chế độ chia theo session (mặc định test trên session_04):
  python pipeline/preprocess_data.py --mode session --test-session 4

  # Chế độ chia ngẫu nhiên (giữ tương thích cũ, stratify 10% test):
  python pipeline/preprocess_data.py --mode random
"""

import os
import sys
import random
import argparse
import numpy as np
from sklearn.model_selection import train_test_split

# Đảm bảo import được feature_utils từ thư mục gốc
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from feature_utils import slice_features, build_features, get_actions

RAW_DIR = os.path.join(PROJECT_ROOT, "dataset", "raw")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "dataset", "processed")
EXPECTED_RAW_SHAPE_1662 = (30, 1662)
EXPECTED_RAW_SHAPE_258 = (30, 258)


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    print(f"[SEED] Đã cố định random seed: {seed}")


def normalize_session_name(session_input):
    """
    Chuẩn hóa tên session:
      4 -> 'session_04'
      '4' -> 'session_04'
      'session_4' -> 'session_04'
      'session_04' -> 'session_04'
    """
    s = str(session_input).strip()
    if s.isdigit():
        return f"session_{int(s):02d}"
    if s.lower().startswith("session_"):
        parts = s.split("_", 1)
        if len(parts) == 2 and parts[1].isdigit():
            return f"session_{int(parts[1]):02d}"
    return s


def find_npy_files_with_session(action_path):
    """
    Quét đệ quy tìm tất cả file .npy bên trong action_path,
    trả về tuple (đường_dẫn_đầy_đủ, tên_session).
    """
    found = []
    for root, dirs, files in os.walk(action_path):
        for f in files:
            if f.endswith('.npy'):
                full_path = os.path.join(root, f)
                rel_path = os.path.relpath(full_path, action_path)
                parts = rel_path.split(os.sep)
                # Thường có cấu trúc: <session_id>/sample_XXX.npy
                if len(parts) > 1 and parts[0].startswith("session_"):
                    session_id = parts[0]
                else:
                    session_id = "session_unknown"
                found.append((full_path, session_id))
    found.sort(key=lambda x: x[0])
    return found


def parse_args():
    parser = argparse.ArgumentParser(description="Tiền xử lý và chia dataset VSL")
    parser.add_argument("--input-dir", type=str, default=RAW_DIR,
                        help="Đường dẫn thư mục dataset đầu vào (mặc định: dataset/raw)")
    parser.add_argument("--output-dir", type=str, default=PROCESSED_DIR,
                        help="Đường dẫn thư mục lưu kết quả tiền xử lý (mặc định: dataset/processed)")
    parser.add_argument("--mode", type=str, choices=["session", "random"], default="session",
                        help="Phương pháp chia dữ liệu: 'session' hoặc 'random'")
    parser.add_argument("--test-session", type=str, default="4",
                        help="Session dùng làm tập test khi --mode session (mặc định: 4)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed cho tính tái lập kết quả (mặc định: 42)")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    input_dir = os.path.abspath(args.input_dir)
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # ============================================================
    # 1. QUÉT VÀ TRÍCH XUẤT DATASET
    # ============================================================
    actions = get_actions(input_dir)
    print(f"Thư mục đầu vào: {input_dir}")
    print(f"Thư mục đầu ra : {output_dir}")
    print(f"Phát hiện {len(actions)} classes: {actions}")

    label_map = {label: num for num, label in enumerate(actions)}

    sequences = []
    raw258_sequences = []
    labels = []
    session_ids = []
    skipped = 0

    # Để theo dõi phân bố (class x session)
    all_sessions_seen = set()
    stats = {action: {} for action in actions}

    print("\n--- ĐANG GOM VÀ TIỀN XỬ LÝ DỮ LIỆU ---")
    for action in actions:
        action_path = os.path.join(input_dir, action)
        n_loaded = 0

        for file_path, session_name in find_npy_files_with_session(action_path):
            file_name = os.path.relpath(file_path, action_path)
            all_sessions_seen.add(session_name)

            try:
                res = np.load(file_path)
            except Exception as e:
                print(f"  [SKIP] {action}/{file_name}: lỗi đọc file ({e})")
                skipped += 1
                continue

            # Kiểm tra và nhận diện shape (30, 1662) hoặc (30, 258)
            if res.shape == EXPECTED_RAW_SHAPE_1662:
                res_sliced = slice_features(res)        # (30, 1662) -> (30, 258)
            elif res.shape == EXPECTED_RAW_SHAPE_258:
                res_sliced = np.asarray(res, dtype=np.float32)  # Đã là 258 chiều
            else:
                print(f"  [SKIP] {action}/{file_name}: shape {res.shape} "
                      f"(kỳ vọng {EXPECTED_RAW_SHAPE_1662} hoặc {EXPECTED_RAW_SHAPE_258})")
                skipped += 1
                continue

            # Kiểm tra sample rỗng
            if np.count_nonzero(res) == 0:
                print(f"  [SKIP] {action}/{file_name}: sample rỗng (toàn 0)")
                skipped += 1
                continue

            res_final = build_features(res_sliced)  # (30, 258)  -> (30, 384)

            # Kiểm tra NaN/Inf sau khi chuẩn hóa (bắt lỗi chia 0)
            if np.isnan(res_final).any() or np.isinf(res_final).any():
                print(f"  [SKIP] {action}/{file_name}: chứa NaN/Inf sau normalize")
                skipped += 1
                continue

            sequences.append(res_final)
            raw258_sequences.append(res_sliced)
            labels.append(label_map[action])
            session_ids.append(session_name)

            stats[action][session_name] = stats[action].get(session_name, 0) + 1
            n_loaded += 1

        print(f"  [{action}] -> {n_loaded} samples hợp lệ")

    if len(sequences) == 0:
        print("\n[!] Không có sample nào hợp lệ. Kiểm tra lại dataset/raw")
        sys.exit(1)

    sorted_sessions = sorted(list(all_sessions_seen))

    # ============================================================
    # 2. BẢNG THỐNG KÊ (CLASS X SESSION) VÀ CẢNH BÁO
    # ============================================================
    print("\n" + "=" * 80)
    print("BẢNG THỐNG KÊ SỐ MẪU HỢP LỆ (CLASS x SESSION)")
    print("=" * 80)
    header = f"{'CLASS':<20}" + "".join(f"{s:>13}" for s in sorted_sessions) + f"{'TỔNG':>10}"
    print(header)
    print("-" * 80)

    warnings_list = []
    for action in actions:
        row = f"{action:<20}"
        total_class_samples = 0
        for s in sorted_sessions:
            cnt = stats[action].get(s, 0)
            total_class_samples += cnt
            row += f"{cnt:>13}"
            if cnt == 0:
                warnings_list.append(f"Class '{action}' THIẾU hoàn toàn '{s}' (0 mẫu)!")
            elif cnt < 30:
                warnings_list.append(f"Class '{action}' thiếu mẫu ở '{s}' ({cnt}/30 mẫu)!")
        row += f"{total_class_samples:>10}"
        print(row)
    print("=" * 80)

    if warnings_list:
        print("\n⚠️  CÁC CẢNH BÁO VỀ SỐ LƯỢNG MẪU VÀ SESSION:")
        for w in warnings_list:
            print(f"  - {w}")
    else:
        print("\n✅ Toàn bộ các class đều đủ 4 session và đủ ít nhất 30 sample/session.")

    # ============================================================
    # 3. ĐÓNG GÓI BẢN ĐẦY ĐỦ (X_all, y_all, session_all)
    # ============================================================
    X_all = np.array(sequences, dtype=np.float32)
    X_raw258_all = np.array(raw258_sequences, dtype=np.float32)
    y_all = np.eye(len(actions))[labels].astype(int)       # one-hot
    session_all = np.array(session_ids, dtype=object)

    print("\n--- THÔNG SỐ TOÀN BỘ DATASET ---")
    print(f"Kích thước X_all      : {X_all.shape} (kỳ vọng: N, 30, 384)")
    print(f"Kích thước X_raw258   : {X_raw258_all.shape} (kỳ vọng: N, 30, 258)")
    print(f"Kích thước y_all      : {y_all.shape}")
    print(f"Kích thước session_all: {session_all.shape}")
    print(f"Tổng sample bị bỏ qua : {skipped}")

    np.save(os.path.join(output_dir, 'X_all.npy'), X_all)
    np.save(os.path.join(output_dir, 'X_raw258_all.npy'), X_raw258_all)
    np.save(os.path.join(output_dir, 'y_all.npy'), y_all)
    np.save(os.path.join(output_dir, 'session_all.npy'), session_all)
    print(f"✓ Đã lưu X_all.npy, X_raw258_all.npy, y_all.npy, session_all.npy vào {output_dir}")

    # ============================================================
    # 4. PHÂN CHIA TẬP TRAIN / TEST
    # ============================================================
    if args.mode == "session":
        target_test_session = normalize_session_name(args.test_session)
        print(f"\n--- PHÂN CHIA THEO SESSION (Test: {target_test_session}) ---")

        test_mask = (session_all == target_test_session)
        train_mask = ~test_mask

        if np.sum(test_mask) == 0:
            print(f"[!] LỖI: Không tìm thấy mẫu nào thuộc session '{target_test_session}'!")
            print(f"    Các session hiện có: {sorted_sessions}")
            sys.exit(1)

        X_train, y_train = X_all[train_mask], y_all[train_mask]
        X_test, y_test = X_all[test_mask], y_all[test_mask]

        train_sessions = sorted(list(set(session_all[train_mask])))
        print(f"Train sessions: {train_sessions}")
        print(f"Test session  : ['{target_test_session}']")

    else:
        print("\n--- PHÂN CHIA NGẪU NHIÊN CÓ PHÂN TẦNG (Stratified 90/10) ---")
        y_int = np.array(labels)
        X_train, X_test, y_train, y_test = train_test_split(
            X_all, y_all,
            test_size=0.1,
            random_state=args.seed,
            stratify=y_int
        )

    print(f"Train samples: {X_train.shape[0]}")
    print(f"Test samples : {X_test.shape[0]}")

    np.save(os.path.join(output_dir, 'X_train.npy'), X_train)
    np.save(os.path.join(output_dir, 'X_test.npy'), X_test)
    np.save(os.path.join(output_dir, 'y_train.npy'), y_train)
    np.save(os.path.join(output_dir, 'y_test.npy'), y_test)

    print(f"\n✅ Hoàn tất tiền xử lý và lưu dữ liệu vào {output_dir}")


if __name__ == "__main__":
    main()
