"""
tools/analyze_weak_classes.py
=============================
Phân tích chuyên sâu nhóm chữ yếu và động học bàn tay qua các session (Task C1 & C2).

1. Task C1: Thống kê Recall từng class yếu qua các seed (42, 7, 123) cho từng session test (S1..S4).
   Danh sách class yếu: h, i, l, u, k, x, m, n, y, a, b, c, aa, dau_hoi, qq.
2. Task C2: Phân tích từ dữ liệu thô X_raw258_all.npy:
   - Tỉ lệ frame phát hiện có tay (hand presence ratio)
   - Tần suất nhảy tracking bất thường của cổ tay (wrist tracking jump: khoảng cách Euclide giữa 2 frame liên tiếp > ngưỡng JUMP_THRESHOLD = 0.15)
   theo từng session (S1..S4) cho từng class yếu.
"""

import os
import sys
import json
import numpy as np

# Ép stdout UTF-8
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
EVAL_DIR = os.path.join(PROJECT_ROOT, "models", "eval")
RAW_DIR = os.path.join(PROJECT_ROOT, "dataset", "raw")
RAW258_PATH = os.path.join(PROJECT_ROOT, "dataset", "processed", "X_raw258_all.npy")
SESSION_PATH = os.path.join(PROJECT_ROOT, "dataset", "processed", "session_all.npy")
Y_ALL_PATH = os.path.join(PROJECT_ROOT, "dataset", "processed", "y_all.npy")

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from feature_utils import get_actions
from pipeline.segment_utils import is_hand_present

TARGET_WEAK_CLASSES = [
    'h', 'i', 'l', 'u', 'k', 'x', 'm', 'n', 'y', 'a', 'b', 'c', 'aa', 'dau_hoi', 'qq'
]
JUMP_THRESHOLD = 0.15  # Ngưỡng dịch chuyển cổ tay bất thường (tọa độ chuẩn hóa [0, 1] / frame)


def analyze_c1_recalls(results_json_list, config_name="Baseline P2"):
    """
    Thống kê recall cho nhóm chữ yếu qua các seed và session test.
    """
    print("=" * 85)
    print(f"C1. BẢNG RECALL NHÓM CHỮ YẾU THEO SESSION TEST - {config_name}")
    print("=" * 85)

    data_by_seed = []
    for fpath in results_json_list:
        if os.path.exists(fpath):
            with open(fpath, "r", encoding="utf-8") as f:
                data_by_seed.append(json.load(f))
        else:
            print(f"[!] Thiếu file: {fpath}")

    if not data_by_seed:
        print("[!] Không có dữ liệu JSON để phân tích C1.")
        return {}

    header = f"{'Class':<12}{'S1 (Fold 1)':<14}{'S2 (Fold 2)':<14}{'S3 (Fold 3)':<14}{'S4 (Fold 4)':<14}{'Mean Recall':<14}"
    print(header)
    print("-" * len(header))

    class_summary = {}

    for cls in TARGET_WEAK_CLASSES:
        session_recalls = {1: [], 2: [], 3: [], 4: []}
        for d in data_by_seed:
            for fold in d["folds"]:
                fold_k = fold["fold"]
                cr = fold.get("classification_report", {})
                rec = cr.get(cls, {}).get("recall", 0.0) * 100.0
                session_recalls[fold_k].append(rec)

        s_means = [float(np.mean(session_recalls[k])) for k in range(1, 5)]
        mean_all = float(np.mean(s_means))
        class_summary[cls] = {
            'S1': s_means[0],
            'S2': s_means[1],
            'S3': s_means[2],
            'S4': s_means[3],
            'Mean': mean_all
        }

        row = f"{cls:<12}" + "".join(f"{s_means[k-1]:>6.1f}%{'':<7}" for k in range(1, 5)) + f"{mean_all:>6.1f}%"
        print(row)

    print("-" * len(header))
    # Trung bình cả nhóm chữ yếu
    group_s = [np.mean([class_summary[c][f'S{k}'] for c in TARGET_WEAK_CLASSES]) for k in range(1, 5)]
    group_mean = np.mean(group_s)
    summary_row = f"{'TRUNG BÌNH':<12}" + "".join(f"{group_s[k-1]:>6.1f}%{'':<7}" for k in range(1, 5)) + f"{group_mean:>6.1f}%"
    print(summary_row)
    print("=" * 85)

    return class_summary


def analyze_c2_tracking_quality():
    """
    Phân tích tỉ lệ frame có tay và số lần nhảy tracking của cổ tay từ X_raw258_all.npy.
    """
    print("\n" + "=" * 90)
    print("C2. PHÂN TÍCH ĐỘ ỔN ĐỊNH BÀN TAY (TỈ LỆ CÓ TAY & NHẢY TRACKING CỔ TAY)")
    print(f"    (Ngưỡng nhảy tracking bất thường: delta wrist > {JUMP_THRESHOLD} điểm ảnh chuẩn hóa / frame)")
    print("=" * 90)

    if not os.path.exists(RAW258_PATH) or not os.path.exists(SESSION_PATH) or not os.path.exists(Y_ALL_PATH):
        print("[!] Thiếu file dữ liệu tiền xử lý.")
        return {}

    X_raw = np.load(RAW258_PATH)       # (5166, 30, 258)
    sessions = np.load(SESSION_PATH, allow_pickle=True)   # (5166,)
    y_all = np.load(Y_ALL_PATH)        # (5166, 43)
    actions = get_actions(RAW_DIR)

    label_indices = np.argmax(y_all, axis=1)

    sess_names = ['session_01', 'session_02', 'session_03', 'session_04']
    sess_labels = ['S1', 'S2', 'S3', 'S4']

    # Header bảng
    print(f"{'Class':<10}" + "".join(f"{'[' + s + ' Hand%]':<11}{'[' + s + ' Jumps]':<11}" for s in sess_labels))
    print("-" * 98)

    stats = {}

    for cls in TARGET_WEAK_CLASSES:
        if cls not in actions:
            continue
        c_idx = actions.index(cls)
        stats[cls] = {}

        row_str = f"{cls:<10}"

        for s_name, s_lbl in zip(sess_names, sess_labels):
            mask = (label_indices == c_idx) & (sessions == s_name)
            sub_X = X_raw[mask]  # (N_samples, 30, 258)
            N = len(sub_X)

            if N == 0:
                stats[cls][s_lbl] = {'hand_ratio': 0.0, 'jumps_per_sample': 0.0}
                row_str += f"{'N/A':<11}{'N/A':<11}"
                continue

            # 1. Tỉ lệ frame có tay
            hand_frames = 0
            total_frames = N * 30
            for seq in sub_X:
                for fr in seq:
                    if is_hand_present(fr):
                        hand_frames += 1
            hand_ratio = (hand_frames / total_frames) * 100.0

            # 2. Số lần nhảy tracking cổ tay bất thường
            # Cổ tay trái: Pose landmark 15 (idx 15*4=60:63) hoặc Hand landmark 0 (132:135)
            # Cổ tay phải: Pose landmark 16 (idx 16*4=64:67) hoặc Hand landmark 0 (195:198)
            total_jumps = 0
            for seq in sub_X:
                # Tính khoảng cách giữa các frame liên tiếp của cổ tay
                # Dùng cổ tay phải (195:197) nếu có, hoặc cổ tay trái (132:134)
                for t in range(1, len(seq)):
                    fr_prev = seq[t-1]
                    fr_curr = seq[t]

                    has_lh_prev = np.count_nonzero(fr_prev[132:195]) > 0
                    has_lh_curr = np.count_nonzero(fr_curr[132:195]) > 0
                    has_rh_prev = np.count_nonzero(fr_prev[195:258]) > 0
                    has_rh_curr = np.count_nonzero(fr_curr[195:258]) > 0

                    jump_detected = False
                    if has_rh_prev and has_rh_curr:
                        dist_rh = np.linalg.norm(fr_curr[195:197] - fr_prev[195:197])
                        if dist_rh > JUMP_THRESHOLD:
                            jump_detected = True
                    if has_lh_prev and has_lh_curr:
                        dist_lh = np.linalg.norm(fr_curr[132:134] - fr_prev[132:134])
                        if dist_lh > JUMP_THRESHOLD:
                            jump_detected = True

                    if jump_detected:
                        total_jumps += 1

            jumps_per_sample = total_jumps / float(N)
            stats[cls][s_lbl] = {'hand_ratio': hand_ratio, 'jumps_per_sample': jumps_per_sample}

            row_str += f"{hand_ratio:>7.1f}%  {jumps_per_sample:>7.2f}  "

        print(row_str)

    print("-" * 98)
    # Trung bình qua các session
    avg_hand_by_sess = [np.mean([stats[c][s]['hand_ratio'] for c in TARGET_WEAK_CLASSES if c in stats]) for s in sess_labels]
    avg_jump_by_sess = [np.mean([stats[c][s]['jumps_per_sample'] for c in TARGET_WEAK_CLASSES if c in stats]) for s in sess_labels]
    mean_row = f"{'TRUNG BÌNH':<10}"
    for h, j in zip(avg_hand_by_sess, avg_jump_by_sess):
        mean_row += f"{h:>7.1f}%  {j:>7.2f}  "
    print(mean_row)
    print("=" * 90)

    print("\n👉 NHẬN XÉT SƠ BỘ VỀ TƯƠNG QUAN (CORRELATION):")
    print(f"  - Session 1: Tỉ lệ có tay TB = {avg_hand_by_sess[0]:.1f}%, Số lần nhảy tracking/mẫu = {avg_jump_by_sess[0]:.2f}")
    print(f"  - Session 2: Tỉ lệ có tay TB = {avg_hand_by_sess[1]:.1f}%, Số lần nhảy tracking/mẫu = {avg_jump_by_sess[1]:.2f}")
    print(f"  - Session 3: Tỉ lệ có tay TB = {avg_hand_by_sess[2]:.1f}%, Số lần nhảy tracking/mẫu = {avg_jump_by_sess[2]:.2f}")
    print(f"  - Session 4: Tỉ lệ có tay TB = {avg_hand_by_sess[3]:.1f}%, Số lần nhảy tracking/mẫu = {avg_jump_by_sess[3]:.2f}")

    return stats


def compare_c3_recalls(baseline_summary, aug_summary, aug_name="Augmentation"):
    """
    So sánh recall nhóm chữ yếu trước (Baseline) và sau Augmentation (Task C3).
    """
    print("\n" + "=" * 95)
    print(f"C3. SO SÁNH RECALL NHÓM CHỮ YẾU: BASELINE P2 vs {aug_name}")
    print("=" * 95)

    header = f"{'Class':<12}{'Baseline Mean':<16}{f'{aug_name} Mean':<20}{'Δ Mean Recall':<16}{'Nhận xét':<20}"
    print(header)
    print("-" * len(header))

    diffs = []
    for cls in TARGET_WEAK_CLASSES:
        b_mean = baseline_summary.get(cls, {}).get('Mean', 0.0)
        a_mean = aug_summary.get(cls, {}).get('Mean', 0.0)
        delta = a_mean - b_mean
        diffs.append(delta)

        if delta > 5.0:
            note = "Tăng rõ rệt (++)"
        elif delta > 0.0:
            note = "Tăng nhẹ (+)"
        elif delta < -5.0:
            note = "Giảm rõ rệt (--)"
        elif delta < 0.0:
            note = "Giảm nhẹ (-)"
        else:
            note = "Không đổi (=)"

        print(f"{cls:<12}{b_mean:>6.1f}%{'':<9}{a_mean:>6.1f}%{'':<13}{delta:>+6.1f}%{'':<9}{note:<20}")

    print("-" * len(header))
    avg_b = np.mean([baseline_summary.get(c, {}).get('Mean', 0.0) for c in TARGET_WEAK_CLASSES])
    avg_a = np.mean([aug_summary.get(c, {}).get('Mean', 0.0) for c in TARGET_WEAK_CLASSES])
    avg_delta = avg_a - avg_b
    print(f"{'TRUNG BÌNH':<12}{avg_b:>6.1f}%{'':<9}{avg_a:>6.1f}%{'':<13}{avg_delta:>+6.1f}%")
    print("=" * 95)


def main():
    import glob
    import argparse
    parser = argparse.ArgumentParser(description="Phân tích chữ yếu và động học (Task C)")
    parser.add_argument("--aug-pattern", type=str, default="loso_results_v4_rot_y_seed*.json",
                        help="Pattern file json augmentation để so sánh C3")
    parser.add_argument("--aug-name", type=str, default="rot_y",
                        help="Tên cấu hình augmentation")
    args = parser.parse_args()

    p2_baseline_files = [
        os.path.join(EVAL_DIR, f"loso_results_v3_P2_seed{s}.json") for s in [42, 7, 123]
    ]
    b_summary = analyze_c1_recalls(p2_baseline_files, "Baseline P2 (None Augment)")
    analyze_c2_tracking_quality()

    # Nếu có file augmentation khớp pattern thì chạy C3
    aug_files = sorted(glob.glob(os.path.join(EVAL_DIR, args.aug_pattern)))
    if aug_files:
        print(f"\n[+] Tìm thấy {len(aug_files)} file augmentation: {[os.path.basename(f) for f in aug_files]}")
        a_summary = analyze_c1_recalls(aug_files, f"{args.aug_name}")
        compare_c3_recalls(b_summary, a_summary, args.aug_name)
    else:
        print(f"\n[-] Chưa có file augmentation khớp pattern '{args.aug_pattern}' để so sánh C3.")


if __name__ == '__main__':
    main()
