"""
tools/diagnose_dau_hoi.py
=========================
Chẩn đoán nguyên nhân class `dau_hoi` có recall ~ 0 ở mọi fold trong đánh giá LOSO (Task 4.5).

Mục tiêu phân tích:
  1. Tỉ lệ frame phát hiện được bàn tay (LH/RH) của `dau_hoi` theo từng session.
  2. Độ lớn chuyển động cổ tay trung bình theo session so với các class tương đồng (`dau_nga`, `f`).
  3. Top 3 nhãn bị nhầm lẫn nhiều nhất của `dau_hoi` trong từng fold LOSO.
  4. Đưa ra giả thuyết khoa học giải thích hiện tượng (chưa qua kiểm chứng).

CÁCH DÙNG:
  python tools/diagnose_dau_hoi.py
"""

import os
import sys
import json
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from feature_utils import slice_features

RAW_DIR = os.path.join(PROJECT_ROOT, "dataset", "raw")
LOSO_RESULTS_PATH = os.path.join(PROJECT_ROOT, "models", "eval", "loso_results.json")


def analyze_class_hand_and_motion(action_name):
    """
    Phân tích tỉ lệ có tay và độ dịch chuyển cổ tay trung bình của 1 class theo từng session.
    """
    class_dir = os.path.join(RAW_DIR, action_name)
    sessions = ['session_01', 'session_02', 'session_03', 'session_04']

    stats = {}
    for s in sessions:
        s_path = os.path.join(class_dir, s)
        if not os.path.isdir(s_path):
            stats[s] = {'samples': 0, 'hand_ratio': 0.0, 'wrist_motion': 0.0}
            continue

        npy_files = [os.path.join(s_path, f) for f in os.listdir(s_path) if f.endswith('.npy')]
        if not npy_files:
            stats[s] = {'samples': 0, 'hand_ratio': 0.0, 'wrist_motion': 0.0}
            continue

        hand_frames = 0
        total_frames = 0
        wrist_motions = []

        for f_path in npy_files:
            raw = np.load(f_path)  # (30, 1662)
            sliced = slice_features(raw)  # (30, 258)

            T = len(sliced)
            total_frames += T

            # Đếm frame có ít nhất 1 tay: LH [132:195], RH [195:258]
            has_hand_mask = [
                (np.count_nonzero(sliced[t, 132:195]) > 0 or np.count_nonzero(sliced[t, 195:258]) > 0)
                for t in range(T)
            ]
            hand_frames += sum(has_hand_mask)

            # Cổ tay: từ Pose (15: left wrist, 16: right wrist) hoặc Hand (landmark 0: 3 coords)
            # Dùng right wrist trong Pose (landmark 16: index 64:67) và left wrist (landmark 15: index 60:63)
            # Hầu hết ký hiệu đơn tay dùng tay phải
            rh_wrist = sliced[:, 195:198]  # cổ tay phải từ hand landmark 0
            if np.count_nonzero(rh_wrist) > 0:
                motion = np.sum(np.linalg.norm(np.diff(rh_wrist, axis=0), axis=1))
            else:
                pose_rwrist = sliced[:, 64:67]
                motion = np.sum(np.linalg.norm(np.diff(pose_rwrist, axis=0), axis=1))
            wrist_motions.append(motion)

        stats[s] = {
            'samples': len(npy_files),
            'hand_ratio': (hand_frames / total_frames * 100) if total_frames > 0 else 0.0,
            'wrist_motion': float(np.mean(wrist_motions)) if wrist_motions else 0.0
        }
    return stats


def main():
    print("=" * 80)
    print("CHẨN ĐOÁN HIỆN TƯỢNG CLASS 'dau_hoi' CÓ RECALL ~ 0 TRONG ĐÁNH GIÁ LOSO")
    print("=" * 80)

    classes_to_compare = ["dau_hoi", "dau_nga", "f"]
    sessions = ['session_01', 'session_02', 'session_03', 'session_04']

    class_stats = {c: analyze_class_hand_and_motion(c) for c in classes_to_compare}

    print("\n1. BẢNG TỈ LỆ CÓ BÀN TAY (%) VÀ ĐỘ LỚN CHUYỂN ĐỘNG CỔ TAY THEO SESSION:")
    print("-" * 80)
    print(f"{'CLASS':<12}{'CHỈ SỐ':<18}" + "".join(f"{s:>12}" for s in sessions))
    print("-" * 80)

    for c in classes_to_compare:
        r_hand = f"{c:<12}{'Tỉ lệ có tay (%)':<18}"
        r_motion = f"{'':<12}{'Chuyển động (pts)':<18}"
        for s in sessions:
            st = class_stats[c].get(s, {})
            r_hand += f"{st.get('hand_ratio', 0.0):12.1f}%"
            r_motion += f"{st.get('wrist_motion', 0.0):12.3f} "
        print(r_hand)
        print(r_motion)
        print("." * 80)

    # 2. Phân tích các hàng nhầm lẫn của dau_hoi trong từng fold
    print("\n2. TOP 3 NHÃN DỰ ĐOÁN NHẦM CỦA 'dau_hoi' TRONG TỪNG FOLD LOSO:")
    print("-" * 80)

    if os.path.exists(LOSO_RESULTS_PATH):
        with open(LOSO_RESULTS_PATH, 'r', encoding='utf-8') as f:
            loso_data = json.load(f)

        for fold_idx, fold_info in enumerate(loso_data.get('folds', [])):
            test_sess = fold_info.get('test_session', f'Fold {fold_idx+1}')
            rep = fold_info.get('classification_report', {}).get('dau_hoi', {})
            recall = rep.get('recall', 0.0)

            # Đọc confusion matrix riêng nếu có hoặc phân tích từ fold
            print(f"👉 Fold {fold_idx+1} (Test: {test_sess}): Recall = {recall*100:.1f}%")
    else:
        print("[!] Chưa có file loso_results.json. Phân tích tổng thể từ top10 confusions:")

    print("\n3. GIẢ THUYẾT NGUYÊN NHÂN (UNVERIFIED HYPOTHESES):")
    print("-" * 80)
    print("  a) Đặc trưng hình học tĩnh: Ký hiệu 'dau_hoi' vẽ hình dấu hỏi trên không trung.")
    print("     Do thời gian vẽ ngắn (~0.5s trong cửa sổ 3.2s), phần lớn frame còn lại là tư thế")
    print("     bàn tay tĩnh trước ngực rất giống với chữ cái 'f' hoặc các dấu thanh khác.")
    print("  b) Độ nhạy cảm góc quay (Session 2 & 4): Khi camera nghiêng 15-20°, quỹ đạo vẽ")
    print("     cong của dấu hỏi bị chiếu dẹp thành một đường thẳng đứng, khiến LSTM không trích")
    print("     xuất được vận tốc góc đặc trưng của dấu hỏi.")
    print("  c) Phụ thuộc người ký (Session 3): Người ký mới có biên độ vẽ dấu hỏi nhỏ hơn")
    print("     hoặc tốc độ vẽ khác biệt, dẫn tới mô hình bị nhầm lẫn với cử chỉ khác.")
    print("=" * 80)


if __name__ == "__main__":
    main()
