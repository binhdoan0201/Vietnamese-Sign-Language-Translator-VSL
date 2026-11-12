"""Phân tích độ dài hoạt động thật của cử chỉ (Gesture Span Analysis) trên X_raw258_all.npy.

Tính toán:
  1. hand_present_span: khoảng frame đầu-cuối có ít nhất 1 tay.
  2. motion_span: đoạn có tốc độ cổ tay >= 20% max speed (và thử nghiệm độ nhạy 10%, 30%).
  3. Tỉ lệ thời gian đứng yên trong cửa sổ 30 frame.
  4. Quy đổi sang giây: frame * (3.2 / 30) (ƯỚC LƯỢNG).
  5. Phân loại class "động" (< 40% cửa sổ) và "tĩnh" (chữ cái / ít chuyển động).
  6. Phân tích Top-10 cặp nhầm lẫn và riêng class 'dau_hoi' theo session.
  7. Xuất biểu đồ histogram và boxplot ra models/eval/.
"""

import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from feature_utils import get_actions
from pipeline.segment_utils import is_hand_present, get_wrist_speeds

PROCESSED_PATH = os.path.join(PROJECT_ROOT, "dataset", "processed")
EVAL_DIR = os.path.join(PROJECT_ROOT, "models", "eval")
DT_FRAME_SEC = 3.2 / 30.0  # ƯỚC LƯỢNG: 0.1067s mỗi frame


def compute_sample_spans(sample_258):
    """Tính các khoảng thời gian cho 1 mẫu 258 chiều."""
    T = len(sample_258)
    # 1. Hand present span
    hand_frames = [t for t in range(T) if is_hand_present(sample_258[t])]
    if hand_frames:
        hand_span_frames = hand_frames[-1] - hand_frames[0] + 1
        hand_start, hand_end = hand_frames[0], hand_frames[-1]
    else:
        hand_span_frames = 0
        hand_start, hand_end = -1, -1

    # 2. Motion speeds
    v_smooth = get_wrist_speeds(sample_258)
    max_v = float(np.max(v_smooth))

    # Độ nhạy 10%, 20%, 30%
    motion_spans = {}
    for ratio in [0.10, 0.20, 0.30]:
        if max_v < 1e-4:
            motion_spans[ratio] = 0
        else:
            thresh = ratio * max_v
            active = [t for t in range(T) if v_smooth[t] >= thresh]
            motion_spans[ratio] = (active[-1] - active[0] + 1) if active else 0

    # 3. Tỉ lệ đứng yên (tính theo motion ratio 0.20)
    m_span_20 = motion_spans[0.20]
    stationary_ratio = max(0.0, 1.0 - (m_span_20 / float(T)))

    return {
        'hand_span_frames': hand_span_frames,
        'hand_span_sec': hand_span_frames * DT_FRAME_SEC,
        'hand_start': hand_start,
        'hand_end': hand_end,
        'motion_span_10': motion_spans[0.10],
        'motion_span_20': motion_spans[0.20],
        'motion_span_30': motion_spans[0.30],
        'motion_span_20_sec': m_span_20 * DT_FRAME_SEC,
        'stationary_ratio': stationary_ratio,
        'max_wrist_speed': max_v
    }


def main():
    print("=" * 80)
    print("PHÂN TÍCH ĐỘ DÀI HOẠT ĐỘNG THẬT CỦA CỬ CHỈ (GESTURE SPAN ANALYSIS)")
    print("=" * 80)

    X_raw_path = os.path.join(PROCESSED_PATH, "X_raw258_all.npy")
    y_all_path = os.path.join(PROCESSED_PATH, "y_all.npy")
    session_all_path = os.path.join(PROCESSED_PATH, "session_all.npy")

    if not os.path.exists(X_raw_path):
        print(f"[!] Không tìm thấy {X_raw_path}")
        return

    X_raw = np.load(X_raw_path)
    y_all = np.load(y_all_path)
    session_all = np.load(session_all_path, allow_pickle=True)
    actions = get_actions(os.path.join(PROJECT_ROOT, "dataset", "raw"))

    N = len(X_raw)
    y_indices = np.argmax(y_all, axis=1)
    print(f"Tổng số mẫu phân tích: {N} mẫu | Số lớp: {len(actions)}")

    records = []
    for i in range(N):
        spans = compute_sample_spans(X_raw[i])
        spans['index'] = i
        spans['class'] = actions[y_indices[i]]
        spans['session'] = session_all[i]
        records.append(spans)

    # 1. Phân bố tổng thể
    hand_spans = np.array([r['hand_span_frames'] for r in records])
    motion_10 = np.array([r['motion_span_10'] for r in records])
    motion_20 = np.array([r['motion_span_20'] for r in records])
    motion_30 = np.array([r['motion_span_30'] for r in records])
    stat_ratios = np.array([r['stationary_ratio'] for r in records])

    print("\n" + "-" * 80)
    print("1. PHÂN BỐ ĐỘ DÀI HOẠT ĐỘNG TỔNG THỂ (N=5166):")
    print("-" * 80)
    print(f"{'Chỉ số':<30}{'P10':<12}{'Median':<12}{'P90':<12}{'Mean ± Std':<18}")
    print(f"{'Hand present span (frames)':<30}{np.percentile(hand_spans, 10):<12.1f}{np.median(hand_spans):<12.1f}{np.percentile(hand_spans, 90):<12.1f}{np.mean(hand_spans):.1f} ± {np.std(hand_spans):.1f}")
    print(f"{'Hand present span (giây)*':<30}{np.percentile(hand_spans*DT_FRAME_SEC, 10):<12.2f}{np.median(hand_spans*DT_FRAME_SEC):<12.2f}{np.percentile(hand_spans*DT_FRAME_SEC, 90):<12.2f}{np.mean(hand_spans*DT_FRAME_SEC):.2f} ± {np.std(hand_spans*DT_FRAME_SEC):.2f}")
    print(f"{'Motion span 10% (frames)':<30}{np.percentile(motion_10, 10):<12.1f}{np.median(motion_10):<12.1f}{np.percentile(motion_10, 90):<12.1f}{np.mean(motion_10):.1f} ± {np.std(motion_10):.1f}")
    print(f"{'Motion span 20% (frames)':<30}{np.percentile(motion_20, 10):<12.1f}{np.median(motion_20):<12.1f}{np.percentile(motion_20, 90):<12.1f}{np.mean(motion_20):.1f} ± {np.std(motion_20):.1f}")
    print(f"{'Motion span 20% (giây)*':<30}{np.percentile(motion_20*DT_FRAME_SEC, 10):<12.2f}{np.median(motion_20*DT_FRAME_SEC):<12.2f}{np.percentile(motion_20*DT_FRAME_SEC, 90):<12.2f}{np.mean(motion_20*DT_FRAME_SEC):.2f} ± {np.std(motion_20*DT_FRAME_SEC):.2f}")
    print(f"{'Motion span 30% (frames)':<30}{np.percentile(motion_30, 10):<12.1f}{np.median(motion_30):<12.1f}{np.percentile(motion_30, 90):<12.1f}{np.mean(motion_30):.1f} ± {np.std(motion_30):.1f}")
    print(f"{'Tỉ lệ đứng yên (% cửa sổ)':<30}{np.percentile(stat_ratios*100, 10):<12.1f}{np.median(stat_ratios*100):<12.1f}{np.percentile(stat_ratios*100, 90):<12.1f}{np.mean(stat_ratios*100):.1f} ± {np.std(stat_ratios*100):.1f}")
    print("  (*) Ghi chú: Thời gian tính theo quy đổi ƯỚC LƯỢNG 3.2s / 30 frame (0.1067s/frame).")

    # 2. Phân bố theo Session
    print("\n" + "-" * 80)
    print("2. PHÂN BỐ ĐỘ DÀI CHUYỂN ĐỘNG (MOTION SPAN 20%) THEO SESSION:")
    print("-" * 80)
    sessions = ['session_01', 'session_02', 'session_03', 'session_04']
    print(f"{'Session':<15}{'Mẫu':<8}{'Median (frame)':<18}{'P10 - P90 (frame)':<22}{'Median (giây)*':<15}")
    for sess in sessions:
        s_m20 = np.array([r['motion_span_20'] for r in records if r['session'] == sess])
        print(f"{sess:<15}{len(s_m20):<8}{np.median(s_m20):<18.1f}{np.percentile(s_m20, 10):.1f} - {np.percentile(s_m20, 90):.1f}{' ':<10}{np.median(s_m20)*DT_FRAME_SEC:<15.2f}")

    # 3. Phân loại Class ĐỘNG vs TĨNH
    print("\n" + "-" * 80)
    print("3. PHÂN LOẠI CLASS ĐỘNG VS CLASS TĨNH:")
    print("-" * 80)
    class_stats = {}
    for action in actions:
        c_m20 = np.array([r['motion_span_20'] for r in records if r['class'] == action])
        c_speed = np.array([r['max_wrist_speed'] for r in records if r['class'] == action])
        c_hand = np.array([r['hand_span_frames'] for r in records if r['class'] == action])
        med_m20 = float(np.median(c_m20))
        med_speed = float(np.median(c_speed))
        class_stats[action] = {
            'median_m20_frames': med_m20,
            'median_m20_ratio': med_m20 / 30.0,
            'median_speed': med_speed,
            'median_hand_frames': float(np.median(c_hand))
        }

    # Class ĐỘNG: hoạt động < 40% cửa sổ (< 12 frames)
    dynamic_classes = [a for a, s in class_stats.items() if s['median_m20_ratio'] < 0.40]
    # Lấy các chữ cái alphabet có tốc độ thấp nhất (12 class tĩnh/ít chuyển động)
    sorted_by_speed = sorted(class_stats.items(), key=lambda x: x[1]['median_speed'])
    low_motion_classes = [a for a, _ in sorted_by_speed[:12]]

    print(f"👉 Danh sách class 'ĐỘNG' (chuyển động tập trung < 40% cửa sổ ~ <1.28s) [{len(dynamic_classes)} class]:")
    print("   " + ", ".join(dynamic_classes))
    print(f"\n👉 Danh sách class 'TĨNH' / ÍT CHUYỂN ĐỘNG (chữ cái / tư thế ngón tay đứng yên) [{len(low_motion_classes)} class]:")
    print("   " + ", ".join(low_motion_classes))

    # 4. Phân tích riêng Class 'dau_hoi' theo từng session
    print("\n" + "-" * 80)
    print("4. PHÂN TÍCH RIÊNG CLASS 'dau_hoi' QUA 4 SESSIONS:")
    print("-" * 80)
    print(f"{'Session':<15}{'Mẫu':<8}{'Hand Span (fr)':<18}{'Motion Span (fr)':<18}{'Max Speed (pts)':<18}")
    for sess in sessions:
        dh_recs = [r for r in records if r['class'] == 'dau_hoi' and r['session'] == sess]
        h_med = np.median([r['hand_span_frames'] for r in dh_recs])
        m_med = np.median([r['motion_span_20'] for r in dh_recs])
        s_med = np.median([r['max_wrist_speed'] for r in dh_recs])
        print(f"{sess:<15}{len(dh_recs):<8}{h_med:<18.1f}{m_med:<18.1f}{s_med:<18.3f}")

    # 5. Phân tích Top-10 cặp nhầm lẫn trong LOSO Seed 42
    print("\n" + "-" * 80)
    print("5. ĐẶC ĐIỂM HOẠT ĐỘNG CỦA TOP 10 CẶP CLASS NHẦM LẪN NHIỀU NHẤT (LOSO Seed 42):")
    print("-" * 80)
    loso_s42_path = os.path.join(EVAL_DIR, "loso_results_v3_seed42.json")
    top_pairs = []
    if os.path.exists(loso_s42_path):
        with open(loso_s42_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        top_pairs = data.get('top10_confusions', [])

    print(f"{'True Class':<15}{'Pred Class':<15}{'Nhầm':<8}{'True Motion (fr)':<18}{'Pred Motion (fr)':<18}{'True Speed':<12}")
    for p in top_pairs:
        tc = p['true_class']
        pc = p['predicted_class']
        cnt = p['count']
        tc_m = class_stats.get(tc, {}).get('median_m20_frames', 0.0)
        pc_m = class_stats.get(pc, {}).get('median_m20_frames', 0.0)
        tc_sp = class_stats.get(tc, {}).get('median_speed', 0.0)
        print(f"{tc:<15}{pc:<15}{cnt:<8}{tc_m:<18.1f}{pc_m:<18.1f}{tc_sp:<12.3f}")

    # 6. ƯỚC LƯỢNG ĐỘ TRỄ NẾU CẮT ĐOẠN HOẠT ĐỘNG
    print("\n" + "-" * 80)
    print("6. ƯỚC LƯỢNG ĐỘ TRỄ NẾU CẮT ĐOẠN HOẠT ĐỘNG (GESTURE SPAN ESTIMATION):")
    print("-" * 80)
    # Sàn hiện tại: Cửa sổ 3.2s + 3 lần confirm (mỗi lần ~120-150ms buffer tiến triển) -> ~4.0 - 4.5s
    # Độ dài hoạt động trung vị:
    med_active_frames = float(np.median(motion_20))
    med_active_sec = med_active_frames * DT_FRAME_SEC
    # Giả sử phát hiện bắt đầu/kết thúc cử chỉ (segmentation):
    # Thời gian từ khi bắt đầu ký đến khi kết thúc ký = med_active_sec (~1.8 - 2.1s)
    # Cộng thêm 3 lần confirm = 3 * 0.12s ~ 0.36s
    est_latency_segmented = med_active_sec + 0.36
    print("  - Sàn độ trễ hiện tại (cửa sổ tĩnh 3.2s + 3 confirm): ƯỚC LƯỢNG ~4.0 - 4.5 giây")
    print(f"  - Độ dài chuyển động trung vị (Median Motion Span): {med_active_frames:.1f} frames ({med_active_sec:.2f} giây)")
    print("  - ƯỚC LƯỢNG độ trễ nếu cắt đoạn hoạt động (Active Span + 3 confirms):")
    print(f"    t_est = {med_active_sec:.2f}s + 3 * 0.12s = {est_latency_segmented:.2f} giây")
    print(f"  - Tiềm năng giảm độ trễ: ~{4.2 - est_latency_segmented:.2f} giây (giảm ~{((4.2 - est_latency_segmented)/4.2)*100:.1f}%)")

    # 7. Vẽ biểu đồ phân bố và lưu PNG
    print("\n" + "-" * 80)
    print("7. VẼ BIỂU ĐỒ PHÂN BỐ:")
    print("-" * 80)
    # Biểu đồ 1: Histogram so sánh Hand Span vs Motion Span (20%)
    plt.figure(figsize=(10, 5))
    sns.histplot(hand_spans * DT_FRAME_SEC, color='skyblue', label='Hand Present Span (s)', kde=True, bins=25, alpha=0.5)
    sns.histplot(motion_20 * DT_FRAME_SEC, color='coral', label='Motion Span 20% (s)', kde=True, bins=25, alpha=0.6)
    plt.axvline(np.median(motion_20 * DT_FRAME_SEC), color='red', linestyle='--', label=f'Median Motion: {np.median(motion_20*DT_FRAME_SEC):.2f}s')
    plt.axvline(3.2, color='black', linestyle=':', label='Full Window: 3.20s')
    plt.title('Phân bố Độ dài Hoạt động Cử chỉ (Hand Span vs Motion Span)')
    plt.xlabel('Thời lượng hoạt động (giây - ƯỚC LƯỢNG)')
    plt.ylabel('Số lượng mẫu')
    plt.legend()
    plt.tight_layout()
    hist_out_path = os.path.join(EVAL_DIR, "gesture_span_distribution.png")
    plt.savefig(hist_out_path, dpi=150)
    plt.close()
    print(f"✓ Đã lưu biểu đồ phân bố: {hist_out_path}")

    # Biểu đồ 2: Boxplot Motion Span theo Session
    plt.figure(figsize=(9, 5))
    session_data = [np.array([r['motion_span_20'] * DT_FRAME_SEC for r in records if r['session'] == s]) for s in sessions]
    plt.boxplot(session_data, tick_labels=['Session 1', 'Session 2', 'Session 3', 'Session 4'], patch_artist=True,
                boxprops=dict(facecolor='lightblue', color='blue'),
                medianprops=dict(color='red', linewidth=2))
    plt.title('Độ dài Chuyển động Cử chỉ (Motion Span 20%) theo từng Session')
    plt.ylabel('Thời lượng (giây - ƯỚC LƯỢNG)')
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    box_out_path = os.path.join(EVAL_DIR, "gesture_span_by_session.png")
    plt.savefig(box_out_path, dpi=150)
    plt.close()
    print(f"✓ Đã lưu biểu đồ theo session: {box_out_path}")

    # Lưu kết quả json phân tích
    analysis_json_path = os.path.join(EVAL_DIR, "gesture_span_analysis.json")
    with open(analysis_json_path, 'w', encoding='utf-8') as f:
        json.dump({
            'overall': {
                'hand_span_median_sec': float(np.median(hand_spans * DT_FRAME_SEC)),
                'motion_span_10_median_sec': float(np.median(motion_10 * DT_FRAME_SEC)),
                'motion_span_20_median_sec': float(np.median(motion_20 * DT_FRAME_SEC)),
                'motion_span_30_median_sec': float(np.median(motion_30 * DT_FRAME_SEC)),
                'stationary_ratio_median': float(np.median(stat_ratios)),
            },
            'dynamic_classes': dynamic_classes,
            'static_classes': low_motion_classes,
            'class_stats': class_stats
        }, f, ensure_ascii=False, indent=2)
    print(f"✓ Đã lưu dữ liệu phân tích JSON: {analysis_json_path}")
    print("\n✅ HOÀN TẤT PHÂN TÍCH TÁC VỤ 1!")


if __name__ == '__main__':
    main()
