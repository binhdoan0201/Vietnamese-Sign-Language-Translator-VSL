"""
tools/eval_web_events.py
========================
Công cụ đánh giá sự kiện nhận dạng web và thống kê cổng chuyển động (Task A4 & D3).

Hỗ trợ 2 cách gọi:
  Cách 1 (cờ lệnh):   python tools/eval_web_events.py --events logs/web_events.csv --ground-truth expected_words.txt [--blocked logs/web_gate_blocked.csv]
  Cách 2 (vị trí):    python tools/eval_web_events.py logs/web_events.csv expected_words.txt
"""

import os
import sys
import csv
import argparse
import numpy as np

# Ép stdout UTF-8 để hiển thị tiếng Việt chuẩn trên PowerShell
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_EVENTS_CSV = os.path.join(PROJECT_ROOT, "logs", "web_events.csv")
DEFAULT_EXPECTED_TXT = os.path.join(PROJECT_ROOT, "expected_words.txt")
DEFAULT_BLOCKED_CSV = os.path.join(PROJECT_ROOT, "logs", "web_gate_blocked.csv")


def align_sequences(expected_list, recognized_list):
    """
    Căn chỉnh 2 chuỗi từ vựng theo thuật toán Needleman-Wunsch / Levenshtein.

    Chi phí:
      - Match: 0
      - Substitution: 1
      - Insertion: 1
      - Deletion: 1
    """
    M = len(expected_list)
    N = len(recognized_list)

    dp = np.zeros((M + 1, N + 1), dtype=int)
    for i in range(M + 1):
        dp[i, 0] = i
    for j in range(N + 1):
        dp[0, j] = j

    for i in range(1, M + 1):
        for j in range(1, N + 1):
            cost = 0 if expected_list[i - 1] == recognized_list[j - 1] else 1
            dp[i, j] = min(
                dp[i - 1, j] + 1,        # Deletion
                dp[i, j - 1] + 1,        # Insertion
                dp[i - 1, j - 1] + cost  # Match / Substitution
            )

    i, j = M, N
    aligned = []
    while i > 0 or j > 0:
        if i > 0 and j > 0 and (dp[i, j] == dp[i - 1, j - 1] + (0 if expected_list[i - 1] == recognized_list[j - 1] else 1)):
            if expected_list[i - 1] == recognized_list[j - 1]:
                aligned.append((expected_list[i - 1], recognized_list[j - 1], 'CORRECT', j - 1))
            else:
                aligned.append((expected_list[i - 1], recognized_list[j - 1], 'SUBSTITUTION', j - 1))
            i -= 1
            j -= 1
        elif i > 0 and dp[i, j] == dp[i - 1, j] + 1:
            aligned.append((expected_list[i - 1], None, 'DELETION', None))
            i -= 1
        else:
            aligned.append((None, recognized_list[j - 1], 'INSERTION', j - 1))
            j -= 1

    aligned.reverse()
    return aligned


def evaluate_blocked_events(blocked_csv_path):
    """Thống kê chi tiết các sự kiện bị Motion Gate chặn (Task D3)."""
    if not blocked_csv_path or not os.path.exists(blocked_csv_path):
        return None

    by_state = {}
    by_action = {}
    total_blocked = 0

    try:
        with open(blocked_csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                total_blocked += 1
                state = row.get('gate_state', 'unknown')
                action = row.get('would_be_action', 'unknown')
                by_state[state] = by_state.get(state, 0) + 1
                by_action[action] = by_action.get(action, 0) + 1
    except Exception as e:
        print(f"[WARN] Không thể đọc file blocked log: {e}")
        return None

    print("\n" + "=" * 70)
    print("THỐNG KÊ SỰ KIỆN BỊ CỔNG CHUYỂN ĐỘNG CHẶN (WEB GATE BLOCKED)")
    print("=" * 70)
    print(f"Tổng số sự kiện bị chặn : {total_blocked}")
    print("\n1. Phân loại theo Gate State:")
    for st, count in sorted(by_state.items(), key=lambda x: x[1], reverse=True):
        print(f"  - {st:<22}: {count:>3} ({count / max(1, total_blocked) * 100:5.1f}%)")

    print("\n2. Top các nhãn bị chặn nhiều nhất (would_be_action):")
    sorted_act = sorted(by_action.items(), key=lambda x: x[1], reverse=True)[:10]
    for act, count in sorted_act:
        print(f"  - '{act}': {count} lần")
    print("=" * 70)

    return {
        'total_blocked': total_blocked,
        'by_state': by_state,
        'by_action': by_action
    }


def evaluate_events(events_csv_path, expected_txt_path, blocked_csv_path=None):
    """Đánh giá chi tiết file web_events so với expected_words."""
    # 1. Kiểm tra sự tồn tại của file Ground Truth
    if not os.path.exists(expected_txt_path):
        print(f"[ERROR] Không tìm thấy file Ground Truth: '{expected_txt_path}'", file=sys.stderr)
        sys.exit(1)

    try:
        with open(expected_txt_path, 'r', encoding='utf-8') as f:
            expected = [line.strip() for line in f if line.strip()]
    except Exception as e:
        print(f"[ERROR] Không thể đọc file Ground Truth '{expected_txt_path}': {e}", file=sys.stderr)
        sys.exit(1)

    # 2. Kiểm tra sự tồn tại của file Events CSV
    if not os.path.exists(events_csv_path):
        print(f"[ERROR] Không tìm thấy file sự kiện: '{events_csv_path}'", file=sys.stderr)
        sys.exit(1)

    recognized_events = []
    try:
        with open(events_csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if 'action' in row:
                    recognized_events.append(row)
    except Exception as e:
        print(f"[ERROR] Không thể đọc file sự kiện '{events_csv_path}': {e}", file=sys.stderr)
        sys.exit(1)

    rec_actions = [r['action'] for r in recognized_events]

    print(f"[INFO] Đã nạp {len(recognized_events)} dòng sự kiện từ: {events_csv_path}")
    print(f"[INFO] Đã nạp {len(expected)} từ kỳ vọng từ: {expected_txt_path}")

    aligned = align_sequences(expected, rec_actions)

    n_correct = sum(1 for _, _, st, _ in aligned if st == 'CORRECT')
    n_sub = sum(1 for _, _, st, _ in aligned if st == 'SUBSTITUTION')
    n_del = sum(1 for _, _, st, _ in aligned if st == 'DELETION')
    n_ins = sum(1 for _, _, st, _ in aligned if st == 'INSERTION')

    latencies = []
    spurious_counts = {}

    for exp_w, rec_w, st, rec_idx in aligned:
        if st in ['CORRECT', 'SUBSTITUTION'] and rec_idx is not None:
            t_str = recognized_events[rec_idx].get('t_first_hand_to_output_s')
            try:
                t_val = float(t_str)
                if t_val > 0:
                    latencies.append(t_val)
            except (ValueError, TypeError):
                pass
        elif st == 'INSERTION':
            spurious_counts[rec_w] = spurious_counts.get(rec_w, 0) + 1

    med_lat = float(np.median(latencies)) if latencies else 0.0
    p90_lat = float(np.percentile(latencies, 90)) if latencies else 0.0

    print("\n" + "=" * 70)
    print("KẾT QUẢ ĐÁNH GIÁ SỰ KIỆN NHẬN DẠNG TRÊN GIAO DIỆN WEB")
    print("=" * 70)
    print(f"Tổng số từ kỳ vọng (Ground Truth)  : {len(expected)}")
    print(f"Tổng số sự kiện phát ra (Detected) : {len(rec_actions)}")
    print("-" * 70)
    print(f"  - Đúng (Correct)         : {n_correct:>3} ({n_correct / max(1, len(expected)) * 100:5.1f}%)")
    print(f"  - Sai (Substitution)    : {n_sub:>3} ({n_sub / max(1, len(expected)) * 100:5.1f}%)")
    print(f"  - Bỏ sót (Deletion)     : {n_del:>3} ({n_del / max(1, len(expected)) * 100:5.1f}%)")
    print(f"  - Dư thừa (Insertion)   : {n_ins:>3} ({n_ins / max(1, len(expected)) * 100:5.1f}%)")
    print("-" * 70)
    print("Độ trễ phản hồi (t_first_hand_to_output_s):")
    print(f"  - Trung vị (Median Latency): {med_lat:.2f} s")
    print(f"  - Phân vị P90 Latency      : {p90_lat:.2f} s")
    print("-" * 70)
    print("Các nhãn dư thừa (Spurious/Ghost Detections) hay gặp:")
    sorted_spurious = sorted(spurious_counts.items(), key=lambda x: x[1], reverse=True)
    if sorted_spurious:
        for w, c in sorted_spurious:
            print(f"  - '{w}': {c} lần")
    else:
        print("  (Không có nhãn dư thừa nào)")
    print("=" * 70)

    # Thống kê bổ sung file blocked log nếu được chỉ định
    blocked_stats = None
    if blocked_csv_path:
        blocked_stats = evaluate_blocked_events(blocked_csv_path)

    return {
        'expected_count': len(expected),
        'detected_count': len(rec_actions),
        'correct': n_correct,
        'substitution': n_sub,
        'deletion': n_del,
        'insertion': n_ins,
        'median_latency_s': med_lat,
        'p90_latency_s': p90_lat,
        'spurious_counts': spurious_counts,
        'aligned_pairs': aligned,
        'blocked_stats': blocked_stats
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Đánh giá chuỗi sự kiện web và thống kê Motion Gate."
    )
    # Hỗ trợ cờ tùy chọn
    parser.add_argument("--events", dest="events_flag", default=None,
                        help="Đường dẫn tới file web_events.csv")
    parser.add_argument("--ground-truth", dest="gt_flag", default=None,
                        help="Đường dẫn tới file expected_words.txt")
    parser.add_argument("--blocked", dest="blocked", default=None,
                        help="Đường dẫn tới file web_gate_blocked.csv (Task D3)")

    # Hỗ trợ đối số vị trí (positional) để giữ tương thích ngược
    parser.add_argument("positional_args", nargs="*",
                        help="[events_path] [ground_truth_path]")

    args = parser.parse_args()

    # Phân giải thứ tự ưu tiên: cờ tùy chọn > đối số vị trí > giá trị mặc định
    events_path = args.events_flag
    gt_path = args.gt_flag

    pos = args.positional_args
    if events_path is None:
        if len(pos) >= 1:
            events_path = pos[0]
        else:
            events_path = DEFAULT_EVENTS_CSV

    if gt_path is None:
        if len(pos) >= 2:
            gt_path = pos[1]
        else:
            gt_path = DEFAULT_EXPECTED_TXT

    return events_path, gt_path, args.blocked


def main():
    events_path, gt_path, blocked_path = parse_args()
    evaluate_events(events_path, gt_path, blocked_path)


if __name__ == '__main__':
    main()
