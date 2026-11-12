"""
tools/evaluate_augment_seeds.py
===============================
Tổng hợp và đánh giá kết quả huấn luyện LOSO theo Protocol P2 giữa các cấu hình Augmentation (Task B3 & B4).

Các cấu hình:
  (i) none: Baseline P2 (seed 42, 7, 123) từ models/eval/loso_results_v3_P2_seed*.json
  (ii) rot_y: Rotate Yaw +-20 độ
  (iii) rot_yz: Rotate Yaw +-20 độ + Roll +-10 độ
  (iv) full: Rotate Yaw + Scale/Translate + Noise + Time-warp

Tính toán:
  - Accuracy & Macro-F1 từng fold (S1..S4), từng seed
  - Trung bình và độ lệch chuẩn mẫu (ddof=1) giữa các seed
  - Accuracy riêng cho S2 và S4 (2 session góc lệch/khó)
  - So sánh với Baseline (i) và kiểm tra điều kiện Rule B4
"""

import os
import sys
import json
import glob
import numpy as np

# Ép stdout UTF-8
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
EVAL_DIR = os.path.join(PROJECT_ROOT, "models", "eval")


def load_config_results(file_pattern):
    """
    Đọc các file json khớp pattern, trả về dict {seed: data}
    """
    files = sorted(glob.glob(os.path.join(EVAL_DIR, file_pattern)))
    results = {}
    for f in files:
        fname = os.path.basename(f)
        try:
            with open(f, "r", encoding="utf-8") as fp:
                data = json.load(fp)
                seed = data.get("seed", None)
                if seed is None:
                    # Parse từ tên file
                    import re
                    m = re.search(r"seed(\d+)", fname)
                    if m:
                        seed = int(m.group(1))
                    else:
                        seed = fname
                results[seed] = data
        except Exception as e:
            print(f"[!] Lỗi đọc file {f}: {e}")
    return results


def summarize_config(results_dict):
    """
    Trích xuất ma trận kết quả:
    folds: [S1, S2, S3, S4] accuracy & f1
    mean_acc, mean_f1
    s2_s4_acc
    """
    seeds = sorted(results_dict.keys(), key=lambda x: int(x) if isinstance(x, int) or str(x).isdigit() else str(x))
    if not seeds:
        return None

    fold_accs = {1: [], 2: [], 3: [], 4: []}
    fold_f1s = {1: [], 2: [], 3: [], 4: []}
    overall_accs = []
    overall_f1s = []
    s2_s4_accs = []

    per_seed_summary = {}

    for s in seeds:
        d = results_dict[s]
        s_fold_accs = []
        s_fold_f1s = []
        for fold in d["folds"]:
            k = fold["fold"]
            acc = fold["test_accuracy"] * 100.0
            f1 = fold["macro_f1"]
            fold_accs[k].append(acc)
            fold_f1s[k].append(f1)
            s_fold_accs.append(acc)
            s_fold_f1s.append(f1)

        m_acc = float(np.mean(s_fold_accs))
        m_f1 = float(np.mean(s_fold_f1s))
        s24 = float((s_fold_accs[1] + s_fold_accs[3]) / 2.0)  # S2 is fold 2 (idx 1), S4 is fold 4 (idx 3)

        overall_accs.append(m_acc)
        overall_f1s.append(m_f1)
        s2_s4_accs.append(s24)

        per_seed_summary[s] = {
            'fold_accs': s_fold_accs,
            'fold_f1s': s_fold_f1s,
            'mean_acc': m_acc,
            'mean_f1': m_f1,
            's2_s4_acc': s24
        }

    # Tính mean và sample std (ddof=1 nếu n >= 2, else 0.0)
    n = len(seeds)
    ddof = 1 if n >= 2 else 0

    agg = {
        'n_seeds': n,
        'seeds': seeds,
        'per_seed': per_seed_summary,
        'fold_acc_mean': [float(np.mean(fold_accs[k])) for k in range(1, 5)],
        'fold_acc_std': [float(np.std(fold_accs[k], ddof=ddof)) if n >= 2 else 0.0 for k in range(1, 5)],
        'fold_f1_mean': [float(np.mean(fold_f1s[k])) for k in range(1, 5)],
        'fold_f1_std': [float(np.std(fold_f1s[k], ddof=ddof)) if n >= 2 else 0.0 for k in range(1, 5)],
        'mean_acc': float(np.mean(overall_accs)),
        'std_acc': float(np.std(overall_accs, ddof=ddof)) if n >= 2 else 0.0,
        'mean_f1': float(np.mean(overall_f1s)),
        'std_f1': float(np.std(overall_f1s, ddof=ddof)) if n >= 2 else 0.0,
        's2_s4_mean': float(np.mean(s2_s4_accs)),
        's2_s4_std': float(np.std(s2_s4_accs, ddof=ddof)) if n >= 2 else 0.0,
    }
    return agg


def main():
    print("=" * 115)
    print("BẢNG ĐÁNH GIÁ TỔNG HỢP CÁC CẤU HÌNH AUGMENTATION DƯỚI PROTOCOL P2 (LOSO, FIXED 40 EPOCHS)")
    print("=" * 115)

    configs = [
        ("(i) None (Baseline P2)", "loso_results_v3_P2_seed*.json"),
        ("(ii) rot_y", "loso_results_v4_rot_y_seed*.json"),
        ("(iii) rot_yz", "loso_results_v4_rot_yz_seed*.json"),
        ("(iv) full", "loso_results_v4_full_seed*.json"),
    ]

    summaries = {}
    for name, pattern in configs:
        res = load_config_results(pattern)
        summ = summarize_config(res)
        summaries[name] = summ
        if summ is None:
            print(f"[-] {name}: Chưa có kết quả JSON (pattern: {pattern})")
        else:
            print(f"[+] {name}: Đã nạp kết quả từ {summ['n_seeds']} seed(s): {summ['seeds']}")

    # Lấy baseline (i)
    baseline_summ = summaries.get("(i) None (Baseline P2)")
    baseline_std_acc = baseline_summ['std_acc'] if baseline_summ else 2.29

    threshold_gain = max(3.0, 2.0 * baseline_std_acc)
    print("\n" + "=" * 115)
    print(f"QUY TẮC B4: Ngưỡng tăng độ chính xác tối thiểu: MAX(3.0%, 2 x sigma_seed) = MAX(3.0%, 2 x {baseline_std_acc:.2f}%) = {threshold_gain:.2f}%")
    print("            Điều kiện phụ: Không fold nào (trung bình seed) tụt quá 3.0% so với Baseline.")
    print("=" * 115)

    # In bảng chi tiết từng Seed
    print("\n1. CHI TIẾT KẾT QUẢ TỪNG SEED & TỪNG FOLD:")
    header_seeds = f"{'Cấu hình':<24}{'Seed':<8}{'Fold 1 (S1)':<14}{'Fold 2 (S2)':<14}{'Fold 3 (S3)':<14}{'Fold 4 (S4)':<14}{'Mean Acc':<12}{'Macro-F1':<12}{'S2+S4 Acc':<12}"
    print(header_seeds)
    print("-" * len(header_seeds))

    for name, summ in summaries.items():
        if summ is None:
            continue
        for s in summ['seeds']:
            p = summ['per_seed'][s]
            f_str = "".join(f"{acc:>6.2f}%{'':<7}" for acc in p['fold_accs'])
            print(f"{name:<24}{s:<8}{f_str}{p['mean_acc']:>6.2f}%{'':<5}{p['mean_f1']:>6.4f}{'':<6}{p['s2_s4_acc']:>6.2f}%")
        print("-" * len(header_seeds))

    # In bảng tổng hợp so sánh Mean +- Std giữa các seed
    print("\n2. BẢNG TỔNG HỢP MEAN ± STD (ddof=1) QUA CÁC SEED:")
    header_agg = f"{'Cấu hình':<24}{'N_Seed':<8}{'Fold 1 (S1)':<16}{'Fold 2 (S2)':<16}{'Fold 3 (S3)':<16}{'Fold 4 (S4)':<16}{'Mean Acc':<18}{'Macro-F1':<16}{'S2+S4 Acc':<16}"
    print(header_agg)
    print("-" * len(header_agg))

    for name, summ in summaries.items():
        if summ is None:
            continue
        folds_str = "".join(f"{summ['fold_acc_mean'][k]:>5.2f}±{summ['fold_acc_std'][k]:<5.2f}% " for k in range(4))
        mean_acc_str = f"{summ['mean_acc']:>5.2f}±{summ['std_acc']:<5.2f}%"
        mean_f1_str = f"{summ['mean_f1']:>6.4f}±{summ['std_f1']:<6.4f}"
        s24_str = f"{summ['s2_s4_mean']:>5.2f}±{summ['s2_s4_std']:<5.2f}%"
        print(f"{name:<24}{summ['n_seeds']:<8}{folds_str} {mean_acc_str:<18}{mean_f1_str:<16}{s24_str:<16}")
    print("=" * len(header_agg))

    # Đánh giá theo Rule B4
    if baseline_summ:
        print("\n3. ĐÁNH GIÁ CHÊNH LỆCH SO VỚI BASELINE (i) & KIỂM TRA ĐIỀU KIỆN RULE B4:")
        header_eval = f"{'Cấu hình':<24}{'Δ Mean Acc':<14}{'Δ Fold 1':<12}{'Δ Fold 2':<12}{'Δ Fold 3':<12}{'Δ Fold 4':<12}{'Δ S2+S4':<12}{'Đạt B4?':<12}"
        print(header_eval)
        print("-" * len(header_eval))

        for name, summ in summaries.items():
            if summ is None or name == "(i) None (Baseline P2)":
                continue
            delta_mean = summ['mean_acc'] - baseline_summ['mean_acc']
            delta_folds = [summ['fold_acc_mean'][k] - baseline_summ['fold_acc_mean'][k] for k in range(4)]
            delta_s24 = summ['s2_s4_mean'] - baseline_summ['s2_s4_mean']

            cond_gain = delta_mean >= threshold_gain
            cond_no_drop = all(df >= -3.0 for df in delta_folds)
            pass_b4 = cond_gain and cond_no_drop

            status_str = "ĐẠT (CHỌN)" if pass_b4 else "KHÔNG ĐẠT"

            df_str = "".join(f"{df:>+6.2f}%{'':<5}" for df in delta_folds)
            print(f"{name:<24}{delta_mean:>+6.2f}%{'':<7}{df_str}{delta_s24:>+6.2f}%{'':<5}{status_str:<12}")
        print("=" * len(header_eval))


if __name__ == '__main__':
    main()
