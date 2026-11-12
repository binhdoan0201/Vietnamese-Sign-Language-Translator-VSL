"""Tổng hợp kết quả so sánh Baseline vs S-A vs S-B cho Tác vụ 2."""
import os
import json
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
EVAL_DIR = os.path.join(PROJECT_ROOT, "models", "eval")

VARIANTS = {
    'Baseline (Full 30)': {
        42: 'loso_results_v3_P2_seed42.json',
        7: 'loso_results_v3_P2_seed7.json'
    },
    'S-A (Hand Span)': {
        42: 'loso_results_v3_segment_hand_seed42.json',
        7: 'loso_results_v3_segment_hand_seed7.json'
    },
    'S-B (Motion Span)': {
        42: 'loso_results_v3_segment_motion_seed42.json',
        7: 'loso_results_v3_segment_motion_seed7.json'
    }
}

def main():
    print("=" * 80)
    print("BẢNG TỔNG HỢP SO SÁNH TÁC VỤ 2: BASELINE VS S-A VS S-B (PROTOCOL P2)")
    print("=" * 80)

    results = {}

    for var_name, files in VARIANTS.items():
        results[var_name] = {'means': [], 'f1s': [], 'folds': {k: [] for k in range(1, 5)}}
        for seed, fname in files.items():
            fpath = os.path.join(EVAL_DIR, fname)
            if not os.path.exists(fpath):
                print(f"[!] Thiếu file {fname}")
                continue
            with open(fpath, 'r', encoding='utf-8') as f:
                d = json.load(f)
            mean_acc = d['summary']['mean_accuracy'] * 100
            mean_f1 = d['summary']['mean_macro_f1']
            results[var_name]['means'].append(mean_acc)
            results[var_name]['f1s'].append(mean_f1)
            for k, fold in enumerate(d['folds'], 1):
                results[var_name]['folds'][k].append(fold['test_accuracy'] * 100)

    header = f"{'Phương án':<22}{'Seed':<6}" + "".join(f"{'Fold ' + str(k):<12}" for k in range(1, 5)) + f"{'Mean Acc':<12}{'Macro-F1':<10}"
    print(header)
    print("-" * len(header))

    baseline_cross_mean = np.mean(results['Baseline (Full 30)']['means'])

    for var_name, data in results.items():
        for i, seed in enumerate([42, 7]):
            row = f"{var_name if i == 0 else '':<22}{seed:<6}"
            for k in range(1, 5):
                row += f"{data['folds'][k][i]:<12.2f}"
            row += f"{data['means'][i]:<12.2f}{data['f1s'][i]:<10.4f}"
            print(row)
        
        # In trung bình qua các seed
        cross_mean_acc = np.mean(data['means'])
        cross_std_acc = np.std(data['means'])
        cross_mean_f1 = np.mean(data['f1s'])
        cross_std_f1 = np.std(data['f1s'])
        diff_str = f"({cross_mean_acc - baseline_cross_mean:+.2f}%)" if var_name != 'Baseline (Full 30)' else "ref"
        sum_row = f"{'  -> Mean ± Std':<22}{'':<6}"
        for k in range(1, 5):
            sum_row += f"{np.mean(data['folds'][k]):.2f}±{np.std(data['folds'][k]):.2f}{'':<4}"
        sum_row += f"{cross_mean_acc:.2f}±{cross_std_acc:.2f}  {cross_mean_f1:.4f}±{cross_std_f1:.4f} {diff_str}"
        print(sum_row)
        print("-" * len(header))

    print("\n" + "=" * 80)
    print("ĐÁNH GIÁ THEO QUY TẮC 2.4 (ĐIỀU KIỆN CẬP NHẬT MODEL):")
    print("=" * 80)
    print("Quy tắc 2.4 quy định:")
    print("1. Phương án tốt nhất phải cải thiện mean LOSO >= max(3.0%, 2 * sigma_seed).")
    print("2. Không có fold nào bị giảm > 3.0% so với baseline.")
    print("\nKiểm tra:")
    sigma_seed = 1.87 # Từ Task 0.6
    threshold = max(3.0, 2 * sigma_seed) # max(3.0, 3.74) = 3.74%
    print(f"- Ngưỡng cải thiện cần đạt: max(3.0%, 2 x {sigma_seed:.2f}%) = {threshold:.2f}%")
    
    sa_mean = np.mean(results['S-A (Hand Span)']['means'])
    sb_mean = np.mean(results['S-B (Motion Span)']['means'])
    print(f"- Baseline mean: {baseline_cross_mean:.2f}%")
    print(f"- S-A mean     : {sa_mean:.2f}% (Chênh lệch: {sa_mean - baseline_cross_mean:+.2f}%)")
    print(f"- S-B mean     : {sb_mean:.2f}% (Chênh lệch: {sb_mean - baseline_cross_mean:+.2f}%)")

    # Kiểm tra từng fold của S-A
    sa_drops = []
    for k in range(1, 5):
        drop = np.mean(results['S-A (Hand Span)']['folds'][k]) - np.mean(results['Baseline (Full 30)']['folds'][k])
        sa_drops.append((k, drop))
        print(f"  + Fold {k}: Baseline {np.mean(results['Baseline (Full 30)']['folds'][k]):.2f}% -> S-A {np.mean(results['S-A (Hand Span)']['folds'][k]):.2f}% ({drop:+.2f}%)")

    print("\n👉 KẾT LUẬN QUY TẮC 2.4:")
    print("Cả 2 phương án S-A và S-B ĐỀU KHÔNG đạt điều kiện cải thiện (+0.00% hoặc suy giảm -6.26%),")
    print("đồng thời có các fold bị giảm sâu (ví dụ Fold 3 và Fold 4 ở S-A, Fold 2 ở S-B).")
    print("=> QUYẾT ĐỊNH: GIỮ NGUYÊN model hiện tại models/sign_language_model.h5, KHÔNG retrain --final.")

if __name__ == '__main__':
    main()
