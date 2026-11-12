"""Công cụ tổng hợp và đánh giá kết quả LOSO đa seed (Task 0.5 & 0.6)."""
import os
import json
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
EVAL_DIR = os.path.join(PROJECT_ROOT, "models", "eval")

TARGET_CLASSES = ['xin_chao', 'xin_loi', 'dau_hoi']


def load_loso_json(filename):
    path = os.path.join(EVAL_DIR, filename)
    if not os.path.exists(path):
        return None
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def analyze_protocol(proto_name, seed_list):
    print("\n================================================================================")
    print(f"TỔNG HỢP PROTOCOL: {proto_name} (SEEDS: {seed_list})")
    print("================================================================================")
    
    data_by_seed = {}
    for seed in seed_list:
        fname = f"loso_results_v3_{proto_name}_seed{seed}.json"
        d = load_loso_json(fname)
        if d is None:
            print(f"  [!] Chưa có file: {fname}")
            continue
        data_by_seed[seed] = d

    if not data_by_seed:
        print("  Không có dữ liệu.")
        return None

    # Bảng 1: Accuracy & F1 từng fold x seed
    print("\n--- 1. BẢNG ACCURACY TỪNG FOLD X SEED ---")
    header = f"{'Seed':<8}" + "".join(f"{'Fold ' + str(k) + ' (S' + str(k) + ')':<16}" for k in range(1, 5)) + f"{'Mean Acc':<12}{'Macro-F1':<10}"
    print(header)
    print("-" * len(header))

    fold_accs = {k: [] for k in range(1, 5)}
    overall_means = []
    overall_f1s = []

    for seed, d in data_by_seed.items():
        folds = d['folds']
        accs = [f['test_accuracy'] * 100 for f in folds]
        for k in range(1, 5):
            fold_accs[k].append(accs[k - 1])
        mean_acc = d['summary']['mean_accuracy'] * 100
        mean_f1 = d['summary']['mean_macro_f1']
        overall_means.append(mean_acc)
        overall_f1s.append(mean_f1)
        row = f"{seed:<8}" + "".join(f"{accs[k-1]:<16.2f}" for k in range(1, 5)) + f"{mean_acc:<12.2f}{mean_f1:<10.4f}"
        print(row)

    print("-" * len(header))
    # Mean ± Std giữa các seed (ddof=1: độ lệch chuẩn mẫu theo Quy định 6)
    summary_row = f"{'Mean±Std':<8}" + "".join(f"{np.mean(fold_accs[k]):.2f}±{np.std(fold_accs[k], ddof=1):.2f}{'':<6}" for k in range(1, 5)) + f"{np.mean(overall_means):.2f}±{np.std(overall_means, ddof=1):.2f}{'':<3}{np.mean(overall_f1s):.4f}±{np.std(overall_f1s, ddof=1):.4f}"
    print(summary_row)

    # Bảng 2: Best Epoch (đặc biệt quan trọng với P1)
    print("\n--- 2. BEST EPOCH TỪNG FOLD / SEED ---")
    ep_header = f"{'Seed':<8}" + "".join(f"{'Fold ' + str(k):<12}" for k in range(1, 5)) + f"{'Median':<10}"
    print(ep_header)
    print("-" * len(ep_header))
    for seed, d in data_by_seed.items():
        folds = d['folds']
        eps = [f['best_epoch'] for f in folds]
        med_ep = int(np.median(eps))
        print(f"{seed:<8}" + "".join(f"{eps[k-1]:<12}" for k in range(1, 5)) + f"{med_ep:<10}")

    # Bảng 3: Recall của target classes
    print("\n--- 3. RECALL CỦA XIN_CHAO, XIN_LOI, DAU_HOI ---")
    print(f"{'Seed':<8}{'Class':<12}" + "".join(f"{'Fold ' + str(k):<12}" for k in range(1, 5)) + f"{'Mean Recall':<12}")
    print("-" * 65)
    for seed, d in data_by_seed.items():
        folds = d['folds']
        for cls in TARGET_CLASSES:
            recs = []
            for f in folds:
                cr = f.get('classification_report', {})
                r_val = cr.get(cls, {}).get('recall', 0.0) * 100
                recs.append(r_val)
            mean_r = np.mean(recs)
            print(f"{seed:<8}{cls:<12}" + "".join(f"{recs[k-1]:<12.1f}" for k in range(1, 5)) + f"{mean_r:<12.1f}")
        print("-" * 65)

    return {
        'protocol': proto_name,
        'cross_seed_mean_acc': float(np.mean(overall_means)),
        'cross_seed_std_acc': float(np.std(overall_means, ddof=1)),
        'cross_seed_mean_f1': float(np.mean(overall_f1s)),
        'cross_seed_std_f1': float(np.std(overall_f1s, ddof=1)),
        'fold_accs': fold_accs
    }


def main():
    p1_stats = analyze_protocol("P1", [42, 7, 123])
    p2_stats = analyze_protocol("P2", [42, 7, 123])

    if p1_stats and p2_stats:
        print("\n" + "=" * 80)
        print("SO SÁNH PROTOCOL P1 VS PROTOCOL P2:")
        print("=" * 80)
        print(f"P1 (EarlyStopping theo Val): Mean Acc = {p1_stats['cross_seed_mean_acc']:.2f}% ± {p1_stats['cross_seed_std_acc']:.2f}% (Std giữa các seed: {p1_stats['cross_seed_std_acc']:.2f}%)")
        print(f"P2 (Cố định 40 Epochs)    : Mean Acc = {p2_stats['cross_seed_mean_acc']:.2f}% ± {p2_stats['cross_seed_std_acc']:.2f}% (Std giữa các seed: {p2_stats['cross_seed_std_acc']:.2f}%)")
        
        if p2_stats['cross_seed_std_acc'] <= p1_stats['cross_seed_std_acc']:
            print("👉 KHUYẾN NGHỊ: Chọn PROTOCOL P2 làm mốc chính (độ lệch giữa các seed nhỏ hơn hoặc tương đương, không bị nhiễu do val session lệch phân phối).")
        else:
            print("👉 KHUYẾN NGHỊ: Xem xét độ ổn định và tính khoa học giữa P1 và P2.")


if __name__ == '__main__':
    main()
