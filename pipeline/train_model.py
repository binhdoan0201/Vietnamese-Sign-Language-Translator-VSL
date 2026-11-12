"""
pipeline/train_model.py
=======================
Train mô hình LSTM trên đặc trưng (N, 30, 384) và xuất đánh giá.
Hỗ trợ:
  1. Đánh giá chuẩn Leave-One-Session-Out (--eval loso): 4 fold xoay vòng
     (Test = 1 session; Val = session kế tiếp; Train = 2 session còn lại).
  2. Đánh giá Baseline ngẫu nhiên (--eval baseline).
  3. Huấn luyện Model cuối (--final) trên toàn bộ 4 session với số epoch = median best_epoch
     của các fold LOSO, không dùng val, lưu đè vào models/sign_language_model.h5.
  4. Mặc định (không cờ): Train trên X_train/X_test đã chuẩn bị trong dataset/processed/.

CÁCH DÙNG:
  # Chạy 4-fold LOSO cross-validation:
  python pipeline/train_model.py --eval loso

  # Chạy baseline ngẫu nhiên:
  python pipeline/train_model.py --eval baseline

  # Huấn luyện mô hình production cuối cùng:
  python pipeline/train_model.py --final

  # Chế độ mặc định (tương thích cũ):
  python pipeline/train_model.py
"""

import os
import sys
import json
import random
import argparse
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import matplotlib
matplotlib.use('Agg')  # Chạy không cần GUI window để tránh block tiến trình
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score

# Đảm bảo import được feature_utils từ thư mục gốc
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from feature_utils import get_actions, build_features
from pipeline.augment_utils import generate_augmented_dataset
from pipeline.segment_utils import segment_dataset

PROCESSED_PATH = os.path.join(PROJECT_ROOT, "dataset", "processed")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
EVAL_DIR = os.path.join(MODEL_DIR, "eval")
MODEL_SAVE_PATH = os.path.join(MODEL_DIR, "sign_language_model.h5")
LOSO_RESULTS_PATH = os.path.join(EVAL_DIR, "loso_results.json")
BASELINE_RESULTS_PATH = os.path.join(EVAL_DIR, "baseline_results.json")


def set_seeds(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    print(f"[SEED] Cố định random seed: {seed}")


def build_lstm_model(input_shape, num_classes):
    """
    Xây dựng kiến trúc LSTM chuẩn theo đồ án (giữ NGUYÊN cấu hình):
      Input -> LSTM(64) -> Dropout(0.2) -> LSTM(128) -> Dropout(0.2) -> Dense(64) -> Softmax
    """
    model = Sequential([
        Input(shape=input_shape),
        LSTM(64, return_sequences=True),
        Dropout(0.2),
        LSTM(128, return_sequences=False),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    optimizer = Adam(learning_rate=0.0005)
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['categorical_accuracy']
    )
    return model


def plot_and_save_confusion_matrix(y_true, y_pred, actions, save_path, title="Confusion Matrix"):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(actions))))
    plt.figure(figsize=(14, 11))
    sns.heatmap(cm, annot=False, cmap='Blues',
                xticklabels=actions, yticklabels=actions)
    plt.xlabel('Dự đoán (Predicted)')
    plt.ylabel('Thực tế (Ground Truth)')
    plt.title(title)
    plt.xticks(rotation=90, fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    return cm


def plot_and_save_history(history, save_path, title_prefix="Training"):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].plot(history.history['categorical_accuracy'], label='Train')
    if 'val_categorical_accuracy' in history.history:
        axes[0].plot(history.history['val_categorical_accuracy'], label='Validation')
    axes[0].set_title(f'{title_prefix} Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(history.history['loss'], label='Train')
    if 'val_loss' in history.history:
        axes[1].plot(history.history['val_loss'], label='Validation')
    axes[1].set_title(f'{title_prefix} Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()


def run_loso_evaluation(args, actions):
    print("\n" + "=" * 80)
    print("BẮT ĐẦU ĐÁNH GIÁ LEAVE-ONE-SESSION-OUT (LOSO) 4 FOLDS")
    print("=" * 80)

    X_all_path = os.path.join(PROCESSED_PATH, 'X_all.npy')
    X_raw258_path = os.path.join(PROCESSED_PATH, 'X_raw258_all.npy')
    y_all_path = os.path.join(PROCESSED_PATH, 'y_all.npy')
    session_all_path = os.path.join(PROCESSED_PATH, 'session_all.npy')

    if not (os.path.exists(X_all_path) and os.path.exists(y_all_path) and os.path.exists(session_all_path)):
        print("[!] Thiếu dữ liệu X_all/y_all/session_all. Hãy chạy: python pipeline/preprocess_data.py trước.")
        sys.exit(1)

    X_all = np.load(X_all_path)
    X_raw258_all = np.load(X_raw258_path) if os.path.exists(X_raw258_path) else None
    y_all = np.load(y_all_path)
    session_all = np.load(session_all_path, allow_pickle=True)

    # Tiền xử lý Segment nếu được yêu cầu
    if args.segment != "none":
        if X_raw258_all is None:
            print("[!] Thiếu X_raw258_all.npy để chạy segment! Chạy preprocess_data.py trước.")
            sys.exit(1)
        print(f"👉 Kích hoạt Gesture Segmentation: '{args.segment}' trên toàn bộ dữ liệu thô 258 chiều...")
        X_seg_raw, seg_stats = segment_dataset(X_raw258_all, mode=args.segment)
        print(f"    -> Fallback giữ full cửa sổ: {seg_stats['fallback_count']}/{seg_stats['total']} ({seg_stats['fallback_ratio']*100:.2f}%)")
        print(f"    -> Đang trích xuất build_features cho toàn bộ {len(X_seg_raw)} mẫu...")
        X_all_segmented = np.array([build_features(s) for s in X_seg_raw], dtype=np.float32)
    else:
        X_all_segmented = X_all

    if args.augment != "none":
        print(f"👉 Kích hoạt Data Augmentation cho tập Train: '{args.augment}' (K={args.aug_factor} biến thể/sample)")

    num_classes = len(actions)
    input_shape = (X_all.shape[1], X_all.shape[2])

    fold_configs = [
        {'fold': 1, 'test': 'session_01', 'val': 'session_02', 'train': ['session_03', 'session_04']},
        {'fold': 2, 'test': 'session_02', 'val': 'session_03', 'train': ['session_04', 'session_01']},
        {'fold': 3, 'test': 'session_03', 'val': 'session_04', 'train': ['session_01', 'session_02']},
        {'fold': 4, 'test': 'session_04', 'val': 'session_01', 'train': ['session_02', 'session_03']},
    ]

    fold_results = []
    overall_cm = np.zeros((num_classes, num_classes), dtype=int)
    all_y_true = []
    all_y_pred = []

    # Xác định tag lưu file
    aug_tag = f"_{args.augment}" if args.augment != "none" else ""
    seg_tag = f"_{args.segment}" if args.segment != "none" else ""
    proto_tag = f"_P2_fixed{args.fixed_epochs}" if args.fixed_epochs > 0 else "_P1"
    seed_tag = f"_seed{args.seed}"
    default_tag = f"{seg_tag}{aug_tag}{proto_tag}{seed_tag}"

    if args.out_json:
        result_json_path = os.path.join(EVAL_DIR, args.out_json)
        tag_for_plots = "_" + os.path.splitext(args.out_json)[0].replace("loso_results_", "")
    else:
        result_json_path = os.path.join(EVAL_DIR, f"loso_results{default_tag}.json")
        tag_for_plots = default_tag

    for cfg in fold_configs:
        k = cfg['fold']
        test_sess = cfg['test']
        val_sess = cfg['val']
        train_sess = cfg['train']

        print(f"\n>>> [FOLD {k}/4] TEST: {test_sess} | VAL: {val_sess} | TRAIN: {train_sess}")

        mask_test = (session_all == test_sess)
        mask_val = (session_all == val_sess)
        mask_train = np.isin(session_all, train_sess)

        if args.augment != "none":
            if X_raw258_all is None:
                print("[!] Thiếu X_raw258_all.npy để chạy augment! Chạy preprocess_data.py trước.")
                sys.exit(1)
            raw_source = X_seg_raw if args.segment != "none" else X_raw258_all
            X_raw_train = raw_source[mask_train]
            y_train_orig = y_all[mask_train]
            X_raw_aug, y_train = generate_augmented_dataset(
                X_raw_train, y_train_orig,
                augment_mode=args.augment,
                aug_factor=args.aug_factor,
                target_duration=3.2,
                seed=args.seed + k
            )
            print(f"    -> Đang trích xuất build_features cho {len(X_raw_aug)} mẫu train...")
            X_train = np.array([build_features(s) for s in X_raw_aug], dtype=np.float32)
        else:
            X_train, y_train = X_all_segmented[mask_train], y_all[mask_train]

        X_val, y_val = X_all_segmented[mask_val], y_all[mask_val]
        X_test, y_test = X_all_segmented[mask_test], y_all[mask_test]

        print(f"    Train: {X_train.shape[0]} | Val: {X_val.shape[0]} | Test: {X_test.shape[0]}")

        # Đặt lại seed trước mỗi fold để đồng bộ
        set_seeds(args.seed + k)

        model = build_lstm_model(input_shape, num_classes)

        if args.fixed_epochs > 0:
            epochs_to_run = args.fixed_epochs
            callbacks = []
            print(f"    [Protocol P2] Cố định {epochs_to_run} epochs (không EarlyStopping, đánh giá epoch cuối)")
        else:
            epochs_to_run = args.epochs
            callbacks = [
                EarlyStopping(
                    monitor='val_categorical_accuracy',
                    patience=args.patience,
                    restore_best_weights=True,
                    verbose=1
                )
            ]
            print(f"    [Protocol P1] EarlyStopping patience={args.patience}, max_epochs={args.epochs}")

        history = model.fit(
            X_train, y_train,
            epochs=epochs_to_run,
            batch_size=args.batch_size,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=2
        )

        val_acc_list = history.history['val_categorical_accuracy']
        if args.fixed_epochs > 0:
            best_epoch = epochs_to_run
            best_val_acc = float(val_acc_list[-1])
        else:
            best_epoch = int(np.argmax(val_acc_list)) + 1
            best_val_acc = float(np.max(val_acc_list))

        # Đánh giá trên tập TEST
        y_pred_probs = model.predict(X_test, verbose=0)
        y_pred_fold = np.argmax(y_pred_probs, axis=1)
        y_true_fold = np.argmax(y_test, axis=1)

        test_acc = float(accuracy_score(y_true_fold, y_pred_fold))
        test_f1 = float(f1_score(y_true_fold, y_pred_fold, average='macro', zero_division=0))

        print(f"    -> Evaluated Epoch: {best_epoch} (Val Acc: {best_val_acc*100:.2f}%)")
        print(f"    -> Test Accuracy ({test_sess}): {test_acc*100:.2f}% | Macro-F1: {test_f1:.4f}")

        # Confusion Matrix cho từng fold
        cm_path = os.path.join(EVAL_DIR, f"confusion_matrix_fold{k}_{test_sess}{tag_for_plots}.png")
        cm_fold = plot_and_save_confusion_matrix(
            y_true_fold, y_pred_fold, actions, cm_path,
            title=f"LOSO Fold {k} - Test on {test_sess} (Acc: {test_acc*100:.2f}%)"
        )
        overall_cm += cm_fold

        all_y_true.extend(y_true_fold.tolist())
        all_y_pred.extend(y_pred_fold.tolist())

        # Lưu biểu đồ quá trình học fold
        hist_path = os.path.join(EVAL_DIR, f"history_fold{k}_{test_sess}{tag_for_plots}.png")
        plot_and_save_history(history, hist_path, title_prefix=f"LOSO Fold {k} ({test_sess})")

        report_dict = classification_report(
            y_true_fold, y_pred_fold,
            labels=list(range(num_classes)),
            target_names=actions,
            output_dict=True,
            zero_division=0
        )

        fold_results.append({
            'fold': k,
            'test_session': test_sess,
            'val_session': val_sess,
            'train_sessions': train_sess,
            'train_samples': int(X_train.shape[0]),
            'val_samples': int(X_val.shape[0]),
            'test_samples': int(X_test.shape[0]),
            'best_epoch': best_epoch,
            'best_val_accuracy': best_val_acc,
            'test_accuracy': test_acc,
            'macro_f1': test_f1,
            'classification_report': report_dict
        })

    # ============================================================
    # TỔNG HỢP VÀ PHÂN TÍCH LỖI
    # ============================================================
    accuracies = [f['test_accuracy'] for f in fold_results]
    f1_scores = [f['macro_f1'] for f in fold_results]
    best_epochs = [f['best_epoch'] for f in fold_results]

    mean_acc = float(np.mean(accuracies))
    std_acc = float(np.std(accuracies))
    mean_f1 = float(np.mean(f1_scores))
    std_f1 = float(np.std(f1_scores))
    median_best_epoch = int(np.median(best_epochs))

    print("\n" + "=" * 80)
    print("BẢNG TỔNG HỢP KẾT QUẢ ĐÁNH GIÁ LEAVE-ONE-SESSION-OUT (LOSO)")
    print("=" * 80)
    print(f"{'FOLD':<6}{'TEST SESSION':<15}{'TRAIN SESSIONS':<28}{'BEST EPOCH':<12}{'ACCURACY':<12}{'MACRO-F1':<10}")
    print("-" * 80)
    for f in fold_results:
        train_str = "+".join(s.replace("session_", "S") for s in f['train_sessions'])
        test_str = f['test_session'].replace("session_", "S")
        print(f"{f['fold']:<6}{test_str:<15}{train_str:<28}{f['best_epoch']:<12}{f['test_accuracy']*100:6.2f}%     {f['macro_f1']:.4f}")
    print("-" * 80)
    print(f"{'MEAN ± STD':<49}{median_best_epoch} (median) {mean_acc*100:6.2f} ± {std_acc*100:.2f}%  {mean_f1:.4f} ± {std_f1:.4f}")
    print("=" * 80)

    # Tìm top 10 cặp class bị nhầm lẫn nhiều nhất
    confusion_pairs = []
    for i in range(num_classes):
        for j in range(num_classes):
            if i != j and overall_cm[i, j] > 0:
                confusion_pairs.append({
                    'true_class': actions[i],
                    'predicted_class': actions[j],
                    'count': int(overall_cm[i, j])
                })
    confusion_pairs.sort(key=lambda x: x['count'], reverse=True)
    top10_confusions = confusion_pairs[:10]

    print("\n--- TOP 10 CẶP CLASS BỊ NHẦM LẪN NHIỀU NHẤT TRONG LOSO ---")
    print(f"{'THỰC TẾ (GROUND TRUTH)':<25} -> {'DỰ ĐOÁN NHẦM (PREDICTED)':<25} {'SỐ MẪU':>10}")
    print("-" * 65)
    for p in top10_confusions:
        print(f"{p['true_class']:<25} -> {p['predicted_class']:<25} {p['count']:>10}")

    # Lưu confusion matrix tổng hợp
    overall_cm_path = os.path.join(EVAL_DIR, f"confusion_matrix_overall_loso{tag_for_plots}.png")
    plot_and_save_confusion_matrix(
        all_y_true, all_y_pred, actions, overall_cm_path,
        title=f"LOSO Overall Confusion Matrix (Mean Acc: {mean_acc*100:.2f}%)"
    )
    print(f"\n✓ Đã lưu biểu đồ ma trận nhầm lẫn tổng thể tại: {overall_cm_path}")

    # Đóng gói kết quả json
    summary_data = {
        'protocol': 'P2' if args.fixed_epochs > 0 else 'P1',
        'fixed_epochs': args.fixed_epochs,
        'seed': args.seed,
        'segment': args.segment,
        'augment': args.augment,
        'aug_factor': args.aug_factor if args.augment != 'none' else 0,
        'folds': fold_results,
        'summary': {
            'mean_accuracy': mean_acc,
            'std_accuracy': std_acc,
            'mean_macro_f1': mean_f1,
            'std_macro_f1': std_f1,
            'best_epochs': best_epochs,
            'median_best_epoch': median_best_epoch,
            'session_accuracies': {f['test_session']: f['test_accuracy'] for f in fold_results}
        },
        'top10_confusions': top10_confusions
    }

    with open(result_json_path, 'w', encoding='utf-8') as f:
        json.dump(summary_data, f, ensure_ascii=False, indent=2)
    print(f"✓ Đã lưu kết quả chi tiết tại: {result_json_path}")


def run_baseline_evaluation(args, actions):
    print("\n" + "=" * 80)
    print("BẮT ĐẦU HUẤN LUYỆN VÀ ĐÁNH GIÁ BASELINE (STRATIFIED RANDOM SPLIT)")
    print("=" * 80)

    X_train_path = os.path.join(PROCESSED_PATH, 'X_train.npy')
    X_test_path = os.path.join(PROCESSED_PATH, 'X_test.npy')
    y_train_path = os.path.join(PROCESSED_PATH, 'y_train.npy')
    y_test_path = os.path.join(PROCESSED_PATH, 'y_test.npy')

    if not all(os.path.exists(p) for p in [X_train_path, X_test_path, y_train_path, y_test_path]):
        print("[!] Thiếu X_train/X_test. Chạy 'python pipeline/preprocess_data.py --mode random' trước.")
        sys.exit(1)

    X_train = np.load(X_train_path)
    X_test = np.load(X_test_path)
    y_train = np.load(y_train_path)
    y_test = np.load(y_test_path)

    print(f"Train samples: {X_train.shape[0]} | Test samples: {X_test.shape[0]}")
    set_seeds(args.seed)

    num_classes = len(actions)
    model = build_lstm_model((X_train.shape[1], X_train.shape[2]), num_classes)

    callbacks = [
        EarlyStopping(
            monitor='val_categorical_accuracy',
            patience=args.patience,
            restore_best_weights=True,
            verbose=1
        )
    ]

    history = model.fit(
        X_train, y_train,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_data=(X_test, y_test),
        callbacks=callbacks,
        verbose=2
    )

    best_epoch = int(np.argmax(history.history['val_categorical_accuracy'])) + 1

    loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    y_pred_probs = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = np.argmax(y_test, axis=1)

    test_f1 = float(f1_score(y_true, y_pred, average='macro', zero_division=0))

    print("\n[BASELINE KẾT QUẢ]")
    print(f"  - Best Epoch: {best_epoch}")
    print(f"  - Test Accuracy: {test_acc*100:.2f}%")
    print(f"  - Macro-F1: {test_f1:.4f}")

    cm_path = os.path.join(EVAL_DIR, "confusion_matrix_baseline.png")
    plot_and_save_confusion_matrix(y_true, y_pred, actions, cm_path,
                                  title=f"Baseline Random Split (Acc: {test_acc*100:.2f}%)")

    hist_path = os.path.join(EVAL_DIR, "history_baseline.png")
    plot_and_save_history(history, hist_path, title_prefix="Baseline Random Split")

    report = classification_report(y_true, y_pred, labels=list(range(num_classes)),
                                   target_names=actions, output_dict=True, zero_division=0)

    res = {
        'mode': 'random_stratified',
        'train_samples': int(X_train.shape[0]),
        'test_samples': int(X_test.shape[0]),
        'best_epoch': best_epoch,
        'test_accuracy': float(test_acc),
        'macro_f1': float(test_f1),
        'classification_report': report
    }
    with open(BASELINE_RESULTS_PATH, 'w', encoding='utf-8') as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print(f"✓ Đã lưu kết quả baseline vào: {BASELINE_RESULTS_PATH}")


def train_final_model(args, actions):
    print("\n" + "=" * 80)
    print("BẮT ĐẦU HUẤN LUYỆN MODEL CUỐI CÙNG TRÊN TOÀN BỘ 4 SESSIONS")
    print("=" * 80)

    X_all_path = os.path.join(PROCESSED_PATH, 'X_all.npy')
    y_all_path = os.path.join(PROCESSED_PATH, 'y_all.npy')

    if not (os.path.exists(X_all_path) and os.path.exists(y_all_path)):
        print("[!] Thiếu file X_all.npy / y_all.npy. Hãy chạy preprocess_data.py trước.")
        sys.exit(1)

    X_all = np.load(X_all_path)
    y_all = np.load(y_all_path)

    # Xác định số epoch huấn luyện model cuối
    if args.fixed_epochs > 0:
        target_epochs = args.fixed_epochs
        print(f"Dùng target_epochs cố định theo Protocol P2: {target_epochs} epochs")
    elif os.path.exists(LOSO_RESULTS_PATH):
        with open(LOSO_RESULTS_PATH, 'r', encoding='utf-8') as f:
            loso_data = json.load(f)
        target_epochs = loso_data['summary']['median_best_epoch']
        print(f"Đã đọc median best_epoch từ LOSO: {target_epochs} epochs")
    else:
        target_epochs = 60
        print(f"Chưa có loso_results.json, dùng mặc định: {target_epochs} epochs")

    X_raw258_path = os.path.join(PROCESSED_PATH, 'X_raw258_all.npy')
    raw_source = None
    if os.path.exists(X_raw258_path):
        raw_source = np.load(X_raw258_path)

    if args.segment != "none":
        if raw_source is None:
            print("[!] Thiếu file X_raw258_all.npy để chạy segment!")
            sys.exit(1)
        print(f"Áp dụng segment '{args.segment}' cho mô hình cuối...")
        raw_source, _ = segment_dataset(raw_source, mode=args.segment)
        X_all = np.array([build_features(s) for s in raw_source], dtype=np.float32)

    if args.augment != "none":
        if raw_source is not None:
            print(f"Áp dụng augmentation '{args.augment}' (K={args.aug_factor}) cho mô hình cuối...")
            X_raw_aug, y_final = generate_augmented_dataset(
                raw_source, y_all,
                augment_mode=args.augment,
                aug_factor=args.aug_factor,
                target_duration=3.2,
                seed=args.seed
            )
            print(f"Trích xuất build_features cho {len(X_raw_aug)} mẫu...")
            X_final = np.array([build_features(s) for s in X_raw_aug], dtype=np.float32)
        else:
            X_final, y_final = X_all, y_all
    else:
        X_final, y_final = X_all, y_all

    print(f"Huấn luyện trên {X_final.shape[0]} samples với {target_epochs} epochs (không dùng val)...")
    set_seeds(args.seed)

    num_classes = len(actions)
    model = build_lstm_model((X_final.shape[1], X_final.shape[2]), num_classes)

    history = model.fit(
        X_final, y_final,
        epochs=target_epochs,
        batch_size=args.batch_size,
        verbose=1
    )

    model.save(MODEL_SAVE_PATH)
    print(f"\n✅ Đã huấn luyện xong và lưu model cuối vào: {MODEL_SAVE_PATH}")

    # Vẽ và lưu biểu đồ final model
    hist_path = os.path.join(MODEL_DIR, "training_history.png")
    plot_and_save_history(history, hist_path, title_prefix="Final Production Model")
    print(f"✓ Đã lưu biểu đồ tại: {hist_path}")


def main():
    parser = argparse.ArgumentParser(description="Huấn luyện và đánh giá mô hình VSL LSTM")
    parser.add_argument("--eval", type=str, choices=["none", "loso", "baseline"], default="none",
                        help="Chế độ đánh giá: 'loso' (4-fold), 'baseline' (random split), 'none'")
    parser.add_argument("--augment", type=str,
                        choices=["none", "rot_y", "rotate_y", "rot_yz", "rot_y_rot_z", "rot_z", "full"],
                        default="none",
                        help="Phương pháp Data Augmentation trên tập train: 'none', 'rot_y', 'rot_yz', 'full'")
    parser.add_argument("--aug-factor", type=int, default=3,
                        help="Hệ số sinh mẫu biến thể K cho mỗi mẫu gốc (mặc định: 3)")
    parser.add_argument("--final", action="store_true",
                        help="Huấn luyện mô hình cuối cùng trên toàn bộ 4 sessions")
    parser.add_argument("--epochs", type=int, default=200, help="Số epoch tối đa (mặc định: 200)")
    parser.add_argument("--fixed-epochs", type=int, default=0, help="Số epoch cố định cho Protocol P2 (mặc định: 0, dùng EarlyStopping)")
    parser.add_argument("--segment", type=str, choices=["none", "hand", "motion"], default="none",
                        help="Phương pháp cắt đoạn cử chỉ: 'none', 'hand', 'motion'")
    parser.add_argument("--out-json", type=str, default=None, help="Tên file kết quả JSON trong models/eval")
    parser.add_argument("--patience", type=int, default=30, help="Patience cho EarlyStopping (mặc định: 30)")
    parser.add_argument("--batch-size", type=int, default=16, help="Kích thước batch (mặc định: 16)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed cố định (mặc định: 42)")

    args = parser.parse_args()
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(EVAL_DIR, exist_ok=True)

    actions = get_actions(os.path.join(PROJECT_ROOT, "dataset", "raw"))

    if args.eval == "loso":
        run_loso_evaluation(args, actions)
    elif args.eval == "baseline":
        run_baseline_evaluation(args, actions)
    elif args.final:
        train_final_model(args, actions)
    else:
        # Chế độ mặc định tương thích cũ: train trên X_train / X_test hiện có
        print("--- CHẠY CHẾ ĐỘ MẶC ĐỊNH (TRAIN TRÊN DATASET/PROCESSED HIỆN TẠI) ---")
        run_baseline_evaluation(args, actions)


if __name__ == "__main__":
    main()
