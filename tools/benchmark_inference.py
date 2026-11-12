"""
tools/benchmark_inference.py
============================
So sánh độ chính xác số học và độ trễ suy luận giữa:
  - Phương pháp 1: model.predict(x) (Keras cũ)
  - Phương pháp 2: model(x, training=False).numpy() (gọi trực tiếp)
  - Phương pháp 3: tf.function warm-up (tối ưu hóa đồ thị tính toán)

Yêu cầu (Tác vụ 2.1):
  - Chứng minh kết quả tương đương số học trên >= 200 sample từ X_all.npy: max |diff| < 1e-5
  - Đo thời gian trước / sau (mean, p95) để lượng hóa mức độ giảm độ trễ

CÁCH DÙNG:
  python tools/benchmark_inference.py
"""

import os
import sys
import time
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from keras.models import load_model

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "dataset", "processed")
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "sign_language_model.h5")
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "backup_round1.h5")


def main():
    print("=" * 80)
    print("BENCHMARK SO SÁNH ĐỘ TRỄ SUY LUẬN (INFERENCE LATENCY)")
    print("=" * 80)

    X_all_path = os.path.join(PROCESSED_DIR, "X_all.npy")
    if not os.path.exists(X_all_path):
        print(f"[!] Không tìm thấy {X_all_path}")
        sys.exit(1)

    X_all = np.load(X_all_path)
    num_test_samples = min(300, len(X_all))
    test_data = X_all[:num_test_samples]
    print(f"✓ Đã nạp {num_test_samples} mẫu thực tế từ X_all.npy để kiểm thử (shape: {test_data.shape})")

    print(f"✓ Đang nạp mô hình từ {MODEL_PATH}...")
    model = load_model(MODEL_PATH)

    # Khởi tạo hàm tf.function được tối ưu
    @tf.function(reduce_retracing=True)
    def infer_fn(x):
        return model(x, training=False)

    # Warmup
    dummy = np.zeros((1, 30, 384), dtype=np.float32)
    _ = model.predict(dummy, verbose=0)
    _ = model(dummy, training=False).numpy()
    _ = infer_fn(dummy).numpy()

    # ============================================================
    # 1. KIỂM CHỨNG TƯƠNG ĐƯƠNG SỐ HỌC (NUMERICAL EQUIVALENCE)
    # ============================================================
    print("\n--- 1. KIỂM CHỨNG TƯƠNG ĐƯƠNG SỐ HỌC TRÊN 300 SAMPLES ---")
    preds_predict = []
    preds_direct = []
    preds_tffunc = []

    for i in range(num_test_samples):
        x = np.expand_dims(test_data[i], axis=0)
        p1 = model.predict(x, verbose=0)[0]
        p2 = model(x, training=False).numpy()[0]
        p3 = infer_fn(x).numpy()[0]

        preds_predict.append(p1)
        preds_direct.append(p2)
        preds_tffunc.append(p3)

    preds_predict = np.array(preds_predict)
    preds_direct = np.array(preds_direct)
    preds_tffunc = np.array(preds_tffunc)

    diff_direct = float(np.max(np.abs(preds_predict - preds_direct)))
    diff_tffunc = float(np.max(np.abs(preds_predict - preds_tffunc)))

    print(f"Max |model.predict - direct call| : {diff_direct:.2e}")
    print(f"Max |model.predict - tf.function| : {diff_tffunc:.2e}")

    passed = (diff_direct < 1e-5 and diff_tffunc < 1e-5)
    print(f"👉 Kết luận tương đương số học (< 1e-5): {'✅ PASS' if passed else '❌ FAIL'}")

    # ============================================================
    # 2. ĐO THỜI GIAN SUY LUẬN (MEAN, P95)
    # ============================================================
    print("\n--- 2. ĐO THỜI GIAN SUY LUẬN TRÊN 300 LẦN CHẠY ---")

    # Đo model.predict
    times_predict = []
    for i in range(num_test_samples):
        x = np.expand_dims(test_data[i], axis=0)
        t0 = time.perf_counter()
        _ = model.predict(x, verbose=0)
        times_predict.append((time.perf_counter() - t0) * 1000.0)

    # Đo direct call: model(x, training=False)
    times_direct = []
    for i in range(num_test_samples):
        x = np.expand_dims(test_data[i], axis=0)
        t0 = time.perf_counter()
        _ = model(x, training=False).numpy()
        times_direct.append((time.perf_counter() - t0) * 1000.0)

    # Đo tf.function: infer_fn(x)
    times_tffunc = []
    for i in range(num_test_samples):
        x = np.expand_dims(test_data[i], axis=0)
        t0 = time.perf_counter()
        _ = infer_fn(x).numpy()
        times_tffunc.append((time.perf_counter() - t0) * 1000.0)

    m1, p95_1 = float(np.mean(times_predict)), float(np.percentile(times_predict, 95))
    m2, p95_2 = float(np.mean(times_direct)), float(np.percentile(times_direct, 95))
    m3, p95_3 = float(np.mean(times_tffunc)), float(np.percentile(times_tffunc, 95))

    print("=" * 80)
    print(f"{'PHƯƠNG PHÁP SUY LUẬN':<35}{'MEAN (ms)':<15}{'P95 (ms)':<15}{'TỐC ĐỘ / GIẢM TRỄ'}")
    print("-" * 80)
    print(f"{'1. model.predict(x) (Bản cũ)':<35}{m1:8.2f} ms     {p95_1:8.2f} ms     Baseline")
    print(f"{'2. model(x, training=False)':<35}{m2:8.2f} ms     {p95_2:8.2f} ms     Giảm {(m1-m2)/m1*100:5.1f}% trễ")
    print(f"{'3. tf.function đã warm-up':<35}{m3:8.2f} ms     {p95_3:8.2f} ms     Giảm {(m1-m3)/m1*100:5.1f}% trễ")
    print("=" * 80)
    print(f"👉 Khuyến nghị: Sử dụng tf.function / direct call giúp giảm từ {m1:.1f}ms xuống {m3:.1f}ms (nhanh gấp ~{m1/m3:.1f} lần)!")


if __name__ == "__main__":
    main()
