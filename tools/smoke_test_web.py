"""
tools/smoke_test_web.py
=======================
Kiểm thử khói (Smoke Test) luồng suy luận và ghi nhận sự kiện của app.py (Task A3).

Quy trình:
  1. Nạp model thật (models/sign_language_model.h5) và tạo hàm fast_predict.
  2. Nạp 1 mẫu thật từ dataset/processed/X_raw258_all.npy (shape 30, 258).
  3. Đưa chuỗi qua đúng đường xử lý dự đoán của app.py:
     - Tạo timestamps 9.5 FPS (khoảng cách 0.1067s, đủ 3.2s)
     - Temporal Resampling (resample_raw_sequence)
     - Cổng chuyển động (Motion Gate)
     - Trích xuất đặc trưng (build_features -> 384 chiều)
     - fast_predict qua tf.function
     - Bộ làm mượt (PredictionSmoother, SMOOTH_WINDOW=3)
     - Ghi nhận sự kiện qua pipeline.web_logging.write_event
  4. Xác nhận file CSV đích có đúng dòng sự kiện được ghi.
"""

import os
import sys
import time
import tempfile
import numpy as np
import tensorflow as tf
from keras.models import load_model

# Ép stdout UTF-8
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

RAW_DIR = os.path.join(PROJECT_ROOT, "dataset", "raw")
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "sign_language_model.h5")
RAW258_PATH = os.path.join(PROJECT_ROOT, "dataset", "processed", "X_raw258_all.npy")
ACTIONS_PATH = os.path.join(PROJECT_ROOT, "dataset", "processed", "y_all.npy")

from feature_utils import build_features, get_actions
from temporal_utils import resample_raw_sequence, PredictionSmoother
from pipeline.segment_utils import is_hand_present, get_wrist_speeds
from pipeline.web_logging import write_event


def run_smoke_test():
    print("=" * 70)
    print("BẮT ĐẦU SMOKE TEST ĐẦU-CUỐI ĐƯỜNG XỬ LÝ DỰ ĐOÁN (APP.PY)")
    print("=" * 70)

    # 1. Kiểm tra tài nguyên
    if not os.path.exists(MODEL_PATH):
        print(f"[FAIL] Không tìm thấy model tại {MODEL_PATH}")
        sys.exit(1)
    if not os.path.exists(RAW258_PATH):
        print(f"[FAIL] Không tìm thấy dữ liệu thô tại {RAW258_PATH}")
        sys.exit(1)

    actions = get_actions(RAW_DIR)
    print(f"[1/5] Đã nạp danh sách {len(actions)} actions.")

    # 2. Nạp model & fast_predict
    print(f"[2/5] Đang nạp model từ {MODEL_PATH}...")
    model = load_model(MODEL_PATH, compile=False)

    @tf.function(reduce_retracing=True)
    def fast_predict(x):
        return model(x, training=False)

    # Warm-up
    _ = fast_predict(tf.zeros((1, 30, 384), dtype=tf.float32)).numpy()
    print("      -> Đã warm-up fast_predict thành công.")

    # 3. Nạp 1 mẫu thật từ dataset
    print(f"[3/5] Đang nạp 1 mẫu dữ liệu thô 258 chiều từ {RAW258_PATH}...")
    raw258_all = np.load(RAW258_PATH)
    sample_258 = raw258_all[0]  # shape (30, 258)
    print(f"      -> Sample shape: {sample_258.shape}")

    # Giả lập buffer 30 frame với timestamp ~9.5 FPS (3.2s)
    fps = 9.5
    dt = 1.0 / fps
    now_wall = time.time()
    t_start = now_wall - (30 * dt)
    time_buffer = [(t_start + i * dt, sample_258[i]) for i in range(30)]
    buffer_duration = time_buffer[-1][0] - time_buffer[0][0]
    print(f"      -> Giả lập buffer: {len(time_buffer)} frames, duration={buffer_duration:.2f}s")

    # 4. Mô phỏng đúng đường xử lý của app.py
    print("[4/5] Chạy qua pipeline xử lý của app.py...")
    TARGET_DURATION = 3.2
    SEQUENCE_LENGTH = 30
    THRESHOLD = 0.6
    SMOOTH_WINDOW = 3
    GATE_ENABLED = False  # Kiểm tra chế độ mặc định
    GATE_MIN_HAND_RATIO = 0.50
    GATE_MIN_MOTION = 0.05
    STATIC_CLASSES = ['i', 'h', 'd', 'l', 'k', 'x', 'u', 'e', 'v', 'dau_nga', 'y', 's']

    smoother = PredictionSmoother(window_size=SMOOTH_WINDOW, threshold=THRESHOLD)

    # Sử dụng file log tạm thời để kiểm thử
    temp_dir = tempfile.TemporaryDirectory()
    test_events_path = os.path.join(temp_dir.name, "smoke_web_events.csv")

    timestamps = [t for t, f in time_buffer]
    raw_frames = [f for t, f in time_buffer]

    # Temporal Resampling
    resampled_258 = resample_raw_sequence(
        timestamps, raw_frames,
        target_duration=TARGET_DURATION,
        target_length=SEQUENCE_LENGTH
    )
    assert resampled_258 is not None, "Resampling trả về None!"

    # Cổng chuyển động (Motion Gate)
    hand_count = sum(1 for f in resampled_258 if is_hand_present(f))
    hand_ratio = hand_count / float(len(resampled_258))
    wrist_speeds = get_wrist_speeds(resampled_258)
    motion_energy = float(np.max(wrist_speeds))

    gate_open = True
    gate_state = "bypassed"
    if GATE_ENABLED:
        if hand_ratio < GATE_MIN_HAND_RATIO:
            gate_open = False
            gate_state = "closed_no_hand"
        elif motion_energy < GATE_MIN_MOTION:
            gate_state = "eval_static"
        else:
            gate_open = True
            gate_state = "open_motion"

    # Trích xuất đặc trưng 384
    norm_seq = build_features(resampled_258)
    norm_tensor = tf.convert_to_tensor(np.expand_dims(norm_seq, axis=0), dtype=tf.float32)

    # Đưa qua 3 lần liên tiếp để smoother xác nhận (SMOOTH_WINDOW = 3)
    accepted = False
    top_idx = None
    top_conf = 0.0

    for step in range(SMOOTH_WINDOW):
        pred_res = fast_predict(norm_tensor).numpy()[0]
        accepted, top_idx, top_conf, avg_probs = smoother.update(pred_res)
        print(f"      -> Lần dự đoán {step+1}/{SMOOTH_WINDOW}: Top = '{actions[top_idx]}' ({top_conf*100:.1f}%) | Accepted = {accepted}")

    assert accepted, "Smoother phải chấp nhận sau 3 lần liên tiếp cùng nhãn!"

    predicted_action = actions[top_idx]
    if GATE_ENABLED and gate_state == "eval_static":
        if predicted_action in STATIC_CLASSES:
            gate_open = True
            gate_state = "open_static"
        else:
            gate_open = False
            gate_state = "closed_low_motion"

    assert gate_open, f"Cổng chuyển động không được chặn trong smoke test! State: {gate_state}"

    # Ghi sự kiện vào CSV
    write_res = write_event(
        test_events_path,
        action=predicted_action,
        confidence=top_conf,
        t_first_hand_to_output_s=1.234,
        gate_state=gate_state
    )
    assert write_res, "write_event trả về False!"

    # 5. Xác minh file CSV
    print(f"[5/5] Xác minh file log sự kiện: {test_events_path}...")
    assert os.path.exists(test_events_path), "File CSV không tồn tại!"
    with open(test_events_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        print(f"      -> Header: {lines[0].strip()}")
        print(f"      -> Dòng 1: {lines[1].strip()}")
        assert len(lines) == 2, f"Kỳ vọng 2 dòng (1 header + 1 data), nhưng có {len(lines)} dòng!"
        data_parts = lines[1].strip().split(",")
        assert data_parts[1] == predicted_action, f"Kỳ vọng nhãn '{predicted_action}', thực tế '{data_parts[1]}'"
        assert data_parts[4] == gate_state, f"Kỳ vọng gate_state '{gate_state}', thực tế '{data_parts[4]}'"

    temp_dir.cleanup()
    print("-" * 70)
    print("✅ SMOKE TEST THÀNH CÔNG 100%! Luồng dự đoán và ghi log web_events.csv hoạt động hoàn hảo.")
    print("=" * 70)


if __name__ == '__main__':
    run_smoke_test()
