"""
tools/benchmark_fps.py
======================
Đo đạc và phân tích hiệu năng:
  (a) Thời gian thực của vòng quay 30 frame giống data_collector.py
      (cap.read + holistic.process + waitKey(33)) -> ước lượng FPS thu thập và độ dài (giây) 1 sample.
  (b) Độ trễ chi tiết của từng thành phần: MediaPipe Holistic, build_features, model.predict.
  (c) So sánh trước vs sau tối ưu (640x480 vs 320x240, model_complexity=1 vs 0).

CÁCH DÙNG:
  python tools/benchmark_fps.py
"""

import os
import sys
import time
import cv2
import numpy as np
import mediapipe as mp
from keras.models import load_model

# Đảm bảo import được từ thư mục gốc
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from feature_utils import build_features

MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "sign_language_model.h5")
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "backup_old_model.h5")


def extract_raw_keypoints(results):
    pose = np.array([[r.x, r.y, r.z, r.visibility]
                     for r in results.pose_landmarks.landmark]).flatten() \
        if results.pose_landmarks else np.zeros(132)
    lh = np.array([[r.x, r.y, r.z]
                   for r in results.left_hand_landmarks.landmark]).flatten() \
        if results.left_hand_landmarks else np.zeros(63)
    rh = np.array([[r.x, r.y, r.z]
                   for r in results.right_hand_landmarks.landmark]).flatten() \
        if results.right_hand_landmarks else np.zeros(63)
    return np.concatenate([pose, lh, rh]).astype(np.float32)


def benchmark_collection_loop(cap, num_trials=3):
    """(a) Đo chu kỳ thu thập 30 frame như data_collector.py"""
    print("\n" + "=" * 70)
    print("PHẦN (a): ĐO VÒNG QUAY 30 FRAME NHƯ data_collector.py")
    print("=" * 70)

    mp_holistic = mp.solutions.holistic
    holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    durations = []
    fps_list = []

    for trial in range(num_trials):
        # Warmup 5 frame
        for _ in range(5):
            ret, frame = cap.read()
            if ret:
                holistic.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        t_start = time.perf_counter()
        for i in range(30):
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(rgb)
            _ = extract_raw_keypoints(results)
            cv2.waitKey(33)  # Giữ đúng waitKey(33) như data_collector.py

        t_end = time.perf_counter()
        dur = t_end - t_start
        fps = 30.0 / dur
        durations.append(dur)
        fps_list.append(fps)
        print(f"  Lần {trial+1}: 30 frames hoàn thành trong {dur:.3f}s | FPS thực tế: {fps:.2f}")

    holistic.close()
    mean_dur = float(np.mean(durations))
    mean_fps = float(np.mean(fps_list))

    print("-" * 70)
    print(f"👉 KẾT QUẢ THU THẬP: Độ dài 1 sample (30 frame) = {mean_dur:.3f} giây")
    print(f"👉 FPS trung bình khi quay dữ liệu = {mean_fps:.2f} FPS")
    print(f"👉 Khuyến nghị cấu hình TARGET_DURATION = {mean_dur:.2f} giây")
    return mean_dur, mean_fps


def benchmark_components(cap):
    """(b) Đo thời gian xử lý từng thành phần riêng biệt"""
    print("\n" + "=" * 70)
    print("PHẦN (b): ĐO CHI TIẾT ĐỘ TRỄ TỪNG MODULE")
    print("=" * 70)

    # 1. MediaPipe Holistic mặc định (640x480, complexity 1)
    mp_holistic = mp.solutions.holistic
    h_default = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    # Đọc 1 frame thật từ camera
    ret, frame_640 = cap.read()
    if not ret or frame_640 is None:
        frame_640 = np.zeros((480, 640, 3), dtype=np.uint8)

    rgb_640 = cv2.cvtColor(frame_640, cv2.COLOR_BGR2RGB)

    # Warmup
    for _ in range(5):
        h_default.process(rgb_640)

    times_h_def = []
    for _ in range(30):
        t0 = time.perf_counter()
        h_default.process(rgb_640)
        times_h_def.append(time.perf_counter() - t0)
    h_default.close()
    lat_h_def = float(np.mean(times_h_def)) * 1000

    # 2. MediaPipe Holistic tối ưu (320x240, model_complexity=0)
    h_opt = mp_holistic.Holistic(
        model_complexity=0,
        refine_face_landmarks=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    frame_320 = cv2.resize(frame_640, (320, 240))
    rgb_320 = cv2.cvtColor(frame_320, cv2.COLOR_BGR2RGB)

    for _ in range(5):
        h_opt.process(rgb_320)

    times_h_opt = []
    for _ in range(30):
        t0 = time.perf_counter()
        h_opt.process(rgb_320)
        times_h_opt.append(time.perf_counter() - t0)
    h_opt.close()
    lat_h_opt = float(np.mean(times_h_opt)) * 1000

    # 3. build_features (30 frames thô 258 -> 384)
    dummy_seq = np.random.randn(30, 258).astype(np.float32)
    times_feat = []
    for _ in range(100):
        t0 = time.perf_counter()
        _ = build_features(dummy_seq)
        times_feat.append(time.perf_counter() - t0)
    lat_feat = float(np.mean(times_feat)) * 1000

    # 4. model.predict
    lat_model = 0.0
    if os.path.exists(MODEL_PATH):
        try:
            model = load_model(MODEL_PATH)
            dummy_input = np.random.randn(1, 30, 384).astype(np.float32)
            # Warmup
            model.predict(dummy_input, verbose=0)
            times_model = []
            for _ in range(50):
                t0 = time.perf_counter()
                model.predict(dummy_input, verbose=0)
                times_model.append(time.perf_counter() - t0)
            lat_model = float(np.mean(times_model)) * 1000
        except Exception as e:
            print(f"[!] Không đo được model.predict: {e}")

    print(f"1. MediaPipe Holistic 640x480 (complexity 1) : {lat_h_def:6.2f} ms/frame (~{1000/lat_h_def:4.1f} FPS)")
    print(f"2. MediaPipe Holistic 320x240 (complexity 0) : {lat_h_opt:6.2f} ms/frame (~{1000/lat_h_opt:4.1f} FPS)")
    print(f"   -> Giảm độ trễ Holistic                  : {lat_h_def - lat_h_opt:6.2f} ms ({(lat_h_def - lat_h_opt)/lat_h_def*100:4.1f}%)")
    print(f"3. build_features (30 frames 258 -> 384)     : {lat_feat:6.2f} ms/lần")
    print(f"4. LSTM model.predict (1 sample 30x384)       : {lat_model:6.2f} ms/lần")
    print(f"   Tổng thời gian 1 chu kỳ predict (tối ưu)  : {lat_h_opt + lat_feat + lat_model:6.2f} ms")


def main():
    print("=" * 70)
    print("BẮT ĐẦU BENCHMARK HỆ THỐNG")
    print("=" * 70)

    cap = cv2.VideoCapture(0)
    has_camera = cap.isOpened()
    if not has_camera:
        print("[!] Không mở được webcam vật lý. Tạo camera giả lập (640x480 synthetic video).")
        # Sử dụng dummy capture
        class DummyCap:
            def isOpened(self): return True
            def read(self):
                # Khung hình giả lập có vẽ người đơn giản để holistic có cái xử lý
                img = np.zeros((480, 640, 3), dtype=np.uint8)
                return True, img
            def release(self): pass
        cap = DummyCap()
    else:
        print("✓ Đã kết nối webcam thành công.")

    try:
        benchmark_collection_loop(cap)
        benchmark_components(cap)
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
