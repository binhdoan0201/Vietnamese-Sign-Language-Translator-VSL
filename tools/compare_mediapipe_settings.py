"""
tools/compare_mediapipe_settings.py
===================================
So sánh trực tiếp 2 cấu hình MediaPipe Holistic trên CÙNG MỘT tập 90 frame:
  - Cấu hình A: 640x480, model_complexity=1, refine_face_landmarks=False (khớp lúc train)
  - Cấu hình B: 320x240, model_complexity=0, refine_face_landmarks=False (tối ưu vòng 1)

Mục đích:
  - Đánh giá tỉ lệ phát hiện bàn tay (LH/RH)
  - Sai lệch tuyệt đối trung bình (MAE) của tọa độ vai và bàn tay sau chuẩn hóa build_features
  - Số cửa sổ mà nhãn dự đoán Top-1 của mô hình bị lệch giữa 2 cấu hình

CÁCH DÙNG:
  python tools/compare_mediapipe_settings.py
"""

import os
import sys
import time
import cv2
import numpy as np
import mediapipe as mp
from keras.models import load_model

# Import feature_utils từ thư mục gốc
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from feature_utils import build_features, get_actions

MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "sign_language_model.h5")
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "backup_old_model.h5")

RAW_DIR = os.path.join(PROJECT_ROOT, "dataset", "raw")


def extract_raw_keypoints(results):
    """Trích xuất (258,) đặc trưng thô: Pose(132) + LH(63) + RH(63)"""
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


def capture_frames_from_webcam(num_frames=90, countdown_sec=3):
    """Mở camera, đếm ngược và chụp num_frames vào RAM"""
    print("\n[CAMERA] Đang mở webcam...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[!] Không thể mở webcam. Vui lòng kiểm tra quyền truy cập camera.")
        return None

    # Countdown
    print(f"[CAMERA] Chuẩn bị ký hiệu tay... Đếm ngược {countdown_sec} giây:")
    for s in range(countdown_sec, 0, -1):
        print(f"  >>> Bắt đầu sau {s}...")
        time.sleep(1.0)
    print("  >>> ĐANG CHỤP 90 KHUNG HÌNH (hãy thực hiện cử chỉ)...")

    captured_frames = []
    for i in range(num_frames):
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        # Lật ngang giống data_collector
        frame_flipped = cv2.flip(frame, 1)
        captured_frames.append(frame_flipped)
        time.sleep(0.03)  # mô phỏng ~30 FPS

    cap.release()
    print(f"[CAMERA] Đã chụp thành công {len(captured_frames)} frames vào bộ nhớ.")
    return captured_frames


def run_comparison(captured_frames):
    if not captured_frames or len(captured_frames) < 30:
        print("[!] Không đủ frames để so sánh.")
        return

    num_frames = len(captured_frames)
    print(f"\n--- BẮT ĐẦU PHÂN TÍCH TRÊN {num_frames} FRAMES ---")

    # Khởi tạo 2 cấu hình Holistic
    mp_holistic = mp.solutions.holistic

    holistic_A = mp_holistic.Holistic(
        model_complexity=1,
        refine_face_landmarks=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    holistic_B = mp_holistic.Holistic(
        model_complexity=0,
        refine_face_landmarks=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    raw_258_A = []
    raw_258_B = []

    hand_detected_A = {'left': 0, 'right': 0}
    hand_detected_B = {'left': 0, 'right': 0}

    # Xử lý CÙNG các frame
    for idx, frame in enumerate(captured_frames):
        # Config A: 640x480
        frame_A = cv2.resize(frame, (640, 480))
        rgb_A = cv2.cvtColor(frame_A, cv2.COLOR_BGR2RGB)
        res_A = holistic_A.process(rgb_A)
        kp_A = extract_raw_keypoints(res_A)
        raw_258_A.append(kp_A)

        if res_A.left_hand_landmarks: hand_detected_A['left'] += 1
        if res_A.right_hand_landmarks: hand_detected_A['right'] += 1

        # Config B: 320x240
        frame_B = cv2.resize(frame, (320, 240))
        rgb_B = cv2.cvtColor(frame_B, cv2.COLOR_BGR2RGB)
        res_B = holistic_B.process(rgb_B)
        kp_B = extract_raw_keypoints(res_B)
        raw_258_B.append(kp_B)

        if res_B.left_hand_landmarks: hand_detected_B['left'] += 1
        if res_B.right_hand_landmarks: hand_detected_B['right'] += 1

    holistic_A.close()
    holistic_B.close()

    raw_258_A = np.array(raw_258_A, dtype=np.float32)
    raw_258_B = np.array(raw_258_B, dtype=np.float32)

    # 1. Tỉ lệ phát hiện bàn tay
    rate_lh_A = hand_detected_A['left'] / num_frames * 100
    rate_rh_A = hand_detected_A['right'] / num_frames * 100
    rate_lh_B = hand_detected_B['left'] / num_frames * 100
    rate_rh_B = hand_detected_B['right'] / num_frames * 100

    print("\n" + "=" * 75)
    print("1. TỈ LỆ PHÁT HIỆN BÀN TAY (HAND DETECTION RATE)")
    print("=" * 75)
    print(f"Cấu hình A (640x480, complexity 1): Tay trái = {rate_lh_A:.1f}% | Tay phải = {rate_rh_A:.1f}%")
    print(f"Cấu hình B (320x240, complexity 0): Tay trái = {rate_lh_B:.1f}% | Tay phải = {rate_rh_B:.1f}%")

    # 2. Sai lệch MAE trên không gian đã chuẩn hóa bởi build_features
    feat_384_A = build_features(raw_258_A)  # (N, 384)
    feat_384_B = build_features(raw_258_B)  # (N, 384)

    # Vai: landmark 11 (left) và 12 (right). Mỗi điểm 4 giá trị (x, y, z, vis) -> [44:52]
    shoulder_mae = float(np.mean(np.abs(feat_384_A[:, 44:52] - feat_384_B[:, 44:52])))

    # Bàn tay global: lh [132:195], rh [195:258]
    lh_mae = float(np.mean(np.abs(feat_384_A[:, 132:195] - feat_384_B[:, 132:195])))
    rh_mae = float(np.mean(np.abs(feat_384_A[:, 195:258] - feat_384_B[:, 195:258])))

    # Bàn tay local: lh [258:321], rh [321:384]
    lh_local_mae = float(np.mean(np.abs(feat_384_A[:, 258:321] - feat_384_B[:, 258:321])))
    rh_local_mae = float(np.mean(np.abs(feat_384_A[:, 321:384] - feat_384_B[:, 321:384])))

    print("\n" + "=" * 75)
    print("2. SAI KHÁC TRUYỆT ĐỐI TRUNG BÌNH (MAE) SAU KHI NORMALIZE BUILD_FEATURES")
    print("=" * 75)
    print(f"  - Tọa độ hai vai (Shoulders MAE)     : {shoulder_mae:.5f}")
    print(f"  - Tọa độ bàn tay (Hands Global MAE)  : LH={lh_mae:.5f} | RH={rh_mae:.5f}")
    print(f"  - Hình dáng bàn tay (Hands Local MAE): LH={lh_local_mae:.5f} | RH={rh_local_mae:.5f}")

    # 3. So sánh Top-1 prediction trên các cửa sổ trượt 30 frame
    actions = get_actions(RAW_DIR)
    model = load_model(MODEL_PATH)

    diff_top1_count = 0
    total_windows = num_frames - 30 + 1

    print("\n" + "=" * 75)
    print(f"3. SO SÁNH TOP-1 PREDICTION TRÊN {total_windows} CỬA SỔ TRƯỢT (WINDOW=30)")
    print("=" * 75)

    for w in range(total_windows):
        sub_A = np.expand_dims(feat_384_A[w:w+30], axis=0)
        sub_B = np.expand_dims(feat_384_B[w:w+30], axis=0)

        pred_A = model.predict(sub_A, verbose=0)[0]
        pred_B = model.predict(sub_B, verbose=0)[0]

        top_A = int(np.argmax(pred_A))
        top_B = int(np.argmax(pred_B))

        if top_A != top_B:
            diff_top1_count += 1
            print(f"  [Lệch ở window {w:02d}]: A='{actions[top_A]}' ({pred_A[top_A]*100:.1f}%) "
                  f"!= B='{actions[top_B]}' ({pred_B[top_B]*100:.1f}%)")

    diff_pct = (diff_top1_count / total_windows) * 100 if total_windows > 0 else 0
    print("-" * 75)
    print(f"👉 Tổng số cửa sổ có Top-1 KHÁC NHAU: {diff_top1_count}/{total_windows} ({diff_pct:.1f}%)")
    print("=" * 75)


def main():
    print("=" * 75)
    print("KIỂM CHỨNG TÁC ĐỘNG CỦA CẤU HÌNH MEDIAPIPE (A vs B)")
    print("=" * 75)

    frames = capture_frames_from_webcam(num_frames=90, countdown_sec=3)
    if frames is None:
        print("\n[!] CẦN NGƯỜI DÙNG CHẠY:")
        print("    Lệnh: python tools/compare_mediapipe_settings.py")
        sys.exit(0)

    run_comparison(frames)


if __name__ == "__main__":
    main()
