"""
realtime_test.py
================
Nhận diện ngôn ngữ ký hiệu realtime qua webcam + phát âm thanh tiếng Việt.

CẢI TIẾN VÒNG 2 (TỐI ƯU HÓA ĐỘ TRỄ & ĐỒNG BỘ MEDIAPIPE):
  - Dùng buffer theo THỜI GIAN (TARGET_DURATION = 3.2 giây) và Temporal Resampling về 30 frame
    trên 258 chiều thô (TRƯỚC build_features), giúp ổn định trên các webcam có FPS khác nhau.
  - Tối ưu suy luận qua tf.function warm-up (độ trễ ~2ms thay vì ~60ms).
  - Cooldown theo THỜI GIAN (COOLDOWN_SEC = 1.0 giây).
  - Làm mượt dự đoán: PredictionSmoother yêu cầu top-1 ổn định qua 3 lần predict và vượt THRESHOLD.
  - MediaPipe Holistic hoàn nguyên về chuẩn: model_complexity=1, refine_face_landmarks=False.
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import time
import json
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import pygame
from gtts import gTTS
from keras.models import load_model
from PIL import Image, ImageDraw, ImageFont

from feature_utils import build_features, get_actions
from temporal_utils import resample_raw_sequence, PredictionSmoother
from pipeline.segment_utils import is_hand_present, get_wrist_speeds

# Khởi tạo bộ phát âm thanh chạy ngầm
pygame.mixer.init()


# ==========================================
# HÀM HỖ TRỢ VẼ CHỮ TIẾNG VIỆT (DÙNG PIL)
# ==========================================
def put_text_vietnamese(img, text, position, font_size, color=(255, 255, 255)):
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)

    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except IOError:
        font = ImageFont.load_default()

    draw.text(position, text, font=font, fill=color)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


# ==========================================
# 1. KHỞI TẠO DANH SÁCH TỪ VỰNG & ÂM THANH
# ==========================================
actions = np.array(get_actions('dataset/raw'))
print(f"Đã nạp {len(actions)} từ vựng: {actions}")

os.makedirs("audio", exist_ok=True)

try:
    with open('dictionary.json', 'r', encoding='utf-8') as f:
        VIETNAMESE_DICT = json.load(f)
except FileNotFoundError:
    print("⚠️ Không tìm thấy dictionary.json, sẽ dùng tạm tên không dấu!")
    VIETNAMESE_DICT = {}

print("Đang kiểm tra dữ liệu âm thanh...")
for action in actions:
    audio_path = f"audio/voice_{action}.mp3"
    if not os.path.exists(audio_path):
        print(f"-> Chưa có âm thanh cho '{action}', đang tạo...")
        try:
            text_to_speak = VIETNAMESE_DICT.get(action, action.replace('_', ' '))
            tts = gTTS(text=text_to_speak, lang='vi')
            tts.save(audio_path)
        except Exception as e:
            print(f"Lỗi mạng khi tạo âm thanh: {e}")
print("✅ Đã chuẩn bị xong toàn bộ âm thanh!")

# ==========================================
# 2. LOAD MODEL
# ==========================================
MODEL_PATH = 'models/sign_language_model.h5'
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = 'models/backup_old_model.h5'

print(f"Đang load model từ {MODEL_PATH}...")
model = load_model(MODEL_PATH)
expected_dim = model.input_shape[-1]
print(f"Model kỳ vọng đầu vào: {model.input_shape} ({expected_dim} chiều)")

# Warm-up tf.function cho suy luận siêu tốc
@tf.function(reduce_retracing=True)
def fast_predict(x):
    return model(x, training=False)

_ = fast_predict(tf.zeros((1, 30, expected_dim), dtype=tf.float32)).numpy()
print("✅ Đã warm-up tf.function suy luận siêu tốc.")

# ==========================================
# 3. TRÍCH XUẤT KEYPOINT (258 chiều thô)
# ==========================================
def extract_keypoints(results):
    """Trả về (258,) tọa độ THÔ: Pose(132) + LH(63) + RH(63)."""
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


# ==========================================
# 4. CẤU HÌNH VÒNG LẶP REALTIME
# ==========================================
TARGET_DURATION = 3.2       # Thời gian cửa sổ chuẩn (giây) theo benchmark thu thập (3.184s)
SEQUENCE_LENGTH = 30        # Số frame đầu ra
MIN_FILL_RATIO = 0.9        # Tỉ lệ tối thiểu của buffer (3.2s * 0.9 = 2.88s) để kích hoạt predict
THRESHOLD = 0.6             # Ngưỡng tự tin tối thiểu
COOLDOWN_SEC = 1.0          # Thời gian nghỉ sau khi nhận diện (giây) - đặt 1.0s
SMOOTH_WINDOW = 3           # Số lần predict liên tiếp để xác nhận
PREDICT_INTERVAL_SEC = 0.05 # Giãn cách tối thiểu giữa các lần predict
MP_COMPLEXITY = 1           # Hoàn nguyên về 1 khớp lúc train

# Cổng chuyển động (Motion Gate - Task 3.1 & Task D1-D2 Vòng 4)
GATE_ENABLED = True          # Mặc định BẬT theo kết quả thực nghiệm A/B (29/38 đúng vs 16/38, triệt tiêu nhãn thừa lúc nghỉ)
GATE_MIN_HAND_RATIO = 0.50    # Tỉ lệ frame có tay >= 50%
GATE_MOTION_STAT = "max"      # "max" | "median" (Task D2, mặc định "max")
GATE_MIN_MOTION_MAX = 0.05    # Vận tốc cổ tay cực đại tối thiểu (ngưỡng từ Vòng 3)
GATE_MIN_MOTION_MEDIAN = 0.025 # Ngưỡng trung vị tốc độ cổ tay (phân tích gesture span vòng 3)
GATE_MIN_MOTION = GATE_MIN_MOTION_MAX if GATE_MOTION_STAT == "max" else GATE_MIN_MOTION_MEDIAN
STATIC_CLASSES = [
    'i', 'h', 'd', 'l', 'k', 'x', 'u', 'e', 'v', 'dau_nga', 'y', 's'
]                             # 12 class tĩnh từ Task 1.2

time_buffer = []            # list các tuple (timestamp, keypoints_258)
current_action = ""
cooldown_until = 0.0
last_predict_time = 0.0
smoother = PredictionSmoother(window_size=SMOOTH_WINDOW, threshold=THRESHOLD)

# MediaPipe Holistic
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Không mở được camera!")
    exit()

print("Load model thành công! Bật camera...")

with mp_holistic.Holistic(
    model_complexity=MP_COMPLEXITY,
    refine_face_landmarks=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as holistic:
    while cap.isOpened():
        t_now = time.perf_counter()
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        image = frame.copy()
        results = holistic.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        mp_drawing.draw_landmarks(image, results.pose_landmarks,
                                  mp_holistic.POSE_CONNECTIONS)
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks,
                                  mp_holistic.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks,
                                  mp_holistic.HAND_CONNECTIONS)

        keypoints = extract_keypoints(results)
        has_hand = (results.left_hand_landmarks is not None or
                    results.right_hand_landmarks is not None)

        now_wall = time.time()

        # ----- Cooldown theo THỜI GIAN -----
        if now_wall < cooldown_until:
            time_buffer = []
        else:
            time_buffer.append((t_now, keypoints))
            # Cắt tỉa buffer chỉ giữ khoảng thời gian gần nhất (hệ số 1.5x)
            min_t = t_now - (TARGET_DURATION * 1.5)
            time_buffer = [(t, f) for (t, f) in time_buffer if t >= min_t]

            buffer_duration = time_buffer[-1][0] - time_buffer[0][0] if len(time_buffer) >= 2 else 0.0

            # ----- Dự đoán khi đủ thời gian cửa sổ và có tay -----
            if (buffer_duration >= TARGET_DURATION * MIN_FILL_RATIO and has_hand
                    and (t_now - last_predict_time >= PREDICT_INTERVAL_SEC)):

                last_predict_time = t_now

                ts = [t for t, f in time_buffer]
                raw_f = [f for t, f in time_buffer]

                # Resample về đúng 30 frame theo TARGET_DURATION trên đặc trưng thô 258
                resampled_258 = resample_raw_sequence(ts, raw_f, TARGET_DURATION, SEQUENCE_LENGTH)

                if resampled_258 is not None:
                    # Kiểm tra Cổng chuyển động (Motion Gate - Task 3.1)
                    hand_count = sum(1 for f in resampled_258 if is_hand_present(f))
                    hand_ratio = hand_count / float(len(resampled_258))
                    wrist_speeds = get_wrist_speeds(resampled_258)
                    if GATE_MOTION_STAT == "median":
                        motion_energy = float(np.median(wrist_speeds)) if len(wrist_speeds) > 0 else 0.0
                    else:
                        motion_energy = float(np.max(wrist_speeds)) if len(wrist_speeds) > 0 else 0.0

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

                    norm_seq = build_features(resampled_258)
                    norm_tensor = tf.convert_to_tensor(np.expand_dims(norm_seq, axis=0), dtype=tf.float32)
                    res = fast_predict(norm_tensor).numpy()[0]

                    accepted, top_idx, top_conf, avg_probs = smoother.update(res)

                    # In thông tin debug
                    print(f"\n--- DỰ ĐOÁN (Buffer: {buffer_duration:.2f}s | Gate: {gate_state}) --- (Top: {actions[top_idx]} {top_conf*100:.1f}%)")
                    for i, action in enumerate(actions[:5]):  # in 5 class đầu
                        bar = '█' * int(avg_probs[i] * 20)
                        print(f" > {action:<15} {avg_probs[i]*100:6.2f}% {bar}")

                    if accepted:
                        predicted_action = actions[top_idx]
                        if GATE_ENABLED and gate_state == "eval_static":
                            if predicted_action in STATIC_CLASSES:
                                gate_open = True
                                gate_state = "open_static"
                            else:
                                gate_open = False
                                gate_state = "closed_low_motion"

                        if not gate_open:
                            smoother.reset()
                        elif predicted_action != current_action:
                            current_action = predicted_action
                            time_buffer = []
                            smoother.reset()
                            cooldown_until = now_wall + COOLDOWN_SEC

                            try:
                                pygame.mixer.music.load(f"audio/voice_{current_action}.mp3")
                                pygame.mixer.music.play()
                            except Exception:
                                pass

        # ----- Hiển thị giao diện OpenCV -----
        cv2.rectangle(image, (0, 0), (image.shape[1], 50), (245, 117, 16), -1)

        if now_wall < cooldown_until:
            rem = cooldown_until - now_wall
            label = VIETNAMESE_DICT.get(current_action, current_action)
            display_text = f"{label}  (nghi {rem:.1f}s...)"
        elif current_action:
            display_text = VIETNAMESE_DICT.get(current_action, current_action)
        else:
            display_text = ""

        image = put_text_vietnamese(image, display_text,
                                    position=(10, 5), font_size=32,
                                    color=(255, 255, 255))

        # Thanh tiến độ gom frame theo thời gian
        cur_dur = time_buffer[-1][0] - time_buffer[0][0] if len(time_buffer) >= 2 else 0.0
        progress = min(1.0, cur_dur / TARGET_DURATION)
        bar_w = int(progress * (image.shape[1] - 20))
        cv2.rectangle(image, (10, image.shape[0] - 25),
                      (10 + bar_w, image.shape[0] - 10), (0, 255, 0), -1)

        cv2.imshow('Dich Thuat Ngon Ngu Ky Hieu - Realtime', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
