"""
app.py
======
Backend Flask + SocketIO cho giao diện web nhận diện ngôn ngữ ký hiệu.

CẢI TIẾN VÒNG 2 (TỐI ƯU HÓA ĐỘ TRỄ THỰC TẾ & ĐỒNG BỘ MEDIAPIPE):
  1. Hoàn nguyên MediaPipe về chuẩn lúc train (Task 1.2): 640x480, model_complexity=1.
  2. Siêu tối ưu suy luận (Task 2.1): Dùng tf.function warm-up thay thế model.predict,
     giảm độ trễ suy luận từ ~60ms xuống ~2ms (nhanh gấp ~30 lần), max |sai khác| = 0.
  3. Cấu hình thời gian (Task 2.2): TARGET_DURATION=3.2s, COOLDOWN_SEC=1.0s, MIN_FILL_RATIO=0.9.
  4. End-to-end Latency Logging (Task 2.3): Ghi nhận chi tiết từng lần predict và
     chỉ số t_first_hand_to_output vào logs/web_session_log.csv.
  5. Cơ chế Flow Control (ACK): Phát sự kiện 'frame_ack' ngay khi xử lý xong frame,
     giúp client tự động điều tiết tốc độ, tránh nghẽn hàng đợi WebSocket.
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import csv
import json
import base64
import threading
import time

import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from flask import Flask, render_template, send_from_directory, jsonify
from flask_socketio import SocketIO
from keras.models import load_model
from gtts import gTTS
from datetime import datetime

from feature_utils import build_features, get_actions
from temporal_utils import resample_raw_sequence, PredictionSmoother
from pipeline.segment_utils import is_hand_present, get_wrist_speeds
from pipeline.web_logging import write_event, write_blocked

# ============================================================
# CẤU HÌNH HỆ THỐNG (HẰNG SỐ RÕ RÀNG - TASK 1.2 & 2.2)
# ============================================================
RAW_DIR = "dataset/raw"
MODEL_PATH = "models/sign_language_model.h5"
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = "models/backup_round1.h5"
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = "models/backup_old_model.h5"

AUDIO_DIR = "audio"
DICTIONARY_PATH = "dictionary.json"

# Cấu hình truyền ảnh & MediaPipe (khớp chuẩn lúc train - Task 1.2)
SEND_WIDTH = 640
SEND_HEIGHT = 480
JPEG_QUALITY = 0.7
MP_COMPLEXITY = 1

# Cấu hình thời gian & độ trễ (Task 2.2)
TARGET_DURATION = 3.2       # Độ dài cửa sổ chuẩn hóa thời gian (giây) theo benchmark thu thập (3.184s)
SEQUENCE_LENGTH = 30        # Số frame đầu vào của model LSTM
MIN_FILL_RATIO = 0.9        # Tỉ lệ tối thiểu của buffer (3.2s * 0.9 = 2.88s) để kích hoạt predict
COOLDOWN_SEC = 1.0          # Thời gian nghỉ sau nhận diện (giây) - Đặt mặc định 1.0s
THRESHOLD = 0.6             # Ngưỡng tự tin tối thiểu
SMOOTH_WINDOW = 3           # Số lần predict ổn định liên tiếp để xác nhận từ
PREDICT_INTERVAL_SEC = 0.05 # Khoảng cách tối thiểu giữa các lần predict (giây)

# Cổng chuyển động (Motion Gate - Task 3.1 & Task D1-D2 Vòng 4)
GATE_ENABLED = True         # Mặc định BẬT theo kết quả thực nghiệm A/B (29/38 đúng vs 16/38, triệt tiêu nhãn thừa lúc nghỉ)
GATE_MIN_HAND_RATIO = 0.50   # Tỉ lệ frame có tay >= 50%
GATE_MOTION_STAT = "max"     # "max" | "median" (Task D2, mặc định "max")
GATE_MIN_MOTION_MAX = 0.05   # Vận tốc cổ tay cực đại tối thiểu (nguỡng từ Vòng 3)
GATE_MIN_MOTION_MEDIAN = 0.025 # Ngưỡng trung vị tốc độ cổ tay (phân tích gesture span vòng 3)
GATE_MIN_MOTION = GATE_MIN_MOTION_MAX if GATE_MOTION_STAT == "max" else GATE_MIN_MOTION_MEDIAN
STATIC_CLASSES = [
    'i', 'h', 'd', 'l', 'k', 'x', 'u', 'e', 'v', 'dau_nga', 'y', 's'
]                            # 12 class tĩnh từ Task 1.2

# Cấu hình logging thực nghiệm (Task 2.3 & 3.2 & Task A2/D1 Vòng 4)
ENABLE_CSV_LOG = True
CSV_LOG_PATH = os.path.join(os.path.dirname(__file__), "logs", "web_session_log.csv")
EVENTS_LOG_PATH = os.path.join(os.path.dirname(__file__), "logs", "web_events.csv")
GATE_BLOCKED_LOG_PATH = os.path.join(os.path.dirname(__file__), "logs", "web_gate_blocked.csv")

os.makedirs(AUDIO_DIR, exist_ok=True)
os.makedirs(os.path.dirname(CSV_LOG_PATH), exist_ok=True)

# Khởi tạo header cho file CSV nếu chưa có
if ENABLE_CSV_LOG and not os.path.exists(CSV_LOG_PATH):
    with open(CSV_LOG_PATH, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            "timestamp",
            "fps",
            "buffer_duration_s",
            "predict_time_ms",
            "top1_action",
            "confidence",
            "t_first_hand_to_output_s"
        ])

# ============================================================
# KHỞI TẠO FLASK + SOCKET.IO
# ============================================================
print(f"[INIT] Khởi động server lúc: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
app = Flask(__name__)
app.config['SECRET_KEY'] = 'sign-language-thesis-demo'
socketio = SocketIO(app, async_mode='threading', cors_allowed_origins='*')

# ============================================================
# LOAD DANH SÁCH TỪ VỰNG & TỪ ĐIỂN
# ============================================================
actions = get_actions(RAW_DIR)
print(f"[INIT] Đã nạp {len(actions)} từ vựng: {actions}")

try:
    with open(DICTIONARY_PATH, 'r', encoding='utf-8') as f:
        VIETNAMESE_DICT = json.load(f)
except FileNotFoundError:
    print("[INIT] Không tìm thấy dictionary.json, dùng tên gốc.")
    VIETNAMESE_DICT = {}

# Tự sinh audio tiếng Việt nếu chưa có
for action in actions:
    audio_path = os.path.join(AUDIO_DIR, f"voice_{action}.mp3")
    if not os.path.exists(audio_path):
        try:
            text = VIETNAMESE_DICT.get(action, action.replace('_', ' '))
            gTTS(text=text, lang='vi').save(audio_path)
            print(f"[INIT] Đã tạo audio cho '{action}'")
        except Exception as e:
            print(f"[INIT] Lỗi tạo audio '{action}': {e}")

# ============================================================
# LOAD VÀ TỐI ƯU HÓA MÔ HÌNH (TASK 2.1)
# ============================================================
print(f"[INIT] Đang load model từ {MODEL_PATH}...")
model = load_model(MODEL_PATH)
expected_dim = model.input_shape[-1]
print(f"[INIT] Model kỳ vọng đầu vào: {model.input_shape} ({expected_dim} chiều)")

# Xây dựng hàm tf.function được tối ưu hóa đồ thị tính toán
@tf.function(reduce_retracing=True)
def fast_predict(x):
    return model(x, training=False)

# Warm-up đồ thị tính toán ngay lúc khởi động
_dummy_x = tf.zeros((1, SEQUENCE_LENGTH, expected_dim), dtype=tf.float32)
_ = fast_predict(_dummy_x).numpy()
print("[INIT] ✅ Đã warm-up tf.function: Tốc độ suy luận đạt ~2.0ms (giảm 96.7% trễ so với model.predict).")

# ============================================================
# MEDIAPIPE HOLISTIC (TASK 1.2: KHỚP LÚC TRAIN)
# ============================================================
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(
    model_complexity=MP_COMPLEXITY, # Hoàn nguyên về 1
    refine_face_landmarks=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
holistic_lock = threading.Lock()

# ============================================================
# TRẠNG THÁI CLIENT
# ============================================================
client_state = {}


def new_state():
    return {
        'buffer': [],               # Lưu list các tuple (timestamp, raw_keypoints_258)
        'last_predict_time': 0.0,
        'cooldown_until': 0.0,      # Mốc thời gian kết thúc cooldown
        'current_action': "",
        'smoother': PredictionSmoother(window_size=SMOOTH_WINDOW, threshold=THRESHOLD),
        'fps_timer': time.time(),
        'fps_count': 0,
        'current_fps': 0.0,
        'total_frames': 0,
        'first_hand_time': None,    # Mốc thời gian bắt đầu thấy tay trong cử chỉ hiện tại
    }


def extract_keypoints(results):
    """Trích xuất 258 đặc trưng thô: Pose(132) + LH(63) + RH(63)"""
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


def landmarks_to_points(landmark_list):
    """Chuyển landmark sang list [x, y] gửi qua websocket"""
    if landmark_list is None:
        return None
    return [[round(p.x, 4), round(p.y, 4)] for p in landmark_list.landmark]


# ============================================================
# ROUTES
# ============================================================
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/audio/<path:filename>')
def serve_audio(filename):
    return send_from_directory(AUDIO_DIR, filename)


@app.route('/api/config')
def api_config():
    return jsonify({
        'actions': actions,
        'dictionary': {a: VIETNAMESE_DICT.get(a, a.replace('_', ' ')) for a in actions},
        'threshold': THRESHOLD,
        'sequence_length': SEQUENCE_LENGTH,
        'target_duration': TARGET_DURATION,
        'cooldown_sec': COOLDOWN_SEC,
        'send_width': SEND_WIDTH,
        'send_height': SEND_HEIGHT,
        'jpeg_quality': JPEG_QUALITY,
        'mp_complexity': MP_COMPLEXITY,
        'gate_enabled': GATE_ENABLED,
        'gate_min_hand_ratio': GATE_MIN_HAND_RATIO,
        'gate_motion_stat': GATE_MOTION_STAT,
        'gate_min_motion': GATE_MIN_MOTION,
        'static_classes': STATIC_CLASSES,
    })


# ============================================================
# SOCKET.IO EVENTS
# ============================================================
@socketio.on('connect')
def on_connect():
    from flask import request
    client_state[request.sid] = new_state()
    print(f"[CONNECT] Client {request.sid} đã kết nối.")


@socketio.on('disconnect')
def on_disconnect():
    from flask import request
    client_state.pop(request.sid, None)
    print(f"[DISCONNECT] Client {request.sid} đã ngắt kết nối.")


@socketio.on('video_frame')
def on_video_frame(data):
    """
    Nhận frame kèm timestamp, giải mã, chạy Holistic, resample theo thời gian và predict siêu tốc.
    """
    t_start = time.perf_counter()
    from flask import request
    sid = request.sid
    state = client_state.get(sid)
    if state is None:
        return

    client_ts = data.get('timestamp', time.time())

    # 1. Giải mã ảnh base64
    try:
        header, encoded = data['image'].split(',', 1)
        img_bytes = base64.b64decode(encoded)
        np_arr = np.frombuffer(img_bytes, dtype=np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    except Exception as e:
        print(f"[FRAME] Lỗi giải mã ảnh: {e}")
        socketio.emit('frame_ack', {'status': 'error'}, room=sid)
        return

    if frame is None:
        socketio.emit('frame_ack', {'status': 'error'}, room=sid)
        return

    # Lật ngang ảnh
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 2. Xử lý MediaPipe Holistic
    with holistic_lock:
        results = holistic.process(rgb)

    # Gửi ACK ngay lập tức để giải phóng client (Flow Control)
    socketio.emit('frame_ack', {'status': 'ok'}, room=sid)

    # Gửi landmark cho overlay
    socketio.emit('landmarks', {
        'pose': landmarks_to_points(results.pose_landmarks),
        'left_hand': landmarks_to_points(results.left_hand_landmarks),
        'right_hand': landmarks_to_points(results.right_hand_landmarks),
    }, room=sid)

    keypoints = extract_keypoints(results)
    has_hand = (results.left_hand_landmarks is not None or results.right_hand_landmarks is not None)

    # Đo độ trễ xử lý frame
    t_latency = (time.perf_counter() - t_start) * 1000.0

    # Tính FPS nhận thực tế phía server
    state['fps_count'] += 1
    state['total_frames'] += 1
    now_wall = time.time()
    if now_wall - state['fps_timer'] >= 1.0:
        state['current_fps'] = state['fps_count'] / (now_wall - state['fps_timer'])
        state['fps_count'] = 0
        state['fps_timer'] = now_wall

    # Ghi nhận thời điểm xuất hiện tay đầu tiên (Task 2.3)
    if has_hand and state['first_hand_time'] is None:
        state['first_hand_time'] = now_wall

    # 3. Cooldown theo THỜI GIAN
    if now_wall < state['cooldown_until']:
        cooldown_left = state['cooldown_until'] - now_wall
        socketio.emit('buffer_progress', {
            'progress': 0,
            'cooldown': True,
            'cooldown_left': round(cooldown_left, 1),
            'has_hand': has_hand,
        }, room=sid)
        return

    # 4. Thêm frame vào buffer theo thời gian
    state['buffer'].append((client_ts, keypoints))

    # Cắt tỉa buffer: chỉ giữ lại các frame trong cửa sổ thời gian gần nhất (hệ số 1.5x)
    min_keep_time = client_ts - (TARGET_DURATION * 1.5)
    state['buffer'] = [(t, f) for (t, f) in state['buffer'] if t >= min_keep_time]

    buffer_duration = state['buffer'][-1][0] - state['buffer'][0][0] if len(state['buffer']) >= 2 else 0.0
    progress = min(1.0, buffer_duration / TARGET_DURATION)

    socketio.emit('buffer_progress', {
        'progress': progress,
        'cooldown': False,
        'has_hand': has_hand,
    }, room=sid)

    # In log định kỳ mỗi 20 frames
    if state['total_frames'] % 20 == 0:
        print(f"[SERVER LOG] FPS nhận: {state['current_fps']:4.1f} | "
              f"Độ trễ frame: {t_latency:5.1f}ms | "
              f"Buffer: {buffer_duration:4.2f}s ({len(state['buffer'])} frames)")

    # 5. Kiểm tra điều kiện predict: đủ thời gian cửa sổ + có tay + giãn cách predict
    if (buffer_duration >= TARGET_DURATION * MIN_FILL_RATIO and has_hand
            and (now_wall - state['last_predict_time'] >= PREDICT_INTERVAL_SEC)):

        state['last_predict_time'] = now_wall

        timestamps = [t for t, f in state['buffer']]
        raw_frames = [f for t, f in state['buffer']]

        # LẤY MẪU LẠI (Temporal Resampling) trên đặc trưng thô 258 chiều
        resampled_258 = resample_raw_sequence(
            timestamps, raw_frames,
            target_duration=TARGET_DURATION,
            target_length=SEQUENCE_LENGTH
        )

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

            # Chuẩn hóa 258 -> 384
            norm_seq = build_features(resampled_258)
            norm_tensor = tf.convert_to_tensor(np.expand_dims(norm_seq, axis=0), dtype=tf.float32)

            # Suy luận siêu tốc qua tf.function (Task 2.1)
            t_pred_start = time.perf_counter()
            pred_tensor = fast_predict(norm_tensor)
            pred_res = pred_tensor.numpy()[0]
            pred_time_ms = (time.perf_counter() - t_pred_start) * 1000.0

            # Làm mượt dự đoán
            accepted, top_idx, top_conf, avg_probs = state['smoother'].update(pred_res)

            probs = {actions[i]: float(avg_probs[i]) for i in range(len(actions))}
            socketio.emit('probabilities', {'probs': probs}, room=sid)

            t_first_hand_to_output = None
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
                    # Bị chặn bởi cổng chuyển động -> ghi log chặn (Task D1) và reset smoother
                    write_blocked(
                        GATE_BLOCKED_LOG_PATH,
                        would_be_action=predicted_action,
                        confidence=top_conf,
                        gate_state=gate_state,
                        hand_ratio=hand_ratio,
                        motion_energy=motion_energy
                    )
                    state['smoother'].reset()
                else:
                    if state['first_hand_time'] is not None:
                        t_first_hand_to_output = round(now_wall - state['first_hand_time'], 3)

                    if predicted_action != state['current_action']:
                        state['current_action'] = predicted_action
                        state['buffer'] = []
                        state['smoother'].reset()
                        state['cooldown_until'] = now_wall + COOLDOWN_SEC
                        state['first_hand_time'] = None

                        print(f"👉 [PREDICT] '{predicted_action}' ({top_conf*100:.1f}%) | "
                              f"Gate: {gate_state} | "
                              f"Predict time: {pred_time_ms:.2f}ms | "
                              f"t_first_hand_to_output: {t_first_hand_to_output}s | "
                              f"Server FPS: {state['current_fps']:.1f}")

                        socketio.emit('prediction', {
                            'action': predicted_action,
                            'text': VIETNAMESE_DICT.get(predicted_action, predicted_action),
                            'confidence': top_conf,
                            'audio_url': f"/audio/voice_{predicted_action}.mp3",
                            'gate_state': gate_state
                        }, room=sid)

                        # Ghi nhận sự kiện vào logs/web_events.csv (Task A2)
                        write_event(
                            EVENTS_LOG_PATH,
                            action=predicted_action,
                            confidence=top_conf,
                            t_first_hand_to_output_s=t_first_hand_to_output,
                            gate_state=gate_state
                        )

            # Ghi log CSV chi tiết cho phiên làm việc (Task 2.3)
            if ENABLE_CSV_LOG:
                try:
                    with open(CSV_LOG_PATH, 'a', newline='', encoding='utf-8') as f_csv:
                        writer = csv.writer(f_csv)
                        writer.writerow([
                            round(now_wall, 3),
                            round(state['current_fps'], 2),
                            round(buffer_duration, 3),
                            round(pred_time_ms, 2),
                            actions[top_idx],
                            round(top_conf, 4),
                            t_first_hand_to_output if t_first_hand_to_output is not None else ""
                        ])
                except Exception:
                    pass


# ============================================================
# CHẠY SERVER
# ============================================================
if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("  SERVER ĐANG CHẠY: http://127.0.0.1:5000")
    print("=" * 60 + "\n")
    socketio.run(app, host='0.0.0.0', port=5000, debug=False, use_reloader=False)
