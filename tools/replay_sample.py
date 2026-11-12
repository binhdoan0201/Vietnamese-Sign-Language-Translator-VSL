"""
tools/replay_sample.py
======================
Xem lại hoạt ảnh khung xương đã lưu trong file .npy (hỗ trợ cả chuẩn 1662 và 126).

CÁCH DÙNG:
  python tools/replay_sample.py [đường_dẫn_tới_file.npy]
  (Nếu không truyền đường dẫn, script sẽ tự động tìm và phát file mẫu đầu tiên tìm thấy)

ĐIỀU KHIỂN:
  SPACE : Tạm dừng / Tiếp tục phát
  R     : Xem lại từ đầu
  Q/ESC : Thoát
"""

import os
import sys
import cv2
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17)
]

POSE_CONNECTIONS = [
    (11, 12),  # vai - vai
    (11, 13), (13, 15),  # tay trái
    (12, 14), (14, 16),  # tay phải
    (11, 23), (12, 24), (23, 24)  # thân trên
]


def find_first_sample():
    raw_dir = os.path.join(PROJECT_ROOT, "dataset", "raw")
    for root, dirs, files in os.walk(raw_dir):
        for f in files:
            if f.endswith('.npy'):
                return os.path.join(root, f)
    return None


def draw_hand(canvas, hand_points_flat, width, height, color):
    if np.all(hand_points_flat == 0):
        return

    pts = []
    for i in range(21):
        x = hand_points_flat[i * 3]
        y = hand_points_flat[i * 3 + 1]
        px = int(x * width)
        py = int(y * height)
        pts.append((px, py))
        cv2.circle(canvas, (px, py), 4, color, -1)

    for start, end in HAND_CONNECTIONS:
        cv2.line(canvas, pts[start], pts[end], color, 2)


def draw_pose(canvas, pose_flat, width, height, color=(200, 200, 0)):
    if np.all(pose_flat == 0):
        return

    pts = {}
    for i in range(33):
        x = pose_flat[i * 4]
        y = pose_flat[i * 4 + 1]
        visibility = pose_flat[i * 4 + 3]
        if visibility > 0.3:
            px = int(x * width)
            py = int(y * height)
            pts[i] = (px, py)
            cv2.circle(canvas, (px, py), 3, color, -1)

    for start, end in POSE_CONNECTIONS:
        if start in pts and end in pts:
            cv2.line(canvas, pts[start], pts[end], color, 2)


def replay(file_path):
    if not os.path.exists(file_path):
        print(f"❌ Không tìm thấy file: {file_path}")
        return

    data = np.load(file_path)
    print(f"📂 Đang mở file: {file_path}")
    print(f"📊 Kích thước dữ liệu: {data.shape}")

    total_frames = len(data)
    width, height = 800, 600
    paused = False
    current_frame = 0

    label_text = os.path.basename(os.path.dirname(os.path.dirname(file_path)))
    session_text = os.path.basename(os.path.dirname(file_path))
    sample_text = os.path.basename(file_path)

    while True:
        frame_data = data[current_frame]
        canvas = np.zeros((height, width, 3), dtype=np.uint8)

        if frame_data.shape[0] == 1662:
            # 1662 = pose(132) + face(1404) + lh(63) + rh(63)
            pose_data = frame_data[:132]
            lh_data = frame_data[1536:1536 + 63]
            rh_data = frame_data[1599:1662]
            draw_pose(canvas, pose_data, width, height)
            draw_hand(canvas, lh_data, width, height, color=(0, 255, 0))    # Tay trái: Xanh lá
            draw_hand(canvas, rh_data, width, height, color=(0, 165, 255))  # Tay phải: Cam
        elif frame_data.shape[0] == 126:
            lh_data = frame_data[:63]
            rh_data = frame_data[63:126]
            draw_hand(canvas, lh_data, width, height, color=(0, 255, 0))
            draw_hand(canvas, rh_data, width, height, color=(0, 165, 255))

        # Hiển thị thông tin
        cv2.putText(canvas, f"Label: {label_text} | {session_text} | {sample_text}",
                    (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        cv2.putText(canvas, f"Frame: {current_frame + 1}/{total_frames}  {'[PAUSED]' if paused else ''}",
                    (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
        cv2.putText(canvas, "SPACE: Pause | R: Replay | Q: Thoat",
                    (20, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)

        cv2.imshow("Sign Language Replay", canvas)

        wait_time = 0 if paused else 40
        key = cv2.waitKey(wait_time) & 0xFF

        if key == ord('q') or key == 27:
            break
        elif key == ord(' '):
            paused = not paused
        elif key == ord('r'):
            current_frame = 0
            continue

        if not paused:
            current_frame += 1
            if current_frame >= total_frames:
                current_frame = 0

    cv2.destroyAllWindows()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        path = find_first_sample()
        if not path:
            print("❌ Không tìm thấy bất kỳ file .npy nào trong dataset/raw/")
            sys.exit(1)
        print(f"💡 Tự động chọn mẫu: {path}")

    replay(path)
