"""
pipeline/data_collector.py
==========================
Bộ thu thập dữ liệu ngôn ngữ ký hiệu quy mô lớn (MediaPipe Holistic, 1662 keypoints).
Hỗ trợ quay theo block (1 từ) hoặc xen kẽ (nhiều từ luân phiên), theo dõi tiến độ theo session.

CÁCH DÙNG:
  python pipeline/data_collector.py
  -> chọn session (1-4)
  -> gõ tên 1 từ để quay theo BLOCK (vd: xin_chao)
  -> gõ 2 tên cách nhau dấu phẩy để quay XEN KẼ (vd: a,aw)
  -> trong cửa sổ camera: SPACE = quay 1 mẫu | N = đổi từ | Q = thoát
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import cv2
import mediapipe as mp
import numpy as np
import json

# ============================================================
# CẤU HÌNH ĐƯỜNG DẪN & THÔNG SỐ
# ============================================================
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATASET_DIR = os.path.join(PROJECT_ROOT, "dataset", "raw")
DICTIONARY_PATH = os.path.join(PROJECT_ROOT, "dictionary.json")

SEQUENCE_LENGTH = 30
TARGET_SESSIONS = 4
SAMPLES_PER_SESSION = 30

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# ============================================================
# 4 KỊCH BẢN MẶC ĐỊNH
# ============================================================
SCENARIOS = {
    "session_01": {
        "ten": "Baseline",
        "vi_tri": "Chính diện, cách camera ~1m",
        "anh_sang": "Đèn phòng giảm cường độ / có rèm che bớt (tối hơn session_03)",
        "nguoi_ky": "Bản thân",
    },
    "session_02": {
        "ten": "Đổi vị trí + trang phục",
        "vi_tri": "Lệch trái ~15-20°, cách camera ~1.2m",
        "anh_sang": "Ánh sáng tự nhiên ban ngày (gần cửa sổ)",
        "nguoi_ky": "Bản thân (hoặc người thứ 2 nếu có)",
    },
    "session_03": {
        "ten": "Đổi ánh sáng + khoảng cách",
        "vi_tri": "Chính diện, cách gần hơn ~0.7m",
        "anh_sang": "Đèn phòng (ánh sáng nhân tạo ổn định)",
        "nguoi_ky": "Ưu tiên người ký MỚI nếu có (đây là session tốt nhất để mời người khác)",
    },
    "session_04": {
        "ten": "Điều kiện khó (stress test)",
        "vi_tri": "Lệch phải ~15-20°, cách camera ~1.2m",
        "anh_sang": "Ngược sáng nhẹ (nguồn sáng bên hông phải, vd cửa sổ sau)",
        "nguoi_ky": "Người khác (nếu có) hoặc bản thân trong điều kiện khó",
    },
}


def print_scenario(session):
    s = SCENARIOS.get(session)
    if not s:
        return
    print(f"\n┌─ KỊCH BẢN {session} — {s['ten']} " + "─" * 20)
    print(f"│ Vị trí     : {s['vi_tri']}")
    print(f"│ Ánh sáng   : {s['anh_sang']}")
    print(f"│ Người ký   : {s['nguoi_ky']}")
    print("└" + "─" * 60)
    print("  (Đổi thông số ở dict SCENARIOS trong code nếu điều kiện thực tế khác)")


# ============================================================
# TRÍCH XUẤT ĐẶC TRƯNG THÔ (1662 điểm)
# ============================================================
def extract_keypoints(results):
    face = np.array([[r.x, r.y, r.z] for r in results.face_landmarks.landmark]).flatten() \
        if results.face_landmarks else np.zeros(1404)
    pose = np.array([[r.x, r.y, r.z, r.visibility] for r in results.pose_landmarks.landmark]).flatten() \
        if results.pose_landmarks else np.zeros(132)
    lh = np.array([[r.x, r.y, r.z] for r in results.left_hand_landmarks.landmark]).flatten() \
        if results.left_hand_landmarks else np.zeros(63)
    rh = np.array([[r.x, r.y, r.z] for r in results.right_hand_landmarks.landmark]).flatten() \
        if results.right_hand_landmarks else np.zeros(63)
    return np.concatenate([pose, face, lh, rh])


# ============================================================
# TIỆN ÍCH: DANH SÁCH TỪ VỰNG & TIẾN ĐỘ
# ============================================================
def load_labels():
    if not os.path.exists(DICTIONARY_PATH):
        print(f"[!] Không tìm thấy {DICTIONARY_PATH}. Tạo file này với "
              f"nội dung dạng {{\"label\": \"Nghĩa tiếng Việt\", ...}}")
        exit(1)
    with open(DICTIONARY_PATH, 'r', encoding='utf-8') as f:
        return json.load(f)   # dict: label -> nghĩa tiếng Việt


def count_samples(label, session):
    path = os.path.join(DATASET_DIR, label, session)
    if not os.path.isdir(path):
        return 0
    return len([f for f in os.listdir(path) if f.endswith('.npy')])


def next_sample_number(save_dir):
    nums = []
    for f in os.listdir(save_dir):
        if f.startswith("sample_") and f.endswith(".npy"):
            nums.append(int(f[7:10]))
    return max(nums, default=0) + 1


def print_progress(labels, sessions, highlight_session=None):
    print("\n" + "=" * 78)
    header = f"{'TỪ':<16}" + "".join(f"{s.replace('session_', 'S'):>9}" for s in sessions)
    print(header)
    print("-" * 78)
    total_done, total_target = 0, 0
    for label in labels:
        row = f"{label:<16}"
        for s in sessions:
            c = count_samples(label, s)
            total_done += min(c, SAMPLES_PER_SESSION)
            total_target += SAMPLES_PER_SESSION
            mark = "✓" if c >= SAMPLES_PER_SESSION else " "
            cell = f"{c:>2}/{SAMPLES_PER_SESSION}{mark}"
            row += f"{cell:>9}"
        print(row)
    print("-" * 78)
    pct = (total_done / total_target * 100) if total_target else 0
    print(f"Tổng tiến độ: {total_done}/{total_target} sample ({pct:.1f}%)")
    print("=" * 78)


# ============================================================
# GHI 1 SAMPLE (countdown + chờ tay + quay 30 frame + lưu)
# ============================================================
def record_one_sample(cap, holistic, label, session, quick=False):
    save_dir = os.path.join(DATASET_DIR, label, session)
    os.makedirs(save_dir, exist_ok=True)
    sample_number = next_sample_number(save_dir)

    win = "Data Collector V3"

    # ----- Countdown: đầy đủ 5s lần đầu, rút gọn 2s khi quay liên tiếp -----
    countdown_seq = [2, 1] if quick else [5, 4, 3, 2, 1]
    for c in countdown_seq:
        ret, frame = cap.read()
        if not ret:
            return False
        frame = cv2.flip(frame, 1)
        cv2.putText(frame, str(c), (400, 300), cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 0, 255), 8)
        cv2.putText(frame, f"Chuan bi: {label}  (mau #{sample_number})",
                    (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imshow(win, frame)
        cv2.waitKey(1000)

    # ----- Chờ phát hiện tay -----
    while True:
        ret, frame = cap.read()
        if not ret:
            return False
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(rgb)

        cv2.putText(frame, "Dua tay vao khung hinh...", (30, 200),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        cv2.imshow(win, frame)

        if results.left_hand_landmarks or results.right_hand_landmarks:
            break
        if cv2.waitKey(1) & 0xFF == ord('q'):
            return False

    # ----- Quay 30 frame -----
    sequence = []
    for i in range(SEQUENCE_LENGTH):
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(rgb)

        frame_data = extract_keypoints(results)
        if len(frame_data) != 1662:
            print(f"[!] Lỗi kích thước đặc trưng: {len(frame_data)}")
            break
        sequence.append(frame_data)

        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
        mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

        cv2.putText(frame, f"REC {i+1}/{SEQUENCE_LENGTH}  [{label}]",
                    (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow(win, frame)
        cv2.waitKey(33)

    if len(sequence) == SEQUENCE_LENGTH:
        arr = np.array(sequence, dtype=np.float32)
        path = os.path.join(save_dir, f"sample_{sample_number:03d}.npy")
        np.save(path, arr)
        print(f"  ✓ Đã lưu {path}  (shape {arr.shape})")
        return True
    else:
        print("  ✗ Sample không hợp lệ, bỏ qua.")
        return False


# ============================================================
# VÒNG LẶP QUAY: BLOCK (1 từ) HOẶC XEN KẼ (nhiều từ luân phiên)
# ============================================================
def recording_loop(cap, holistic, session, label_cycle):
    idx = 0
    is_first = True
    win = "Data Collector V3"

    mode_name = "XEN KE" if len(label_cycle) > 1 else "BLOCK"
    print(f"\n[Chế độ {mode_name}] Từ: {', '.join(label_cycle)}")
    print("  SPACE = quay 1 mẫu | N = đổi từ khác | Q = thoát chương trình\n")

    while True:
        label = label_cycle[idx % len(label_cycle)]
        current_count = count_samples(label, session)

        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        cv2.putText(frame, f"[{mode_name}] {label}  ({current_count}/{SAMPLES_PER_SESSION})",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.putText(frame, "SPACE=quay | N=doi tu | Q=thoat",
                    (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        scenario = SCENARIOS.get(session)
        if scenario:
            cv2.putText(frame, f"{session}: {scenario['ten']}",
                        (20, frame.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (0, 200, 255), 1)
        cv2.imshow(win, frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            return 'QUIT'
        if key == ord('n'):
            return 'MENU'
        if key == 32:  # SPACE
            success = record_one_sample(cap, holistic, label, session, quick=not is_first)
            is_first = False
            if success and len(label_cycle) > 1:
                idx += 1


# ============================================================
# MENU CHỌN SESSION / TỪ VỰNG
# ============================================================
def choose_session(sessions):
    print("\nChọn session:")
    for i, s in enumerate(sessions, 1):
        print(f"  {i}. {s}")
    raw = input(">> Nhập số hoặc tên session: ").strip()
    if raw.isdigit() and 1 <= int(raw) <= len(sessions):
        return sessions[int(raw) - 1]
    if raw in sessions:
        return raw
    return None


def choose_labels(labels):
    print("\nGõ TÊN từ (hoặc số thứ tự) để quay theo BLOCK.")
    print("Gõ 2+ tên cách nhau dấu phẩy để quay XEN KẼ (vd: a,aw).")
    print("Gõ 'list' để xem lại bảng tiến độ. Gõ 'q' để thoát.")
    raw = input(">> ").strip()

    if raw.lower() == 'q':
        return 'QUIT'
    if raw.lower() == 'list':
        return 'LIST'

    parts = [p.strip() for p in raw.split(',') if p.strip()]
    resolved = []
    for p in parts:
        if p.isdigit() and 1 <= int(p) <= len(labels):
            resolved.append(labels[int(p) - 1])
        elif p in labels:
            resolved.append(p)
        else:
            print(f"  [!] Không tìm thấy nhãn '{p}', bỏ qua.")
    return resolved if resolved else None


# ============================================================
# MAIN
# ============================================================
def main():
    labels_dict = load_labels()
    labels = list(labels_dict.keys())
    sessions = [f"session_{i:02d}" for i in range(1, TARGET_SESSIONS + 1)]

    print("=" * 78)
    print("  SIGN LANGUAGE DATA COLLECTOR V3")
    print(f"  {len(labels)} từ vựng | {TARGET_SESSIONS} session | "
          f"{SAMPLES_PER_SESSION} sample/session")
    print("=" * 78)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Không mở được camera!")
        return

    current_session = None

    with mp_holistic.Holistic(min_detection_confidence=0.5,
                              min_tracking_confidence=0.5) as holistic:
        while True:
            if current_session is None:
                print_progress(labels, sessions)
                current_session = choose_session(sessions)
                if current_session is None:
                    print("[!] Lựa chọn session không hợp lệ.")
                    continue
                print_scenario(current_session)

            print_progress(labels, sessions)
            print(f"\n[Session hiện tại: {current_session}]")
            choice = choose_labels(labels)

            if choice == 'QUIT':
                break
            if choice == 'LIST':
                continue
            if choice is None:
                continue

            result = recording_loop(cap, holistic, current_session, choice)
            if result == 'QUIT':
                break

    cap.release()
    cv2.destroyAllWindows()
    print("\nĐã thoát chương trình thu thập dữ liệu.")


if __name__ == "__main__":
    main()
