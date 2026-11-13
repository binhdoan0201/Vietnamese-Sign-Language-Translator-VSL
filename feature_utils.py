"""
feature_utils.py
================
Module CHUNG xử lý đặc trưng cho cả training và realtime.

QUAN TRỌNG: Cả preprocess_data.py và realtime_test.py đều phải import
từ file này. Nếu 2 bên xử lý khác nhau -> model train một kiểu, realtime
đưa vào một kiểu -> nhận diện sai hoàn toàn.

Pipeline đặc trưng:
    MediaPipe Holistic (1662)
        -> slice_features()  -> 258  (bỏ face, giữ pose + 2 tay)
        -> build_features()  -> 384  (thêm đặc trưng bàn tay cục bộ)
"""

import numpy as np

# ============================================================
# HẰNG SỐ
# ============================================================
RAW_DIM = 1662          # MediaPipe Holistic đầy đủ
FEATURE_DIM = 258       # pose(132) + lh(63) + rh(63)
FEATURE_DIM_FINAL = 384 # 258 + lh_local(63) + rh_local(63)

MIN_SCALE = 1e-3        # ngưỡng tối thiểu coi pose là đáng tin


# ============================================================
# BƯỚC 1: CẮT BỎ FACE MESH (1662 -> 258)
# ============================================================
def slice_features(sequence):
    """
    Input : (T, 1662) - thứ tự concat trong data_collector_v2.py là
                        [pose(132), face(1404), lh(63), rh(63)]
    Output: (T, 258)  - [pose(132), lh(63), rh(63)]

    Lý do bỏ face: 1404/1662 = 84% chiều dữ liệu là face mesh, hầu như
    không mang thông tin phân biệt từ vựng ký hiệu -> chủ yếu là nhiễu,
    rất nguy hiểm khi dataset nhỏ.
    """
    sequence = np.asarray(sequence, dtype=np.float32)

    pose = sequence[:, 0:132]
    # face = sequence[:, 132:1536]   # bỏ
    lh = sequence[:, 1536:1599]
    rh = sequence[:, 1599:1662]

    return np.concatenate([pose, lh, rh], axis=1)


# ============================================================
# BƯỚC 2: CHUẨN HÓA TOÀN CỤC (theo cơ thể)
# ============================================================
def normalize_sequence(sequence):
    """
    Input : (T, 258) tọa độ thô
    Output: (T, 258) đã chuẩn hóa theo trung điểm 2 vai

    Mục đích: bất biến với vị trí đứng và khoảng cách tới camera.
    Giữ thông tin "tay đang ở ĐÂU" so với cơ thể.

    An toàn với:
      - Buffer tái sử dụng ở realtime (dùng .copy(), không ghi đè mảng gốc)
      - Frame mất pose (fallback về frame hợp lệ gần nhất)
      - Tay không detect được (giữ nguyên 0, không bị trừ center)
    """
    seq = np.asarray(sequence, dtype=np.float32)
    normalized = np.zeros_like(seq)

    last_valid = None  # frame hợp lệ gần nhất

    for i in range(len(seq)):
        # .copy() để KHÔNG ghi đè lên buffer gốc
        frame = seq[i].copy()
        pose = frame[0:132].reshape(33, 4).copy()
        lh = frame[132:195].reshape(21, 3).copy()
        rh = frame[195:258].reshape(21, 3).copy()

        left_shoulder = pose[11, :3]
        right_shoulder = pose[12, :3]

        center = (left_shoulder + right_shoulder) / 2.0
        scale = np.linalg.norm(left_shoulder - right_shoulder)

        # ----- Pose không đáng tin (mất detect / vai trùng nhau) -----
        if scale < MIN_SCALE:
            if last_valid is not None:
                normalized[i] = last_valid   # giữ liên tục chuyển động
            else:
                normalized[i] = 0.0          # chưa có frame nào hợp lệ
            continue

        # ----- Chuẩn hóa -----
        pose[:, :3] = (pose[:, :3] - center) / scale
        lh = (lh - center) / scale
        rh = (rh - center) / scale

        # Tay không detect được -> ép về 0 để giữ ý nghĩa "vắng mặt"
        # (nếu không, phép trừ center sẽ biến 0 thành giá trị khác 0)
        if np.count_nonzero(frame[132:195]) == 0:
            lh[:] = 0.0
        if np.count_nonzero(frame[195:258]) == 0:
            rh[:] = 0.0

        normalized[i, 0:132] = pose.flatten()
        normalized[i, 132:195] = lh.flatten()
        normalized[i, 195:258] = rh.flatten()

        last_valid = normalized[i].copy()

    return normalized


# ============================================================
# BƯỚC 3: CHUẨN HÓA CỤC BỘ BÀN TAY (fix nhầm 'a' vs 'aw')
# ============================================================
def normalize_hand_local(hand_flat):
    """
    Input : (63,) tọa độ thô 1 bàn tay
    Output: (63,) chuẩn hóa theo CHÍNH bàn tay đó
            -> gốc = cổ tay, scale = kích thước bàn tay

    Mục đích: giữ thông tin "tay có HÌNH GÌ" (tư thế ngón tay),
    độc lập hoàn toàn với vị trí tay trong không gian.

    Lý do cần: khi chia tọa độ tay cho khoảng cách 2 vai (~0.3), khác biệt
    giữa các ngón tay (~0.01) bị nén nhỏ hơn ~30 lần so với biến thiên vị
    trí tay (~1.0) -> model gần như không "nhìn thấy" hình dạng ngón.
    """
    hand_flat = np.asarray(hand_flat, dtype=np.float32)

    if np.count_nonzero(hand_flat) == 0:
        return np.zeros(63, dtype=np.float32)

    hand = hand_flat.reshape(21, 3).copy()

    wrist = hand[0].copy()
    hand = hand - wrist  # gốc tọa độ = cổ tay

    # Scale = khoảng cách xa nhất từ cổ tay tới các điểm
    dists = np.linalg.norm(hand, axis=1)
    scale = dists.max()
    if scale < 1e-6:
        return np.zeros(63, dtype=np.float32)

    return (hand / scale).flatten()


# ============================================================
# BƯỚC 4: GHÉP ĐẶC TRƯNG CUỐI CÙNG (258 -> 384)
# ============================================================
def build_features(sequence):
    """
    Input : (T, 258) tọa độ thô (sau slice_features)
    Output: (T, 384)
            [pose_global(132), lh_global(63), rh_global(63),
             lh_local(63), rh_local(63)]

    Kết hợp 2 loại thông tin:
      - Global: tay ở ĐÂU  (phân biệt ký hiệu khác vị trí thực hiện)
      - Local : tay HÌNH GÌ (phân biệt ký hiệu khác tư thế ngón tay)
    """
    seq = np.asarray(sequence, dtype=np.float32)

    # Phần global
    global_part = normalize_sequence(seq)  # (T, 258)

    # Phần local
    local_part = np.zeros((len(seq), 126), dtype=np.float32)
    for i in range(len(seq)):
        local_part[i, 0:63] = normalize_hand_local(seq[i][132:195])
        local_part[i, 63:126] = normalize_hand_local(seq[i][195:258])

    return np.concatenate([global_part, local_part], axis=1)  # (T, 384)


# ============================================================
# TIỆN ÍCH: lấy danh sách class nhất quán giữa các file
# ============================================================
def get_actions(raw_dir="dataset/raw"):
    """
    Trả về danh sách class đã sort, chỉ lấy thư mục (bỏ file rác
    như .DS_Store, Thumbs.db...).

    QUAN TRỌNG: mọi file đều phải dùng hàm này để thứ tự class
    luôn khớp nhau giữa preprocess / train / realtime.

    Cơ chế dự phòng (fallback): Nếu raw_dir không tồn tại hoặc không chứa
    thư mục lớp nào (ví dụ khi chạy demo từ repo sạch chưa tải dữ liệu),
    hàm sẽ tự động đọc danh sách 43 lớp chuẩn từ models/classes.json.
    """
    import os
    import json

    if os.path.exists(raw_dir) and os.path.isdir(raw_dir):
        classes = sorted([
            d for d in os.listdir(raw_dir)
            if os.path.isdir(os.path.join(raw_dir, d))
        ])
        if classes:
            return classes

    # Dự phòng: đọc từ models/classes.json (tính từ vị trí file feature_utils.py)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    classes_json_path = os.path.join(base_dir, "models", "classes.json")
    if os.path.exists(classes_json_path):
        with open(classes_json_path, "r", encoding="utf-8") as f:
            classes = json.load(f)
        return sorted(classes)

    raise FileNotFoundError(
        f"Không tìm thấy thư mục dữ liệu tại '{raw_dir}' "
        f"và không tìm thấy file dự phòng tại '{classes_json_path}'."
    )
