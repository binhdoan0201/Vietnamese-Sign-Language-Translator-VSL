"""
temporal_utils.py
=================
Các tiện ích nội suy thời gian (resampling) và làm mượt dự đoán cho
nhận diện ngôn ngữ ký hiệu theo thời gian thực (web demo & desktop).

Tuân thủ:
- Quy định 4: Nội suy trên đặc trưng THÔ 258 chiều (TRƯỚC build_features).
- Quy định 5: Không nội suy giữa frame có tay và frame mất tay (toàn 0 giữ ý nghĩa vắng mặt).
"""

import numpy as np


def resample_raw_sequence(timestamps, raw_frames, target_duration, target_length=30):
    """
    Lấy mẫu lại chuỗi đặc trưng thô 258 chiều về đúng `target_length` (30) frame
    đều nhau trên cửa sổ thời gian `target_duration` (giây).

    Args:
        timestamps: list/array các mốc thời gian (giây), độ dài M >= 2.
        raw_frames: list/array các frame đặc trưng thô (M, 258).
        target_duration: độ dài cửa sổ thời gian chuẩn (vd: 3.2 giây).
        target_length: số frame đầu ra (mặc định: 30).

    Returns:
        np.ndarray shape (target_length, 258) hoặc None nếu không đủ dữ liệu.
    """
    if len(timestamps) < 2 or len(raw_frames) < 2:
        return None

    times = np.asarray(timestamps, dtype=np.float64)
    frames = np.asarray(raw_frames, dtype=np.float32)

    t_end = times[-1]
    t_start = max(times[0], t_end - target_duration)

    # Nếu toàn bộ cửa sổ chưa đủ ít nhất 2 mốc
    if t_end <= t_start:
        return None

    # Lưới thời gian mục tiêu: target_length điểm cách đều trong [t_start, t_end]
    target_times = np.linspace(t_start, t_end, target_length)
    resampled = np.zeros((target_length, 258), dtype=np.float32)

    for k, tau in enumerate(target_times):
        # Tìm đoạn [times[idx], times[idx+1]] chứa tau
        idx = np.searchsorted(times, tau, side='right') - 1
        idx = np.clip(idx, 0, len(times) - 2)

        t_a = times[idx]
        t_b = times[idx + 1]
        dt = t_b - t_a

        if dt > 1e-6:
            alpha = float(np.clip((tau - t_a) / dt, 0.0, 1.0))
        else:
            alpha = 0.0

        f_a = frames[idx]
        f_b = frames[idx + 1]

        # 1. Pose (0:132): Nội suy tuyến tính bình thường
        resampled[k, 0:132] = (1.0 - alpha) * f_a[0:132] + alpha * f_b[0:132]

        # 2. Left hand (132:195):
        # Tuân thủ Quy định 5: Frame mất tay (toàn 0) giữ ý nghĩa vắng mặt.
        # Không nội suy giữa có tay và mất tay, dùng nearest frame.
        lh_a = f_a[132:195]
        lh_b = f_b[132:195]
        has_lh_a = np.count_nonzero(lh_a) > 0
        has_lh_b = np.count_nonzero(lh_b) > 0

        if has_lh_a and has_lh_b:
            resampled[k, 132:195] = (1.0 - alpha) * lh_a + alpha * lh_b
        elif (not has_lh_a) and (not has_lh_b):
            resampled[k, 132:195] = 0.0
        elif has_lh_a and (not has_lh_b):
            resampled[k, 132:195] = lh_a.copy() if alpha < 0.5 else 0.0
        else:  # not has_lh_a and has_lh_b
            resampled[k, 132:195] = 0.0 if alpha < 0.5 else lh_b.copy()

        # 3. Right hand (195:258): tương tự left hand
        rh_a = f_a[195:258]
        rh_b = f_b[195:258]
        has_rh_a = np.count_nonzero(rh_a) > 0
        has_rh_b = np.count_nonzero(rh_b) > 0

        if has_rh_a and has_rh_b:
            resampled[k, 195:258] = (1.0 - alpha) * rh_a + alpha * rh_b
        elif (not has_rh_a) and (not has_rh_b):
            resampled[k, 195:258] = 0.0
        elif has_rh_a and (not has_rh_b):
            resampled[k, 195:258] = rh_a.copy() if alpha < 0.5 else 0.0
        else:
            resampled[k, 195:258] = 0.0 if alpha < 0.5 else rh_b.copy()

    return resampled


class PredictionSmoother:
    """
    Bộ làm mượt kết quả dự đoán qua N lần predict liên tiếp:
    - Lưu xác suất của N lần gần nhất (Moving Average)
    - Chỉ chấp nhận khi top-1 ổn định và vượt ngưỡng THRESHOLD
    """
    def __init__(self, window_size=3, threshold=0.6):
        self.window_size = window_size
        self.threshold = threshold
        self.history = []

    def update(self, prob_vector):
        """
        prob_vector: array (num_classes,) xác suất softmax của 1 lần predict.
        Returns:
            (accepted, top_idx, avg_conf, avg_probs)
        """
        self.history.append(np.asarray(prob_vector, dtype=np.float32))
        if len(self.history) > self.window_size:
            self.history.pop(0)

        # Trung bình xác suất các lần gần nhất
        avg_probs = np.mean(self.history, axis=0)
        top_idx = int(np.argmax(avg_probs))
        top_conf = float(avg_probs[top_idx])

        # Kiểm tra xem top_idx có phải là class cao nhất trong TẤT CẢ các lần gần nhất không
        all_same_top = all(int(np.argmax(p)) == top_idx for p in self.history)

        accepted = (
            len(self.history) >= self.window_size
            and all_same_top
            and top_conf >= self.threshold
        )
        return accepted, top_idx, top_conf, avg_probs

    def reset(self):
        self.history.clear()
