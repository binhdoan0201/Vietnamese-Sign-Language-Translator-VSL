"""Module tiền xử lý cắt đoạn hoạt động (Gesture Segmentation) trên dữ liệu thô 258 chiều.

Triển khai 2 chiến lược:
  - S-A (hand): Cắt theo hand_present_span (+ đệm 3 frame mỗi bên) rồi resample về 30 frame.
  - S-B (motion): Cắt theo motion_span (+ đệm 3 frame mỗi bên) rồi resample;
                  nếu cử chỉ tĩnh / không chuyển động thì fallback về S-A.

Tuân thủ nghiêm ngặt:
  - Làm trên 258 chiều thô TRƯỚC build_features.
  - Ngưỡng cố định, không học từ test/val.
  - Bảo toàn frame vắng tay (dùng resample_raw_sequence).
  - Nếu đoạn cắt < 8 frame thì giữ nguyên cả cửa sổ 30 frame và ghi nhận fallback.
"""

import numpy as np
from temporal_utils import resample_raw_sequence

PAD_FRAMES = 3
MIN_SPAN_FRAMES = 8
TARGET_DURATION = 3.2
TARGET_LENGTH = 30
DEFAULT_MOTION_RATIO = 0.20


def is_hand_present(raw_frame_258):
    """Kiểm tra frame có ít nhất 1 bàn tay xuất hiện hay không."""
    lh_present = np.count_nonzero(raw_frame_258[132:195]) > 0
    rh_present = np.count_nonzero(raw_frame_258[195:258]) > 0
    return lh_present or rh_present


def get_hand_present_span(sequence_258):
    """
    Xác định khoảng frame đầu - cuối có ít nhất 1 tay xuất hiện.
    Returns:
        (start_idx, end_idx) hoặc None nếu cả chuỗi không có tay.
    """
    present_indices = [t for t in range(len(sequence_258)) if is_hand_present(sequence_258[t])]
    if not present_indices:
        return None
    return present_indices[0], present_indices[-1]


def get_wrist_speeds(sequence_258):
    """
    Tính tốc độ di chuyển của cổ tay sau khi chuẩn hóa theo độ rộng 2 vai,
    áp dụng làm mượt cửa sổ 3 frame (moving average).
    
    Returns:
        v_smooth: mảng (T,) chứa tốc độ cổ tay tại từng frame.
        scale_valid: bool thể hiện pose có hợp lệ hay không.
    """
    seq = np.asarray(sequence_258, dtype=np.float32)
    T = len(seq)
    speeds = np.zeros(T, dtype=np.float32)

    # Lấy tọa độ cổ tay chuẩn hóa theo trung điểm và khoảng cách vai
    wrist_coords = np.zeros((T, 3), dtype=np.float32)

    for i in range(T):
        frame = seq[i]
        pose = frame[0:132].reshape(33, 4)
        lh = frame[132:195].reshape(21, 3)
        rh = frame[195:258].reshape(21, 3)

        left_shoulder = pose[11, :3]
        right_shoulder = pose[12, :3]
        scale = np.linalg.norm(left_shoulder - right_shoulder)

        # Chọn tọa độ cổ tay ưu tiên tay detect được, nếu không lấy pose wrist (15: lh, 16: rh)
        lh_present = np.count_nonzero(frame[132:195]) > 0
        rh_present = np.count_nonzero(frame[195:258]) > 0

        if rh_present:
            wrist = rh[0].copy()
        elif lh_present:
            wrist = lh[0].copy()
        else:
            # Fallback sang pose wrist
            wrist = pose[16, :3].copy() if pose[16, 3] >= pose[15, 3] else pose[15, :3].copy()

        if scale > 1e-3:
            center = (left_shoulder + right_shoulder) / 2.0
            wrist = (wrist - center) / scale

        wrist_coords[i] = wrist

    # Tính độ biến thiên khoảng cách delta_p
    for i in range(1, T):
        speeds[i] = np.linalg.norm(wrist_coords[i] - wrist_coords[i - 1])

    # Làm mượt moving average cửa sổ 3 frame
    v_smooth = np.zeros(T, dtype=np.float32)
    for i in range(T):
        i_start = max(0, i - 1)
        i_end = min(T, i + 2)
        v_smooth[i] = np.mean(speeds[i_start:i_end])

    return v_smooth


def get_motion_span(sequence_258, ratio=DEFAULT_MOTION_RATIO):
    """
    Xác định khoảng frame có vận tốc cổ tay >= ratio * max_speed.
    
    Returns:
        (start_idx, end_idx) hoặc None nếu cử chỉ tĩnh (max_speed quá nhỏ).
    """
    v_smooth = get_wrist_speeds(sequence_258)
    max_v = float(np.max(v_smooth))
    if max_v < 1e-4:
        return None

    threshold = ratio * max_v
    active_indices = [t for t in range(len(v_smooth)) if v_smooth[t] >= threshold]
    if not active_indices:
        return None
    return active_indices[0], active_indices[-1]


def segment_raw_sample(sequence_258, mode='none', motion_ratio=DEFAULT_MOTION_RATIO,
                       pad=PAD_FRAMES, min_span=MIN_SPAN_FRAMES, target_duration=TARGET_DURATION):
    """
    Cắt đoạn hoạt động và lấy mẫu lại về đúng 30 frame.

    Args:
        sequence_258: chuỗi thô shape (T, 258)
        mode: 'none', 'hand', 'motion'
        motion_ratio: ngưỡng tỉ lệ vận tốc (mặc định 0.20)
        pad: số frame đệm mỗi bên (mặc định 3)
        min_span: số frame tối thiểu sau đệm, nếu < min_span thì fallback giữ full
        target_duration: thời lượng tương ứng với 30 frame (3.2s)

    Returns:
        resampled_258: chuỗi (30, 258)
        was_fallback: bool báo hiệu có bị fallback hay không
    """
    seq = np.asarray(sequence_258, dtype=np.float32)
    T = len(seq)
    if mode == 'none':
        return seq.copy(), False

    span = None
    if mode == 'hand':
        span = get_hand_present_span(seq)
    elif mode == 'motion':
        span = get_motion_span(seq, ratio=motion_ratio)
        if span is None:
            # Fallback sang S-A nếu cử chỉ tĩnh không phát hiện chuyển động
            span = get_hand_present_span(seq)

    if span is None:
        # Không tìm thấy bàn tay -> fallback giữ nguyên cả cửa sổ
        return seq.copy(), True

    start_idx, end_idx = span
    start_pad = max(0, start_idx - pad)
    end_pad = min(T - 1, end_idx + pad)
    span_len = end_pad - start_pad + 1

    if span_len < min_span:
        # Đoạn cắt quá ngắn -> fallback dùng cả cửa sổ 30 frame
        return seq.copy(), True

    # Cắt đoạn
    slice_raw = seq[start_pad:end_pad + 1]

    # Quy đổi timestamp giả lập từ 0 đến dt * (span_len - 1)
    dt = target_duration / T
    timestamps = np.linspace(start_pad * dt, end_pad * dt, span_len)
    slice_duration = float(timestamps[-1] - timestamps[0])

    if slice_duration < 1e-4:
        return seq.copy(), True

    resampled = resample_raw_sequence(
        timestamps=timestamps,
        raw_frames=slice_raw,
        target_duration=slice_duration,
        target_length=TARGET_LENGTH
    )

    if resampled is None:
        return seq.copy(), True

    return resampled.astype(np.float32), False


def segment_dataset(X_raw258, mode='none', motion_ratio=DEFAULT_MOTION_RATIO):
    """
    Cắt đoạn toàn bộ tập dữ liệu.
    Returns:
        X_segmented: (N, 30, 258)
        stats: dict thống kê fallback
    """
    if mode == 'none':
        return X_raw258.copy(), {'mode': 'none', 'fallback_count': 0, 'total': len(X_raw258)}

    N = len(X_raw258)
    X_out = np.zeros_like(X_raw258)
    fallback_count = 0

    for i in range(N):
        sample, fallback = segment_raw_sample(X_raw258[i], mode=mode, motion_ratio=motion_ratio)
        X_out[i] = sample
        if fallback:
            fallback_count += 1

    stats = {
        'mode': mode,
        'fallback_count': fallback_count,
        'total': N,
        'fallback_ratio': fallback_count / N
    }
    return X_out, stats
