"""
pipeline/web_logging.py
=======================
Module quản lý ghi log sự kiện dự đoán và sự kiện bị chặn bởi Motion Gate.
Phục vụ phân tích thực nghiệm và kiểm thử đầu-cuối (End-to-End).
"""

import os
import csv
import time
import traceback
from datetime import datetime

_first_event_error = True
_first_blocked_error = True
_blocked_timestamps = []

EVENT_CSV_HEADER = ["timestamp", "action", "confidence", "t_first_hand_to_output_s", "gate_state"]
BLOCKED_CSV_HEADER = ["timestamp", "would_be_action", "confidence", "gate_state", "hand_ratio", "motion_energy"]


def write_event(filepath, action, confidence, t_first_hand_to_output_s, gate_state, timestamp=None):
    """
    Ghi nhận một sự kiện dự đoán thành công vào file CSV.

    Parameters:
        filepath: Đường dẫn tới file web_events.csv.
        action: Tên lớp cử chỉ dự đoán.
        confidence: Độ tin cậy (float hoặc chuỗi).
        t_first_hand_to_output_s: Thời gian từ lúc phát hiện tay đến lúc xuất từ (s).
        gate_state: Trạng thái của cổng ('OPEN_STATIC', 'OPEN_DYNAMIC', 'BYPASS_DISABLED', ...).
        timestamp: Chuỗi ISO timestamp; nếu None sẽ lấy thời gian hiện tại.
    """
    global _first_event_error
    try:
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        file_exists = os.path.exists(filepath) and os.path.getsize(filepath) > 0

        if timestamp is None:
            timestamp = datetime.now().isoformat()

        conf_str = f"{float(confidence):.4f}" if isinstance(confidence, (int, float)) else str(confidence)
        t_lat_str = f"{float(t_first_hand_to_output_s):.3f}" if t_first_hand_to_output_s not in [None, ""] else ""

        with open(filepath, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(EVENT_CSV_HEADER)
            writer.writerow([timestamp, action, conf_str, t_lat_str, gate_state])
        return True
    except Exception as e:
        if _first_event_error:
            _first_event_error = False
            print("[ERROR] Lỗi nghiêm trọng khi ghi web_events.csv (traceback lần đầu):")
            traceback.print_exc()
        else:
            print(f"[WARN] Lỗi ghi web_events.csv: {e}")
        return False


def write_blocked(filepath, would_be_action, confidence, gate_state, hand_ratio, motion_energy,
                  timestamp=None, max_rate_per_sec=2):
    """
    Ghi nhận sự kiện một ứng viên bị Motion Gate chặn (Task D1).
    Có cơ chế giới hạn tần suất tối đa max_rate_per_sec (mặc định 2 dòng/giây).

    Parameters:
        filepath: Đường dẫn tới file web_gate_blocked.csv.
        would_be_action: Tên lớp cử chỉ mà bộ làm mượt đã chấp nhận nhưng bị gate chặn.
        confidence: Độ tin cậy.
        gate_state: Trạng thái đóng ('closed_no_hand' hoặc 'closed_low_motion').
        hand_ratio: Tỉ lệ frame có tay trong cửa sổ [0.0, 1.0].
        motion_energy: Năng lượng hoặc vận tốc chuyển động cổ tay.
        timestamp: Chuỗi ISO timestamp; nếu None sẽ lấy thời gian hiện tại.
        max_rate_per_sec: Tần suất tối đa cho phép ghi (dòng/giây).
    """
    global _first_blocked_error, _blocked_timestamps
    now_ts = time.time()
    # Lọc bỏ các timestamp quá 1.0 giây trước
    _blocked_timestamps = [t for t in _blocked_timestamps if now_ts - t < 1.0]
    if len(_blocked_timestamps) >= max_rate_per_sec:
        # Đã đạt giới hạn tần suất trong 1 giây qua
        return False

    try:
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        file_exists = os.path.exists(filepath) and os.path.getsize(filepath) > 0

        if timestamp is None:
            timestamp = datetime.now().isoformat()

        conf_str = f"{float(confidence):.4f}" if isinstance(confidence, (int, float)) else str(confidence)
        hr_str = f"{float(hand_ratio):.3f}" if isinstance(hand_ratio, (int, float)) else str(hand_ratio)
        me_str = f"{float(motion_energy):.4f}" if isinstance(motion_energy, (int, float)) else str(motion_energy)

        with open(filepath, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(BLOCKED_CSV_HEADER)
            writer.writerow([timestamp, would_be_action, conf_str, gate_state, hr_str, me_str])

        _blocked_timestamps.append(now_ts)
        return True
    except Exception as e:
        if _first_blocked_error:
            _first_blocked_error = False
            print("[ERROR] Lỗi nghiêm trọng khi ghi web_gate_blocked.csv (traceback lần đầu):")
            traceback.print_exc()
        else:
            print(f"[WARN] Lỗi ghi web_gate_blocked.csv: {e}")
        return False
