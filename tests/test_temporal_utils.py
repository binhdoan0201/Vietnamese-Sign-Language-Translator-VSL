"""
tests/test_temporal_utils.py
============================
Bộ kiểm thử tính chất toán học của hàm resample_raw_sequence trong temporal_utils.py.

Các tính chất cần kiểm tra:
  1. Output shape đúng (30, 258).
  2. Chuỗi hằng số vẫn là hằng số sau khi resample.
  3. Chuỗi tuyến tính theo thời gian giữ nguyên dạng tuyến tính sau resample.
  4. Tay vắng mặt (toàn 0) giữ nguyên toàn 0, không bị nội suy với frame có tay (Quy định 5).
  5. Timestamp không đều (jitter) vẫn cho kết quả đều đặn theo thời gian.
  6. Đầu vào ngắn (< 2 frame) trả về None.

CÁCH CHẠY:
  python tests/test_temporal_utils.py
  # hoặc: pytest tests/test_temporal_utils.py
"""

import os
import sys
import numpy as np

# Thêm thư mục gốc vào sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from temporal_utils import resample_raw_sequence


def test_output_shape():
    """Kiểm tra shape đầu ra chuẩn (30, 258)"""
    ts = np.linspace(0.0, 3.2, 20)
    frames = np.random.randn(20, 258).astype(np.float32)
    res = resample_raw_sequence(ts, frames, target_duration=3.2, target_length=30)
    assert res is not None, "Kết quả không được là None"
    assert res.shape == (30, 258), f"Kỳ vọng shape (30, 258), nhận được {res.shape}"
    return True


def test_constant_sequence():
    """Chuỗi hằng số phải giữ nguyên giá trị hằng số ở mọi frame"""
    ts = np.linspace(10.0, 13.2, 25)
    const_val = np.random.randn(258).astype(np.float32)
    frames = np.tile(const_val, (25, 1))

    res = resample_raw_sequence(ts, frames, target_duration=3.2, target_length=30)
    assert res is not None
    max_err = np.max(np.abs(res - const_val))
    assert max_err < 1e-5, f"Lỗi chuỗi hằng số quá lớn: {max_err}"
    return True


def test_linear_sequence():
    """Chuỗi biến thiên tuyến tính theo thời gian phải giữ đúng hàm tuyến tính"""
    ts = np.linspace(0.0, 3.2, 35)
    # Giả lập pose (0:132) tăng dần từ 1.0 đến 4.2
    slope = 2.5
    intercept = 1.0
    frames = np.zeros((35, 258), dtype=np.float32)
    for i, t in enumerate(ts):
        frames[i, 0:132] = slope * t + intercept

    res = resample_raw_sequence(ts, frames, target_duration=3.2, target_length=30)
    assert res is not None

    target_ts = np.linspace(0.0, 3.2, 30)
    expected_pose = slope * target_ts[:, None] + intercept
    max_err = np.max(np.abs(res[:, 0:132] - expected_pose))
    assert max_err < 1e-4, f"Lỗi nội suy tuyến tính: {max_err}"
    return True


def test_absent_hand_preservation():
    """
    Quy định 5: Frame tay không detect (toàn 0) phải giữ ý nghĩa 'vắng mặt'.
    Không nội suy làm tay xuất hiện tọa độ ảo lơ lửng.
    """
    ts = np.array([0.0, 1.0, 2.0, 3.2])
    frames = np.zeros((4, 258), dtype=np.float32)
    # Frame 0: tay vắng mặt (0)
    # Frame 1: tay xuất hiện
    frames[1, 132:195] = 5.0
    # Frame 2: tay biến mất (0)
    # Frame 3: tay vắng mặt (0)

    res = resample_raw_sequence(ts, frames, target_duration=3.2, target_length=30)
    assert res is not None

    # Kiểm tra tại các thời điểm gần frame 0, 2, 3 (xa frame 1) tay phải bằng 0 hoàn toàn
    # Đặc biệt frame 0 và frame cuối cùng (k=0 và k=29) phải là 0
    assert np.all(res[0, 132:195] == 0.0), "Frame đầu phải giữ tay vắng mặt (toàn 0)"
    assert np.all(res[-1, 132:195] == 0.0), "Frame cuối phải giữ tay vắng mặt (toàn 0)"

    # Không được có giá trị nội suy trung gian dạng 0 < val < 5.0 nếu áp dụng Nearest Neighbor khi mất tay
    hand_vals = res[:, 132:195]
    non_zero_vals = hand_vals[hand_vals != 0.0]
    assert np.all(np.isclose(non_zero_vals, 5.0)), f"Phát hiện giá trị nội suy lai giữa có tay và mất tay: {non_zero_vals}"
    return True


def test_jittered_timestamps():
    """Kiểm tra timestamp không đều (jitter) vẫn resample cách đều theo thời gian"""
    # Tạo chuỗi thời gian giật cục: 0, 0.1, 0.5, 0.6, 1.8, 3.0, 3.2
    ts = np.array([0.0, 0.12, 0.45, 0.88, 1.25, 2.10, 2.95, 3.20])
    frames = np.zeros((len(ts), 258), dtype=np.float32)
    for i, t in enumerate(ts):
        frames[i, 0] = t  # giá trị bằng chính timestamp

    res = resample_raw_sequence(ts, frames, target_duration=3.2, target_length=30)
    assert res is not None

    # Giá trị tại frame k phải xấp xỉ linspace(0, 3.2, 30)
    expected_times = np.linspace(0.0, 3.2, 30)
    max_err = np.max(np.abs(res[:, 0] - expected_times))
    assert max_err < 1e-4, f"Lỗi jitter timestamp: {max_err}"
    return True


def test_short_input_returns_none():
    """Đầu vào ngắn (< 2 frame) phải trả về None an toàn"""
    ts_empty = []
    frames_empty = []
    assert resample_raw_sequence(ts_empty, frames_empty, 3.2) is None

    ts_one = [1.0]
    frames_one = [np.zeros(258)]
    assert resample_raw_sequence(ts_one, frames_one, 3.2) is None
    return True


def main():
    print("=" * 70)
    print("BẮT ĐẦU KIỂM THỬ TOÁN HỌC: TEMPORAL RESAMPLING (TEST_TEMPORAL_UTILS)")
    print("=" * 70)

    tests = [
        ("Output Shape (30, 258)", test_output_shape),
        ("Chuỗi hằng số (Constant Sequence)", test_constant_sequence),
        ("Chuỗi tuyến tính (Linear Sequence)", test_linear_sequence),
        ("Bảo toàn tay vắng mặt (Absent Hand Preservation)", test_absent_hand_preservation),
        ("Khử nhiễu thời gian Jitter (Jittered Timestamps)", test_jittered_timestamps),
        ("Xử lý chuỗi ngắn < 2 frames (Short Input Fallback)", test_short_input_returns_none),
    ]

    all_passed = True
    for name, test_fn in tests:
        try:
            res = test_fn()
            status = "✅ PASS" if res else "❌ FAIL"
            print(f"  [{status}] {name}")
        except Exception as e:
            all_passed = False
            print(f"  [❌ FAIL] {name}: {e}")

    print("=" * 70)
    print(f"TỔNG KẾT: {'TẤT CẢ TEST PASS (6/6)' if all_passed else 'CÓ TEST THẤT BẠI'}")
    print("=" * 70)
    if not all_passed:
        sys.exit(1)


if __name__ == "__main__":
    main()
