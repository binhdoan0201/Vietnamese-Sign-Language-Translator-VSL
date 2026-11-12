"""Unit tests cho module pipeline/segment_utils.py."""
import unittest
import numpy as np
from pipeline.segment_utils import (
    is_hand_present,
    get_hand_present_span,
    get_motion_span,
    segment_raw_sample,
    segment_dataset
)


class TestSegmentUtils(unittest.TestCase):

    def setUp(self):
        # Tạo sample cơ sở có vai chuẩn ở pose landmarks 11 và 12
        self.sample = np.zeros((30, 258), dtype=np.float32)
        # Left shoulder (11): (x=0.4, y=0.5, z=0.0) -> index 11*4 = 44
        self.sample[:, 44:47] = [0.4, 0.5, 0.0]
        self.sample[:, 47] = 1.0  # visibility
        # Right shoulder (12): (x=0.6, y=0.5, z=0.0) -> index 12*4 = 48
        self.sample[:, 48:51] = [0.6, 0.5, 0.0]
        self.sample[:, 51] = 1.0  # visibility

    def test_hand_present_detection(self):
        """Kiểm tra phát hiện frame có tay."""
        frame = np.zeros(258, dtype=np.float32)
        self.assertFalse(is_hand_present(frame))

        # Đặt tay phải có giá trị
        frame[195] = 0.5
        self.assertTrue(is_hand_present(frame))

        # Đặt tay trái có giá trị
        frame_lh = np.zeros(258, dtype=np.float32)
        frame_lh[132] = 0.2
        self.assertTrue(is_hand_present(frame_lh))

    def test_hand_present_span(self):
        """Kiểm tra tìm khoảng xuất hiện bàn tay."""
        s = self.sample.copy()
        # Bàn tay xuất hiện từ frame 10 đến frame 20
        s[10:21, 195:258] = 0.5
        span = get_hand_present_span(s)
        self.assertEqual(span, (10, 20))

        # Không có tay
        s_empty = np.zeros((30, 258), dtype=np.float32)
        self.assertIsNone(get_hand_present_span(s_empty))

    def test_motion_span_detection(self):
        """Kiểm tra tìm khoảng chuyển động cổ tay."""
        s = self.sample.copy()
        # Cho cổ tay di chuyển mạnh từ frame 8 đến frame 18
        s[:, 195:258] = 0.5  # tay có xuất hiện
        for t in range(8, 19):
            s[t, 195] = 0.5 + (t - 8) * 0.05  # x thay đổi liên tục

        span = get_motion_span(s, ratio=0.20)
        self.assertIsNotNone(span)
        start, end = span
        self.assertTrue(start >= 7 and end <= 20)

    def test_segment_hand_shape_and_fallback(self):
        """Kiểm tra shape (30, 258) và cơ chế fallback khi < 8 frame."""
        s = self.sample.copy()
        # Chỉ có 1 frame có tay -> span = (10, 10), sau đệm 3 mỗi bên = 7 frame (< 8 frame)
        s[10, 195:258] = 0.5
        out, fallback = segment_raw_sample(s, mode='hand', min_span=8)
        self.assertTrue(fallback)
        self.assertEqual(out.shape, (30, 258))

        # Có tay từ 5 đến 22 -> span = (5, 22), sau đệm = 24 frame (>= 8 frame)
        s[5:23, 195:258] = 0.5
        out2, fallback2 = segment_raw_sample(s, mode='hand', min_span=8)
        self.assertFalse(fallback2)
        self.assertEqual(out2.shape, (30, 258))

    def test_absent_hand_preservation_in_segmentation(self):
        """Kiểm tra bảo toàn frame vắng tay (tay trái toàn 0 thì sau cắt vẫn toàn 0)."""
        s = self.sample.copy()
        # Chỉ có tay phải, tay trái hoàn toàn vắng
        s[5:25, 195:258] = 0.5
        out, _ = segment_raw_sample(s, mode='hand')
        self.assertEqual(np.count_nonzero(out[:, 132:195]), 0)

    def test_segment_dataset_stats(self):
        """Kiểm tra thống kê segment_dataset."""
        X = np.zeros((10, 30, 258), dtype=np.float32)
        for i in range(10):
            X[i, :, 44:47] = [0.4, 0.5, 0.0]
            X[i, :, 48:51] = [0.6, 0.5, 0.0]
        # 5 mẫu có tay span dài (không fallback), 5 mẫu không tay (fallback)
        for i in range(5):
            X[i, 5:25, 195:258] = 0.5

        X_seg, stats = segment_dataset(X, mode='hand')
        self.assertEqual(X_seg.shape, (10, 30, 258))
        self.assertEqual(stats['fallback_count'], 5)
        self.assertEqual(stats['total'], 10)


if __name__ == '__main__':
    unittest.main()
