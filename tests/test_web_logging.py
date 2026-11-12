"""
tests/test_web_logging.py
=========================
Kiểm thử đơn vị cho module pipeline/web_logging.py (Task A3 & D1).
"""

import os
import csv
import unittest
import tempfile
from pipeline.web_logging import (
    write_event,
    write_blocked,
    EVENT_CSV_HEADER,
    BLOCKED_CSV_HEADER
)


class TestWebLogging(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.event_file = os.path.join(self.temp_dir.name, "test_events.csv")
        self.blocked_file = os.path.join(self.temp_dir.name, "test_blocked.csv")

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_write_event_creates_header_and_single_row(self):
        """(1) Tạo file có header khi chưa có và có đúng 1 dòng dữ liệu."""
        self.assertFalse(os.path.exists(self.event_file))
        res = write_event(self.event_file, "xin_chao", 0.9542, 1.25, "OPEN_DYNAMIC")
        self.assertTrue(res)
        self.assertTrue(os.path.exists(self.event_file))

        with open(self.event_file, "r", encoding="utf-8") as f:
            reader = list(csv.reader(f))
            self.assertEqual(len(reader), 2)  # 1 header + 1 data row
            self.assertEqual(reader[0], EVENT_CSV_HEADER)
            self.assertEqual(reader[1][1], "xin_chao")
            self.assertEqual(reader[1][2], "0.9542")
            self.assertEqual(reader[1][3], "1.250")
            self.assertEqual(reader[1][4], "OPEN_DYNAMIC")

    def test_write_event_append_mode(self):
        """(2) Ghi nối tiếp nhiều dòng dữ liệu."""
        write_event(self.event_file, "xin_chao", 0.95, 1.2, "OPEN_DYNAMIC")
        write_event(self.event_file, "cam_on", 0.88, 0.9, "OPEN_STATIC")
        write_event(self.event_file, "tam_biet", 0.92, None, "BYPASS_DISABLED")

        with open(self.event_file, "r", encoding="utf-8") as f:
            reader = list(csv.reader(f))
            self.assertEqual(len(reader), 4)  # 1 header + 3 data rows
            self.assertEqual(reader[1][1], "xin_chao")
            self.assertEqual(reader[2][1], "cam_on")
            self.assertEqual(reader[3][1], "tam_biet")
            self.assertEqual(reader[3][3], "")  # None latency -> ""

    def test_write_event_does_not_raise_on_valid_data(self):
        """(3) Hàm không ném lỗi ngoại lệ với dữ liệu hợp lệ."""
        try:
            write_event(self.event_file, "a", 1.0, 0.5, "OPEN_STATIC")
        except Exception as e:
            self.fail(f"write_event ném ngoại lệ không mong muốn: {e}")

    def test_write_blocked_creates_header_and_respects_rate_limit(self):
        """Kiểm tra write_blocked tạo file, ghi header và giới hạn rate 2 dòng/giây."""
        self.assertFalse(os.path.exists(self.blocked_file))
        res1 = write_blocked(self.blocked_file, "anh_em", 0.55, "closed_low_motion", 0.8, 0.02)
        res2 = write_blocked(self.blocked_file, "f", 0.52, "closed_low_motion", 0.9, 0.01)
        res3 = write_blocked(self.blocked_file, "m", 0.58, "closed_low_motion", 0.7, 0.03)

        self.assertTrue(res1)
        self.assertTrue(res2)
        # Lần 3 bị rate limit chặn vì cùng thời điểm (> 2 calls trong 1s)
        self.assertFalse(res3)

        with open(self.blocked_file, "r", encoding="utf-8") as f:
            reader = list(csv.reader(f))
            self.assertEqual(len(reader), 3)  # 1 header + 2 allowed data rows
            self.assertEqual(reader[0], BLOCKED_CSV_HEADER)
            self.assertEqual(reader[1][1], "anh_em")
            self.assertEqual(reader[2][1], "f")


if __name__ == '__main__':
    unittest.main()
