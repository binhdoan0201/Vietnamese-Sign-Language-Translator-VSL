"""Unit test kiểm tra endpoint GET /api/config của Flask app."""
import unittest
import os
import sys

# Đảm bảo import được app
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
from app import app


class TestAppConfig(unittest.TestCase):
    """Kiểm tra phản hồi của endpoint cấu hình /api/config."""

    def setUp(self):
        self.client = app.test_client()

    def test_get_config_success(self):
        """Xác minh endpoint /api/config trả về đúng mã 200 và các tham số chuẩn."""
        response = self.client.get('/api/config')
        self.assertEqual(response.status_code, 200)

        data = response.get_json()
        self.assertIsNotNone(data)

        # Kiểm tra các tham số truyền nhận hình ảnh
        self.assertEqual(data.get('send_width'), 640)
        self.assertEqual(data.get('send_height'), 480)
        self.assertEqual(data.get('jpeg_quality'), 0.7)
        self.assertEqual(data.get('mp_complexity'), 1)

        # Kiểm tra thông số thời gian
        self.assertEqual(data.get('target_duration'), 3.2)
        self.assertEqual(data.get('cooldown_sec'), 1.0)

        # Kiểm tra cổng chuyển động (Motion Gate - mặc định BẬT sau vòng 4)
        self.assertTrue(data.get('gate_enabled'), "GATE_ENABLED phải mặc định là True")
        self.assertEqual(data.get('gate_min_hand_ratio'), 0.50)
        self.assertEqual(data.get('gate_min_motion'), 0.05)

        # Kiểm tra danh sách các lớp tĩnh
        static_classes = data.get('static_classes', [])
        self.assertGreaterEqual(len(static_classes), 10)


if __name__ == '__main__':
    unittest.main()
