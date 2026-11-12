"""
tests/test_preprocess_dual_format.py
====================================
Unit test kiểm thử khả năng nhận dạng và xử lý cả 2 định dạng đầu vào:
(30, 1662) và (30, 258) của pipeline/preprocess_data.py.
"""
import os
import sys
import shutil
import tempfile
import unittest
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from feature_utils import slice_features, build_features


class TestPreprocessDualFormat(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.input_1662_dir = os.path.join(self.temp_dir, "raw_1662")
        self.input_258_dir = os.path.join(self.temp_dir, "raw_258")
        self.out_1662_dir = os.path.join(self.temp_dir, "out_1662")
        self.out_258_dir = os.path.join(self.temp_dir, "out_258")

        # Tạo cấu trúc thư mục giả lập 2 class x 2 session x 2 sample
        np.random.seed(42)
        for c in ["class_a", "class_b"]:
            for s in ["session_01", "session_02"]:
                p1662 = os.path.join(self.input_1662_dir, c, s)
                p258 = os.path.join(self.input_258_dir, c, s)
                os.makedirs(p1662, exist_ok=True)
                os.makedirs(p258, exist_ok=True)

                for i in range(2):
                    # Giả lập mẫu 1662 có pose hợp lệ
                    sample_1662 = np.random.uniform(0.1, 0.9, (30, 1662)).astype(np.float32)
                    # Gán vai trái (idx 11*4=44) và vai phải (idx 12*4=48)
                    sample_1662[:, 44:47] = [0.4, 0.5, 0.0]
                    sample_1662[:, 47] = 0.9
                    sample_1662[:, 48:51] = [0.6, 0.5, 0.0]
                    sample_1662[:, 51] = 0.9

                    # Bản 258 tương ứng
                    sample_258 = slice_features(sample_1662)

                    np.save(os.path.join(p1662, f"sample_{i:03d}.npy"), sample_1662)
                    np.save(os.path.join(p258, f"sample_{i:03d}.npy"), sample_258)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_equivalence_1662_and_258(self):
        """Xác minh đầu ra từ 1662 và 258 hoàn toàn tương đương."""
        from pipeline.preprocess_data import find_npy_files_with_session, EXPECTED_RAW_SHAPE_1662, EXPECTED_RAW_SHAPE_258

        for c in ["class_a", "class_b"]:
            p1662 = os.path.join(self.input_1662_dir, c)
            p258 = os.path.join(self.input_258_dir, c)

            files_1662 = find_npy_files_with_session(p1662)
            files_258 = find_npy_files_with_session(p258)

            self.assertEqual(len(files_1662), len(files_258))

            for (f1662, s1), (f258, s2) in zip(files_1662, files_258):
                self.assertEqual(s1, s2)
                arr_1662 = np.load(f1662)
                arr_258 = np.load(f258)

                self.assertEqual(arr_1662.shape, EXPECTED_RAW_SHAPE_1662)
                self.assertEqual(arr_258.shape, EXPECTED_RAW_SHAPE_258)

                # Cắt từ 1662
                sliced_from_1662 = slice_features(arr_1662)
                np.testing.assert_array_equal(sliced_from_1662, arr_258)

                # Tính đặc trưng 384
                feat_1662 = build_features(sliced_from_1662)
                feat_258 = build_features(arr_258)

                np.testing.assert_allclose(feat_1662, feat_258, atol=1e-6)


if __name__ == '__main__':
    unittest.main()
