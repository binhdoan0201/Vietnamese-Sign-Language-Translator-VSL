"""
tests/test_public_dataset.py
============================
Unit test kiểm thử chức năng của tools/verify_dataset.py và tools/export_public_dataset.py
với dữ liệu giả lập (synthetic data).
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

from tools.verify_dataset import verify_dataset
from tools.export_public_dataset import export_dataset


class TestPublicDataset(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.raw_dir = os.path.join(self.temp_dir, "raw")
        self.public_dir = os.path.join(self.temp_dir, "public")

        # Tạo 2 class x 2 session x 2 sample thô (30, 1662)
        np.random.seed(42)
        self.classes = ["hello", "thanks"]
        for c in self.classes:
            for s in ["session_01", "session_02"]:
                sess_dir = os.path.join(self.raw_dir, c, s)
                os.makedirs(sess_dir, exist_ok=True)
                for i in range(2):
                    sample = np.random.uniform(0.1, 0.9, (30, 1662)).astype(np.float32)
                    np.save(os.path.join(sess_dir, f"sample_{i:03d}.npy"), sample)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_export_and_verify(self):
        """Kiểm tra export_dataset -> sinh (30, 258) và verify_dataset xác nhận hợp lệ."""
        processed, skipped = export_dataset(
            input_dir=self.raw_dir,
            output_dir=self.public_dir
        )
        self.assertEqual(processed, 8)
        self.assertEqual(skipped, 0)

        # Xác minh các file sinh ra
        self.assertTrue(os.path.exists(os.path.join(self.public_dir, "metadata.csv")))
        self.assertTrue(os.path.exists(os.path.join(self.public_dir, "classes.json")))
        self.assertTrue(os.path.exists(os.path.join(self.public_dir, "sessions.json")))
        self.assertTrue(os.path.exists(os.path.join(self.public_dir, "checksums.sha256")))
        self.assertTrue(os.path.exists(os.path.join(self.public_dir, "LICENSE-DATA")))

        # Chạy verify_dataset
        success = verify_dataset(self.public_dir)
        self.assertTrue(success)

    def test_exclude_signers_without_map_raises_error(self):
        """Kiểm tra --exclude-signers khi chưa có bản đồ xác nhận phải báo lỗi hoặc exit."""
        with self.assertRaises(SystemExit):
            export_dataset(
                input_dir=self.raw_dir,
                output_dir=self.public_dir,
                exclude_signers=["signer_B"]
            )


if __name__ == '__main__':
    unittest.main()
