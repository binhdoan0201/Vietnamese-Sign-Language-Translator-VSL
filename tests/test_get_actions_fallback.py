"""
tests/test_get_actions_fallback.py
==================================
Unit test kiểm thử cơ chế dự phòng (fallback) của hàm get_actions trong feature_utils.py.
Xác minh hàm hoạt động chính xác trong cả 4 kịch bản:
  (i)   Thư mục dữ liệu giả lập với vài thư mục lớp -> trả về đúng tên sắp xếp.
  (ii)  Đường dẫn dữ liệu không tồn tại -> đọc dự phòng từ models/classes.json.
  (iii) Thư mục dữ liệu tồn tại nhưng rỗng -> kích hoạt dự phòng từ models/classes.json.
  (iv)  Kết quả trả về luôn là danh sách đã được sắp xếp tăng dần.
"""
import os
import sys
import shutil
import tempfile
import unittest

# Đảm bảo đường dẫn gốc dự án nằm trong sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from feature_utils import get_actions


class TestGetActionsFallback(unittest.TestCase):
    """Kiểm thử tính năng lấy danh sách lớp và cơ chế dự phòng của get_actions."""

    def test_synthetic_dir_with_classes(self):
        """(i) Thư mục dữ liệu giả lập có vài lớp -> trả về đúng tên đã sắp xếp."""
        temp_dir = tempfile.mkdtemp()
        try:
            # Tạo các thư mục lớp theo thứ tự bất kỳ kèm file rác
            class_names = ["xin_loi", "bo_me", "anh_em", "a"]
            for c in class_names:
                os.makedirs(os.path.join(temp_dir, c), exist_ok=True)
            # Tạo file rác không phải thư mục (phải bị bỏ qua)
            with open(os.path.join(temp_dir, ".DS_Store"), "w") as f:
                f.write("junk")
            with open(os.path.join(temp_dir, "sample.txt"), "w") as f:
                f.write("junk")

            actions = get_actions(raw_dir=temp_dir)
            expected = sorted(class_names)
            self.assertEqual(actions, expected)
            self.assertEqual(len(actions), 4)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_nonexistent_dir_fallback(self):
        """(ii) Đường dẫn không tồn tại -> trả về đúng 43 tên từ models/classes.json."""
        nonexistent_path = os.path.join(PROJECT_ROOT, "path_that_does_not_exist_12345")
        actions = get_actions(raw_dir=nonexistent_path)

        self.assertIsInstance(actions, list)
        self.assertEqual(len(actions), 43)
        self.assertEqual(actions[0], "a")
        self.assertEqual(actions[-1], "z")
        self.assertIn("xin_chao", actions)
        self.assertIn("dau_hoi", actions)

    def test_empty_dir_fallback(self):
        """(iii) Thư mục tồn tại nhưng rỗng -> tự động dự phòng đọc từ models/classes.json."""
        temp_empty_dir = tempfile.mkdtemp()
        try:
            actions = get_actions(raw_dir=temp_empty_dir)
            self.assertIsInstance(actions, list)
            self.assertEqual(len(actions), 43)
            self.assertEqual(actions[0], "a")
            self.assertEqual(actions[-1], "z")
        finally:
            shutil.rmtree(temp_empty_dir, ignore_errors=True)

    def test_result_always_sorted(self):
        """(iv) Kết quả trả về luôn là danh sách đã được sắp xếp."""
        # Kiểm tra với fallback mặc định
        actions_fallback = get_actions(raw_dir="dataset/raw")
        self.assertEqual(actions_fallback, sorted(actions_fallback))

        # Kiểm tra với thư mục giả lập
        temp_dir = tempfile.mkdtemp()
        try:
            unsorted_classes = ["z", "k", "b", "d", "c"]
            for c in unsorted_classes:
                os.makedirs(os.path.join(temp_dir, c), exist_ok=True)
            actions = get_actions(raw_dir=temp_dir)
            self.assertEqual(actions, sorted(unsorted_classes))
            self.assertEqual(actions, sorted(actions))
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == '__main__':
    unittest.main()
