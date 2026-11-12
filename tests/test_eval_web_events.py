"""
tests/test_eval_web_events.py
=============================
Unit test và integration test qua subprocess cho tools/eval_web_events.py (Task A4 & D3).
"""

import os
import sys
import unittest
import tempfile
import subprocess
from tools.eval_web_events import align_sequences, evaluate_blocked_events

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PYTHON_EXE = sys.executable


class TestEvalWebEvents(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.events_file = os.path.join(self.temp_dir.name, "web_events.csv")
        self.expected_file = os.path.join(self.temp_dir.name, "expected_words.txt")
        self.blocked_file = os.path.join(self.temp_dir.name, "web_gate_blocked.csv")

        # Tạo file expected_words giả lập (3 từ: xin_chao, cam_on, tam_biet)
        with open(self.expected_file, "w", encoding="utf-8") as f:
            f.write("xin_chao\ncam_on\ntam_biet\n")

        # Tạo file web_events giả lập (xin_chao, f, cam_on, tam_biet)
        with open(self.events_file, "w", encoding="utf-8") as f:
            f.write("timestamp,action,confidence,t_first_hand_to_output_s,gate_state\n")
            f.write("2026-11-11T12:00:01,xin_chao,0.9500,1.250,OPEN_DYNAMIC\n")
            f.write("2026-11-11T12:00:03,f,0.5200,0.850,OPEN_STATIC\n")
            f.write("2026-11-11T12:00:05,cam_on,0.9100,1.100,OPEN_DYNAMIC\n")
            f.write("2026-11-11T12:00:08,tam_biet,0.8900,1.350,OPEN_DYNAMIC\n")

        # Tạo file web_gate_blocked giả lập (Task D3)
        with open(self.blocked_file, "w", encoding="utf-8") as f:
            f.write("timestamp,would_be_action,confidence,gate_state,hand_ratio,motion_energy\n")
            f.write("2026-11-11T12:00:02,anh_em,0.5500,closed_low_motion,0.800,0.0210\n")
            f.write("2026-11-11T12:00:04,anh_em,0.5800,closed_low_motion,0.750,0.0180\n")
            f.write("2026-11-11T12:00:06,m,0.5100,closed_no_hand,0.300,0.0050\n")

    def tearDown(self):
        self.temp_dir.cleanup()

    # --- 1. KIỂM THỬ TOÁN HỌC CĂN CHỈNH CHUỖI ---
    def test_perfect_alignment(self):
        """Kiểm tra khi nhận dạng đúng 100%."""
        expected = ["xin_chao", "bo_me", "cam_on"]
        rec = ["xin_chao", "bo_me", "cam_on"]
        aligned = align_sequences(expected, rec)
        self.assertEqual(len(aligned), 3)
        for _, _, st, _ in aligned:
            self.assertEqual(st, 'CORRECT')

    def test_spurious_insertion(self):
        """Kiểm tra phát hiện từ dư thừa khi tay chuyển tiếp."""
        expected = ["xin_chao", "cam_on"]
        rec = ["xin_chao", "f", "cam_on"]
        aligned = align_sequences(expected, rec)
        statuses = [st for _, _, st, _ in aligned]
        self.assertEqual(statuses, ['CORRECT', 'INSERTION', 'CORRECT'])
        self.assertEqual(aligned[1][1], 'f')

    def test_deletion_and_substitution(self):
        """Kiểm tra nhận dạng sai và bỏ sót từ."""
        expected = ["a", "b", "c"]
        rec = ["a", "d"]
        aligned = align_sequences(expected, rec)
        statuses = [st for _, _, st, _ in aligned]
        self.assertEqual(statuses.count('CORRECT'), 1)
        self.assertEqual(statuses.count('SUBSTITUTION'), 1)
        self.assertEqual(statuses.count('DELETION'), 1)

    # --- 2. KIỂM THỬ THỐNG KÊ BLOCKED LOG (Task D3) ---
    def test_evaluate_blocked_events(self):
        """Kiểm tra phân tích thống kê file web_gate_blocked.csv."""
        res = evaluate_blocked_events(self.blocked_file)
        self.assertIsNotNone(res)
        self.assertEqual(res['total_blocked'], 3)
        self.assertEqual(res['by_state']['closed_low_motion'], 2)
        self.assertEqual(res['by_state']['closed_no_hand'], 1)
        self.assertEqual(res['by_action']['anh_em'], 2)
        self.assertEqual(res['by_action']['m'], 1)

    # --- 3. KIỂM THỬ SUBPROCESS: CẢ HAI CÁCH GỌI (Task A4) ---
    def test_subprocess_flag_style_call(self):
        """Kiểm tra gọi script bằng cờ --events, --ground-truth, --blocked."""
        cmd = [
            PYTHON_EXE,
            os.path.join(PROJECT_ROOT, "tools", "eval_web_events.py"),
            "--events", self.events_file,
            "--ground-truth", self.expected_file,
            "--blocked", self.blocked_file
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
        self.assertEqual(proc.returncode, 0, f"Lỗi script: {proc.stderr}")
        self.assertIn("Đã nạp 4 dòng sự kiện", proc.stdout)
        self.assertIn("Đã nạp 3 từ kỳ vọng", proc.stdout)
        self.assertIn("Đúng (Correct)         :   3", proc.stdout)
        self.assertIn("Dư thừa (Insertion)   :   1", proc.stdout)
        self.assertIn("THỐNG KÊ SỰ KIỆN BỊ CỔNG CHUYỂN ĐỘNG CHẶN", proc.stdout)

    def test_subprocess_positional_style_call(self):
        """Kiểm tra gọi script bằng đối số vị trí: script [events] [ground_truth]."""
        cmd = [
            PYTHON_EXE,
            os.path.join(PROJECT_ROOT, "tools", "eval_web_events.py"),
            self.events_file,
            self.expected_file
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
        self.assertEqual(proc.returncode, 0, f"Lỗi script: {proc.stderr}")
        self.assertIn("Đã nạp 4 dòng sự kiện", proc.stdout)
        self.assertIn("Đã nạp 3 từ kỳ vọng", proc.stdout)
        self.assertIn("Đúng (Correct)         :   3", proc.stdout)

    def test_subprocess_fails_on_missing_file(self):
        """Kiểm tra script thoát mã lỗi != 0 khi file không tồn tại."""
        non_existent_file = os.path.join(self.temp_dir.name, "does_not_exist.csv")
        cmd = [
            PYTHON_EXE,
            os.path.join(PROJECT_ROOT, "tools", "eval_web_events.py"),
            "--events", non_existent_file,
            "--ground-truth", self.expected_file
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
        self.assertNotEqual(proc.returncode, 0, "Script phải thoát mã lỗi khi file không tồn tại!")
        self.assertIn("Không tìm thấy file sự kiện", proc.stderr)


if __name__ == '__main__':
    unittest.main()
