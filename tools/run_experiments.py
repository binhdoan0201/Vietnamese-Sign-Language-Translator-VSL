"""
tools/run_experiments.py
========================
Script điều phối chạy tuần tự các thí nghiệm LOSO Protocol P2 cho Vòng 4.
Ghi nhận chính xác thời gian bắt đầu, kết thúc, thời lượng chạy của từng thí nghiệm (UTF-8).
"""

import os
import sys
import time
import subprocess
from datetime import datetime

# Ép stdout UTF-8
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PYTHON_EXE = os.path.join(PROJECT_ROOT, ".venv", "Scripts", "python.exe")
TRAIN_SCRIPT = os.path.join(PROJECT_ROOT, "pipeline", "train_model.py")
LOGS_DIR = os.path.join(PROJECT_ROOT, "logs")


def run_single_experiment(aug_type, seed, out_json, log_name, aug_factor=3, fixed_epochs=40):
    log_path = os.path.join(LOGS_DIR, log_name)
    start_time = datetime.now()
    t0 = time.time()
    
    print("\n" + "=" * 80)
    print(f"BẮT ĐẦU THÍ NGHIỆM: Augment={aug_type} | Seed={seed} | K={aug_factor} | Epochs={fixed_epochs}")
    print(f"Thời gian bắt đầu: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Output JSON: {out_json}")
    print(f"Log file: {log_path}")
    print("=" * 80)

    cmd = [
        PYTHON_EXE,
        TRAIN_SCRIPT,
        "--eval", "loso",
        "--fixed-epochs", str(fixed_epochs),
        "--augment", aug_type,
        "--aug-factor", str(aug_factor),
        "--seed", str(seed),
        "--out-json", out_json
    ]

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"

    with open(log_path, "w", encoding="utf-8") as f_log:
        f_log.write(f"=== BẮT ĐẦU LOSO P2: Augment={aug_type}, Seed={seed}, Epochs={fixed_epochs}, K={aug_factor} ===\n")
        f_log.write(f"Thời gian bắt đầu: {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f_log.flush()

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=PROJECT_ROOT,
            env=env,
            text=True,
            encoding="utf-8"
        )

        for line in proc.stdout:
            f_log.write(line)
            f_log.flush()
            # In các dòng quan trọng ra console
            if any(k in line for k in ["FOLD", "Evaluated Epoch", "Test Accuracy", "MEAN ± STD", "Epoch 1/40", "Epoch 20/40", "Epoch 40/40"]):
                print("  " + line.strip())

        proc.wait()

        end_time = datetime.now()
        duration_sec = time.time() - t0
        duration_min = duration_sec / 60.0

        f_log.write("\n\n=== HOÀN THÀNH LOSO P2 ===")
        f_log.write(f"\nThời gian kết thúc: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        f_log.write(f"\nTổng thời gian chạy: {duration_min:.2f} phút ({duration_sec:.1f} giây)")
        f_log.write(f"\nExit code: {proc.returncode}\n")

    print(f"\n✓ Hoàn tất {aug_type} Seed {seed}: Thời gian {duration_min:.2f} phút. Exit code: {proc.returncode}")
    return proc.returncode, duration_min


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Chạy thí nghiệm LOSO Vòng 4")
    parser.add_argument("--augment", type=str, required=True, choices=["rot_y", "rot_yz", "rot_z", "full"])
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out-json", type=str, required=True)
    parser.add_argument("--log-name", type=str, required=True)
    parser.add_argument("--aug-factor", type=int, default=3)
    parser.add_argument("--fixed-epochs", type=int, default=40)

    args = parser.parse_args()

    run_single_experiment(
        aug_type=args.augment,
        seed=args.seed,
        out_json=args.out_json,
        log_name=args.log_name,
        aug_factor=args.aug_factor,
        fixed_epochs=args.fixed_epochs
    )


if __name__ == '__main__':
    main()
