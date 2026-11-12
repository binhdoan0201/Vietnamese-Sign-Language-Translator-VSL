"""
tools/package_release.py
========================
Đóng gói release_package.zip chứa tất cả tài liệu, cấu hình, mã nguồn mới,
bằng chứng thực nghiệm A/B và thông tin bộ dữ liệu mở.
"""
import os
import zipfile

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ZIP_PATH = os.path.join(PROJECT_ROOT, "release_package.zip")

FILES_TO_PACK = [
    "REPORT_RELEASE.md",
    "README.md",
    "CONTRIBUTING.md",
    "LICENSE",
    ".gitignore",
    "requirements.txt",
    "requirements-dev.txt",
    "tree_after.txt",
    "git_add_dry_run.txt",
    "git_size.txt",
    "changes.diff",
    "cleanup_moves.csv",
    "incoming/signer_map_proposed.csv",
    "results/INDEX.md",
    "release_assets/vsl_landmarks_258_v1.zip.sha256",
    "release_assets/dataset_public/DATASET_CARD.md",
    "release_assets/dataset_public/LICENSE-DATA",
    "release_assets/dataset_public/checksums.sha256",
    "release_assets/dataset_public/metadata.csv",
    "release_assets/dataset_public/classes.json",
    "release_assets/dataset_public/sessions.json",
    "tools/export_public_dataset.py",
    "tools/verify_dataset.py",
    "tools/generate_dataset_card.py",
    "tools/prove_equivalence.py",
    "tools/check_privacy.py",
    "tools/propose_signer_clusters.py",
    "tools/scan_sensitive.py",
    "tests/test_preprocess_dual_format.py",
    "tests/test_public_dataset.py",
    "tests/test_app_config.py",
]

# Thêm tất cả file trong results/ab_web
for root, _, files in os.walk(os.path.join(PROJECT_ROOT, "results", "ab_web")):
    for f in files:
        full_path = os.path.join(root, f)
        rel_path = os.path.relpath(full_path, PROJECT_ROOT).replace("\\", "/")
        FILES_TO_PACK.append(rel_path)

def main():
    print(f"Đang tạo gói {ZIP_PATH}...")
    if os.path.exists(ZIP_PATH):
        os.remove(ZIP_PATH)

    with zipfile.ZipFile(ZIP_PATH, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for f in sorted(set(FILES_TO_PACK)):
            full_path = os.path.join(PROJECT_ROOT, f)
            if os.path.exists(full_path):
                zipf.write(full_path, arcname=f)
                print(f"  [+] Đã thêm: {f}")
            else:
                print(f"  [-] KHÔNG TÌM THẤY: {f}")

    size_bytes = os.path.getsize(ZIP_PATH)
    print(f"\n✅ ĐÃ TẠO THÀNH CÔNG: {ZIP_PATH}")
    print(f"   Dung lượng: {size_bytes:,} bytes ({size_bytes / (1024*1024):.2f} MB)")

if __name__ == '__main__':
    main()
