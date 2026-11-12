"""
tools/scan_sensitive.py
=======================
Quét tìm thông tin nhạy cảm trong các file văn bản sẽ được theo dõi bởi Git:
- Đường dẫn máy cục bộ: 'C:\\Users', 'D:\\'
- Chuỗi bí mật: 'token', 'password', 'api_key'
- Địa chỉ email
"""
import os
import re

PATTERNS = {
    "local_path_c": re.compile(r"C:\\Users", re.IGNORECASE),
    "local_path_d": re.compile(r"D:\\", re.IGNORECASE),
    "email": re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+"),
    "password": re.compile(r"password\s*[:=]", re.IGNORECASE),
    "token": re.compile(r"token\s*[:=]", re.IGNORECASE),
    "api_key": re.compile(r"api_key\s*[:=]", re.IGNORECASE),
}

IGNORE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.h5', '.npy', '.zip', '.ico', '.mp3'}

def main():
    print("=" * 80)
    print("BẮT ĐẦU QUÉT THÔNG TIN NHẠY CẢM TRONG CÁC FILE SẼ THEO DÕI BỞI GIT")
    print("=" * 80)

    with open('git_add_dry_run.txt', 'r', encoding='utf-8') as f:
        lines = [l.strip() for l in f.readlines() if l.strip().startswith('add ')]

    files = [l[5:-1] for l in lines]
    text_files = [f for f in files if os.path.splitext(f)[1].lower() not in IGNORE_EXTENSIONS]

    findings = []

    for f in text_files:
        if not os.path.exists(f):
            continue
        try:
            with open(f, 'r', encoding='utf-8', errors='ignore') as fp:
                for line_idx, line in enumerate(fp, 1):
                    for pat_name, regex in PATTERNS.items():
                        matches = regex.findall(line)
                        if matches:
                            findings.append({
                                'file': f,
                                'line': line_idx,
                                'pattern': pat_name,
                                'match': matches[0] if len(matches) == 1 else str(matches),
                                'text': line.strip()[:100]
                            })
        except Exception as e:
            print(f"[SKIP] Lỗi đọc {f}: {e}")

    print(f"Tổng số file văn bản đã quét: {len(text_files)}")
    print(f"Tổng số phát hiện: {len(findings)}\n")

    # Phân loại: Tài liệu / Code vs Bằng chứng results/
    docs_code = [item for item in findings if not item['file'].startswith('results/') and not item['file'].startswith('models/eval/')]
    evidence_logs = [item for item in findings if item['file'].startswith('results/') or item['file'].startswith('models/eval/')]

    print("--- 1. PHÁT HIỆN TRONG TÀI LIỆU / MÃ NGUỒN (CẦN SỬA SANG TƯƠNG ĐỐI) ---")
    if docs_code:
        for it in docs_code:
            print(f"  [{it['pattern']}] {it['file']}:{it['line']} -> {it['text']}")
    else:
        print("  ✅ Không có chuỗi nhạy cảm nào trong tài liệu hoặc mã nguồn!")

    print("\n--- 2. PHÁT HIỆN TRONG CÁC LOG BẰNG CHỨNG RESULTS/ (GIỮ NGUYÊN NỘI DUNG GỐC) ---")
    files_with_logs = sorted(list(set(it['file'] for it in evidence_logs)))
    print(f"  Số file log chứa đường dẫn cá nhân: {len(files_with_logs)}")
    for lf in files_with_logs:
        count = sum(1 for it in evidence_logs if it['file'] == lf)
        print(f"  - {lf:<50} ({count} vị trí chứa đường dẫn)")

    print("=" * 80)

if __name__ == '__main__':
    main()
