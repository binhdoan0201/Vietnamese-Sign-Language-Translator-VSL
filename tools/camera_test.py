"""
tools/camera_test.py
====================
Kiểm tra khả năng mở và bắt hình ảnh từ webcam.

CÁCH DÙNG:
  python tools/camera_test.py
  (Nhấn Q trên cửa sổ camera để thoát)
"""

import cv2

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Không mở được camera! Hãy kiểm tra quyền truy cập hoặc camera khác.")
    exit(1)

print("✅ Camera đang chạy. Nhấn 'Q' trên cửa sổ hiển thị để thoát.")

while True:
    ret, frame = cap.read()

    if not ret:
        print("❌ Không đọc được frame từ camera!")
        break

    # Lật ngang như gương soi để dễ thao tác
    display_frame = cv2.flip(frame, 1)
    cv2.putText(display_frame, "Camera Test - Nhan 'Q' de thoat", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("Sign Language Translator - Camera Test", display_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
