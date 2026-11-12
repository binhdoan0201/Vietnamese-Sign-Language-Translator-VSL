"""
pipeline/augment_utils.py
=========================
Các phép biến đổi dữ liệu (Data Augmentation) trên đặc trưng THÔ 258 chiều
TRƯỚC khi đưa vào build_features (Task B1 Vòng 4).

Tuân thủ nghiêm ngặt:
- Quy định 4: Áp dụng trên 258 chiều thô (Pose 132 + LH 63 + RH 63).
- Quy định 5: Bàn tay vắng mặt (toàn 0) PHẢI giữ nguyên toàn 0,
  không cộng nhiễu, không xoay hay dịch chuyển làm xuất hiện tọa độ giả.

Các phép Augmentation:
  a) Xoay quanh trục dọc Y (Yaw) khoảng ±20° quanh trung điểm 2 vai.
  b) Xoay quanh trục Z (Roll) khoảng ±10° trong mặt phẳng ảnh (Task B1 Vòng 4).
  c) Scale ±10% và Dịch chuyển (Translation) nhỏ ±0.03.
  d) Nhiễu Gaussian nhỏ (sigma=0.003) trên các landmark tồn tại.
  e) Time-warp nhẹ (±15%) thông qua resample_raw_sequence.
"""

import numpy as np
from temporal_utils import resample_raw_sequence


def get_shoulder_center(pose_flat):
    """
    Tính trung điểm 2 vai từ vector pose 132 chiều (33 landmarks x 4).
    Landmark 11: vai trái [44:48] = (x, y, z, vis)
    Landmark 12: vai phải [48:52] = (x, y, z, vis)
    """
    left_shoulder = pose_flat[44:47]
    right_shoulder = pose_flat[48:51]
    center = (left_shoulder + right_shoulder) / 2.0
    return center  # (3,)


def rotate_landmarks_y(landmarks_xyz, center_xyz, angle_rad):
    """
    Xoay các điểm (N, 3) quanh trục dọc Y đi qua center_xyz một góc angle_rad.
    x' = xc + (x - xc)*cos(theta) + (z - zc)*sin(theta)
    y' = y
    z' = zc - (x - xc)*sin(theta) + (z - zc)*cos(theta)
    """
    cos_t = np.cos(angle_rad)
    sin_t = np.sin(angle_rad)

    dx = landmarks_xyz[:, 0] - center_xyz[0]
    dy = landmarks_xyz[:, 1]
    dz = landmarks_xyz[:, 2] - center_xyz[2]

    new_x = center_xyz[0] + dx * cos_t + dz * sin_t
    new_y = dy
    new_z = center_xyz[2] - dx * sin_t + dz * cos_t

    return np.column_stack([new_x, new_y, new_z])


def rotate_landmarks_z(landmarks_xyz, center_xyz, angle_rad):
    """
    Xoay các điểm (N, 3) quanh trục Z (Roll) đi qua center_xyz một góc angle_rad.
    x' = xc + (x - xc)*cos(theta) - (y - yc)*sin(theta)
    y' = yc + (x - xc)*sin(theta) + (y - yc)*cos(theta)
    z' = z
    """
    cos_t = np.cos(angle_rad)
    sin_t = np.sin(angle_rad)

    dx = landmarks_xyz[:, 0] - center_xyz[0]
    dy = landmarks_xyz[:, 1] - center_xyz[1]
    dz = landmarks_xyz[:, 2]

    new_x = center_xyz[0] + dx * cos_t - dy * sin_t
    new_y = center_xyz[1] + dx * sin_t + dy * cos_t
    new_z = dz

    return np.column_stack([new_x, new_y, new_z])


def augment_single_sequence(
    seq_258,
    rotate_y=True,
    rotate_z=False,
    scale_trans=True,
    noise=True,
    time_warp=True,
    max_angle_deg=20.0,
    max_roll_deg=10.0,
    scale_range=0.10,
    trans_range=0.03,
    noise_sigma=0.003,
    time_warp_range=0.15,
    target_duration=3.2,
    rng=None
):
    """
    Áp dụng augmentation cho 1 mẫu (30, 258).
    Returns:
        np.ndarray shape (30, 258)
    """
    if rng is None:
        rng = np.random.default_rng()

    seq = seq_258.copy()
    T = len(seq)

    # 1. Time-warp nhẹ (kéo giãn thời gian ±15%)
    if time_warp:
        warp_factor = rng.uniform(1.0 - time_warp_range, 1.0 + time_warp_range)
        warped_duration = target_duration * warp_factor
        orig_times = np.linspace(0.0, warped_duration, T)
        warped_seq = resample_raw_sequence(
            orig_times, seq,
            target_duration=target_duration,
            target_length=T
        )
        if warped_seq is not None:
            seq = warped_seq

    # Sinh các thông số ngẫu nhiên cho toàn bộ chuỗi cử chỉ
    angle_y_rad = np.radians(rng.uniform(-max_angle_deg, max_angle_deg)) if rotate_y else 0.0
    angle_z_rad = np.radians(rng.uniform(-max_roll_deg, max_roll_deg)) if rotate_z else 0.0
    scale = rng.uniform(1.0 - scale_range, 1.0 + scale_range) if scale_trans else 1.0
    tx = rng.uniform(-trans_range, trans_range) if scale_trans else 0.0
    ty = rng.uniform(-trans_range, trans_range) if scale_trans else 0.0

    augmented_seq = np.zeros_like(seq)

    for i in range(T):
        frame = seq[i].copy()
        pose = frame[0:132].reshape(33, 4)
        lh = frame[132:195].reshape(21, 3)
        rh = frame[195:258].reshape(21, 3)

        has_lh = np.count_nonzero(lh) > 0
        has_rh = np.count_nonzero(rh) > 0

        # Lấy tâm quay là trung điểm 2 vai của frame hiện tại
        center = (pose[11, :3] + pose[12, :3]) / 2.0

        # a) Xoay quanh trục Y (Yaw ±20°)
        if rotate_y and angle_y_rad != 0.0:
            pose[:, :3] = rotate_landmarks_y(pose[:, :3], center, angle_y_rad)
            if has_lh:
                lh = rotate_landmarks_y(lh, center, angle_y_rad)
            if has_rh:
                rh = rotate_landmarks_y(rh, center, angle_y_rad)

        # b) Xoay quanh trục Z (Roll ±10° - Task B1)
        if rotate_z and angle_z_rad != 0.0:
            pose[:, :3] = rotate_landmarks_z(pose[:, :3], center, angle_z_rad)
            if has_lh:
                lh = rotate_landmarks_z(lh, center, angle_z_rad)
            if has_rh:
                rh = rotate_landmarks_z(rh, center, angle_z_rad)

        # c) Scale & Translation
        if scale_trans:
            pose[:, 0] = center[0] + scale * (pose[:, 0] - center[0]) + tx
            pose[:, 1] = center[1] + scale * (pose[:, 1] - center[1]) + ty
            pose[:, 2] = center[2] + scale * (pose[:, 2] - center[2])

            if has_lh:
                lh[:, 0] = center[0] + scale * (lh[:, 0] - center[0]) + tx
                lh[:, 1] = center[1] + scale * (lh[:, 1] - center[1]) + ty
                lh[:, 2] = center[2] + scale * (lh[:, 2] - center[2])

            if has_rh:
                rh[:, 0] = center[0] + scale * (rh[:, 0] - center[0]) + tx
                rh[:, 1] = center[1] + scale * (rh[:, 1] - center[1]) + ty
                rh[:, 2] = center[2] + scale * (rh[:, 2] - center[2])

        # d) Nhiễu Gaussian nhỏ
        if noise and noise_sigma > 0:
            pose[:, :3] += rng.normal(0, noise_sigma, pose[:, :3].shape)
            if has_lh:
                lh += rng.normal(0, noise_sigma, lh.shape)
            if has_rh:
                rh += rng.normal(0, noise_sigma, rh.shape)

        # e) BẢO TOÀN TAY VẮNG MẶT: Nếu frame gốc không có tay, ép về 0 hoàn toàn
        if not has_lh:
            lh[:] = 0.0
        if not has_rh:
            rh[:] = 0.0

        augmented_seq[i, 0:132] = pose.flatten()
        augmented_seq[i, 132:195] = lh.flatten()
        augmented_seq[i, 195:258] = rh.flatten()

    return augmented_seq


def generate_augmented_dataset(
    X_raw_train,
    y_train,
    augment_mode="full",
    aug_factor=3,
    target_duration=3.2,
    seed=42
):
    """
    Tạo dữ liệu huấn luyện mở rộng từ X_raw_train (N, 30, 258).
    Mỗi mẫu gốc sinh thêm `aug_factor` biến thể.
    Tổng số mẫu đầu ra: N * (1 + aug_factor).

    augment_mode:
      - 'none'                : không augment
      - 'rot_y' / 'rotate_y'  : chỉ xoay quanh trục Y (±20°)
      - 'rot_yz' / 'rot_y_rot_z': xoay trục Y (±20°) + xoay trục Z (±10°)
      - 'rot_z'               : chỉ xoay quanh trục Z (±10°)
      - 'full'                : xoay Y + scale/translate + noise + time-warp
    """
    mode = augment_mode.lower().replace("-", "_")
    if mode in ["none", "off"] or aug_factor <= 0:
        return X_raw_train, y_train

    rotate_y = mode in ["rot_y", "rotate_y", "rot_yz", "rot_y_rot_z", "full"]
    rotate_z = mode in ["rot_z", "rotate_z", "rot_yz", "rot_y_rot_z"]
    scale_trans = (mode == "full")
    noise = (mode == "full")
    time_warp = (mode == "full")

    rng = np.random.default_rng(seed)

    augmented_X = [X_raw_train]
    augmented_y = [y_train]

    N = len(X_raw_train)
    print(f"  [AUGMENT] Chế độ: '{augment_mode}' (rot_y={rotate_y}, rot_z={rotate_z}, scale_trans={scale_trans}) | Hệ số K={aug_factor} (sinh {N*aug_factor} mẫu mới)...")

    for k in range(aug_factor):
        sub_X = np.zeros_like(X_raw_train)
        for i in range(N):
            sub_X[i] = augment_single_sequence(
                X_raw_train[i],
                rotate_y=rotate_y,
                rotate_z=rotate_z,
                scale_trans=scale_trans,
                noise=noise,
                time_warp=time_warp,
                target_duration=target_duration,
                rng=rng
            )
        augmented_X.append(sub_X)
        augmented_y.append(y_train)

    final_X = np.concatenate(augmented_X, axis=0)
    final_y = np.concatenate(augmented_y, axis=0)

    # Xáo trộn ngẫu nhiên tập train sau khi augment
    perm = rng.permutation(len(final_X))
    return final_X[perm], final_y[perm]
