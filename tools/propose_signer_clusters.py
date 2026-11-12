"""
tools/propose_signer_clusters.py
================================
Phân tích mtime và hình thể khung xương (pose proportions) trong session_03
để đề xuất phân cụm người ký (signer_A, signer_B, signer_C).
"""
import os
import glob
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

RAW_DIR = "dataset/raw"

def analyze_mtime():
    classes = sorted(os.listdir(RAW_DIR))
    class_times = []
    for c in classes:
        p = os.path.join(RAW_DIR, c, "session_03")
        if not os.path.exists(p):
            continue
        files = glob.glob(os.path.join(p, "*.npy"))
        if not files:
            continue
        mtimes = [os.path.getmtime(f) for f in files]
        min_t = min(mtimes)
        max_t = max(mtimes)
        class_times.append({
            'class': c,
            'min_mtime': min_t,
            'max_mtime': max_t,
            'str_start': datetime.fromtimestamp(min_t).strftime('%Y-%m-%d %H:%M:%S'),
            'str_end': datetime.fromtimestamp(max_t).strftime('%Y-%m-%d %H:%M:%S'),
            'count': len(files)
        })
    df = pd.DataFrame(class_times).sort_values('min_mtime').reset_index(drop=True)
    
    # Nhóm theo khe hở > 10 phút (600 giây)
    groups = []
    curr_group = [df.iloc[0]['class']]
    for i in range(1, len(df)):
        gap = df.iloc[i]['min_mtime'] - df.iloc[i-1]['max_mtime']
        if gap > 600:
            groups.append(curr_group)
            curr_group = [df.iloc[i]['class']]
        else:
            curr_group.append(df.iloc[i]['class'])
    if curr_group:
        groups.append(curr_group)
        
    return df, groups

def extract_body_ratios(sample_1662):
    """
    Tính các tỉ lệ hình thể từ 132 giá trị pose (33 landmark x 4).
    Chỉ dùng các frame có visibility vai > 0.5.
    """
    # pose: shape (30, 132) -> reshape (30, 33, 4)
    pose = sample_1662[:, :132].reshape(-1, 33, 4)
    
    ratios = []
    for frame in pose:
        # idx 11: L_shoulder, 12: R_shoulder
        ls, rs = frame[11], frame[12]
        # idx 13: L_elbow, 14: R_elbow
        le, re = frame[13], frame[14]
        # idx 15: L_wrist, 16: R_wrist
        lw, rw = frame[15], frame[16]
        # idx 23: L_hip, 24: R_hip
        lh, rh = frame[23], frame[24]
        
        # Kiểm tra visibility vai
        if ls[3] < 0.5 or rs[3] < 0.5:
            continue
            
        shoulder_width = np.linalg.norm(ls[:3] - rs[:3])
        if shoulder_width < 1e-4:
            continue
            
        row = []
        # 1. Cánh tay trên / vai
        if le[3] >= 0.5:
            row.append(np.linalg.norm(ls[:3] - le[:3]) / shoulder_width)
        else:
            row.append(np.nan)
            
        if re[3] >= 0.5:
            row.append(np.linalg.norm(rs[:3] - re[:3]) / shoulder_width)
        else:
            row.append(np.nan)
            
        # 2. Cẳng tay / vai
        if le[3] >= 0.5 and lw[3] >= 0.5:
            row.append(np.linalg.norm(le[:3] - lw[:3]) / shoulder_width)
        else:
            row.append(np.nan)
            
        if re[3] >= 0.5 and rw[3] >= 0.5:
            row.append(np.linalg.norm(re[:3] - rw[:3]) / shoulder_width)
        else:
            row.append(np.nan)
            
        # 3. Vai / Thân
        if lh[3] >= 0.5 and rh[3] >= 0.5:
            mid_shoulder = (ls[:3] + rs[:3]) / 2.0
            mid_hip = (lh[:3] + rh[:3]) / 2.0
            torso_length = np.linalg.norm(mid_shoulder - mid_hip)
            if torso_length > 1e-4:
                row.append(shoulder_width / torso_length)
            else:
                row.append(np.nan)
        else:
            row.append(np.nan)
            
        ratios.append(row)
        
    if not ratios:
        return None
    ratios = np.array(ratios)
    # Lấy median theo frame (bỏ qua NaN)
    sample_median = np.nanmedian(ratios, axis=0)
    return sample_median

def main():
    print("=" * 80)
    print("1. PHÂN TÍCH METADATA FILE (MTIME) TRONG SESSION_03")
    print("=" * 80)
    df_mtime, groups = analyze_mtime()
    print(f"Tổng số class trong session_03: {len(df_mtime)}")
    print(f"Số nhóm ghi hình (khe hở > 10 phút): {len(groups)}")
    for idx, grp in enumerate(groups, 1):
        grp_df = df_mtime[df_mtime['class'].isin(grp)]
        t_start = grp_df['str_start'].min()
        t_end = grp_df['str_end'].max()
        print(f"Nhóm {idx} ({len(grp)} class): {t_start} -> {t_end}")
        print(f"  Các class: {', '.join(grp)}")

    print("\n" + "=" * 80)
    print("2. PHÂN TÍCH ĐẶC TRƯNG HÌNH THỂ POSE TRONG SESSION_03")
    print("=" * 80)
    
    classes = sorted(os.listdir(RAW_DIR))
    class_features = {}
    
    for c in classes:
        p = os.path.join(RAW_DIR, c, "session_03")
        if not os.path.exists(p):
            continue
        files = glob.glob(os.path.join(p, "*.npy"))
        sample_medians = []
        for f in files:
            arr = np.load(f)
            r = extract_body_ratios(arr)
            if r is not None:
                sample_medians.append(r)
        if sample_medians:
            class_features[c] = np.nanmedian(sample_medians, axis=0)
            
    df_feat = pd.DataFrame.from_dict(class_features, orient='index', 
                                    columns=['upper_arm_l', 'upper_arm_r', 'forearm_l', 'forearm_r', 'shoulder_torso'])
    
    # Kiểm tra cột có NaN nhiều không
    print("Thông tin đặc trưng hình thể:")
    print(df_feat.describe().to_string())
    
    # Điền NaN nếu có (thường shoulder_torso có thể thiếu nếu hông ngoài khung)
    df_clean = df_feat.fillna(df_feat.median())
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_clean)
    
    # Phân cụm KMeans k=3
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=20)
    labels = kmeans.fit_predict(X_scaled)
    sil_score = silhouette_score(X_scaled, labels)
    
    print("\n" + "=" * 80)
    print(f"KẾT QUẢ PHÂN CỤM (K=3): Silhouette Score = {sil_score:.4f}")
    print("=" * 80)
    
    df_clean['cluster'] = labels
    counts = df_clean['cluster'].value_counts()
    print("Kích thước từng cụm:")
    for k, v in counts.items():
        print(f"  Cụm {k}: {v} class (~{v*30} mẫu)")
        
    # So sánh với kỳ vọng: ~32 class (signer_A), ~8 class (signer_B), ~3 class (signer_C)
    print("\nĐối chiếu kỳ vọng: signer_A ~32 class (~960 mẫu), signer_B ~8 class (~240 mẫu), signer_C ~3 class (~90 mẫu)")
    
    # Phán đoán chất lượng cụm
    is_strong = (sil_score >= 0.25) and (len(counts) == 3)
    
    # Tạo proposed mapping
    os.makedirs("incoming", exist_ok=True)
    csv_rows = []
    
    # Sắp xếp cụm theo kích thước giảm dần
    cluster_order = counts.index.tolist()
    signer_map_by_cluster = {
        cluster_order[0]: "signer_A",
        cluster_order[1]: "signer_B" if len(cluster_order) > 1 else "unknown",
        cluster_order[2]: "signer_C" if len(cluster_order) > 2 else "unknown"
    }
    
    for c in sorted(classes):
        if c in df_clean.index:
            cl = df_clean.loc[c, 'cluster']
            prop = signer_map_by_cluster[cl]
            conf = f"{sil_score:.2f}"
            evid = f"KMeans_k3_cluster_{cl} (size={counts[cl]}), sil={sil_score:.2f}"
            if not is_strong:
                prop = "unknown"
                evid = f"KHONG_DU_BANG_CHUNG: sil={sil_score:.2f} < 0.25 hoac cluster mismatch"
            csv_rows.append({
                'class': c,
                'session': 'session_03',
                'signer_proposed': prop,
                'confidence': conf,
                'evidence': evid
            })
            
    df_out = pd.DataFrame(csv_rows)
    df_out.to_csv("incoming/signer_map_proposed.csv", index=False, encoding='utf-8')
    print("\nĐã lưu đề xuất vào incoming/signer_map_proposed.csv")
    print("Mẫu 5 dòng đầu:")
    print(df_out.head())

if __name__ == '__main__':
    main()
