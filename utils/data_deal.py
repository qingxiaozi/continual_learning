import pandas as pd
import ast
import numpy as np
import matplotlib.pyplot as plt
import os
from geopy.distance import great_circle  # 或使用 geodesic

# 读取数据
df = pd.read_csv('./data/PortoTaxi/train.csv')

def is_valid_traj(polyline_str):
    try:
        pts = ast.literal_eval(polyline_str)
    except:
        return False, [], 0

    if len(pts) < 2:
        return False, [], 0

    # 起点终点不能相同
    if np.allclose(pts[0], pts[-1], atol=1e-5):
        return False, [], 0

    # 检查是否有异常跳跃并使用geopy计算距离
    for i in range(1, len(pts)):
        dist = great_circle(pts[i-1][::-1], pts[i][::-1]).km  # geopy需要(纬度,经度)
        if dist > 1:  # 相邻点距离>2km为异常
            return False, [], 0

    # 计算路径总长度
    total_length = sum(great_circle(pts[i-1][::-1], pts[i][::-1]).km for i in range(1, len(pts)))

    # 路径总长度大于100km
    if total_length <= 100:
        return False, [], total_length

    return True, pts, total_length

# 提取有效轨迹，同时保存长度信息
trajectory_data = []
for _, row in df.iterrows():
    valid, pts, length = is_valid_traj(row['POLYLINE'])
    if valid:
        trajectory_data.append({
            'TRIP_ID': row['TRIP_ID'],
            'POLYLINE': row['POLYLINE'],
            'LENGTH_KM': length
        })
        if len(trajectory_data) >= 2000:
            break

# 保存CSV文件（只包含TRIP_ID和POLYLINE）
trajectory_df = pd.DataFrame(trajectory_data)
trajectory_df[['TRIP_ID', 'POLYLINE']].to_csv('./data/PortoTaxi/trajectory.csv', index=False)

# 随机选20条绘图
os.makedirs('results', exist_ok=True)
sampled = trajectory_df.sample(n=min(20, len(trajectory_df)), random_state=42)

for i, row in sampled.iterrows():
    pts = ast.literal_eval(row['POLYLINE'])
    xs, ys = zip(*pts)
    plt.figure(figsize=(6, 6))
    plt.plot(xs, ys, marker='o', markersize=2)
    plt.title(f"Trip {row['TRIP_ID']}\nLength: {row['LENGTH_KM']:.1f}km")
    plt.axis('equal')
    plt.savefig(f'./results/traj_{i+1}.png', dpi=150, bbox_inches='tight')
    plt.close()

print("Done: trajectory.csv saved and 20 plots generated in 'results/' folder.")