import pandas as pd
import ast
import numpy as np
import matplotlib.pyplot as plt
import os
from geopy.distance import great_circle  # 或使用 geodesic

# # 读取数据
# df = pd.read_csv('./data/PortoTaxi/train.csv')

# def is_valid_traj(polyline_str):
#     try:
#         pts = ast.literal_eval(polyline_str)
#     except:
#         return False, [], 0

#     if len(pts) < 2:
#         return False, [], 0

#     # 起点终点不能相同
#     if np.allclose(pts[0], pts[-1], atol=1e-5):
#         return False, [], 0

#     # 检查是否有异常跳跃并使用geopy计算距离
#     for i in range(1, len(pts)):
#         dist = great_circle(pts[i-1][::-1], pts[i][::-1]).km  # geopy需要(纬度,经度)
#         if dist > 1:  # 相邻点距离>2km为异常
#             return False, [], 0

#     # 计算路径总长度
#     total_length = sum(great_circle(pts[i-1][::-1], pts[i][::-1]).km for i in range(1, len(pts)))

#     # 路径总长度大于100km
#     if total_length <= 100:
#         return False, [], total_length

#     return True, pts, total_length

# # 提取有效轨迹，同时保存长度信息
# trajectory_data = []
# for _, row in df.iterrows():
#     valid, pts, length = is_valid_traj(row['POLYLINE'])
#     if valid:
#         trajectory_data.append({
#             'TRIP_ID': row['TRIP_ID'],
#             'POLYLINE': row['POLYLINE'],
#             'LENGTH_KM': length
#         })
#         if len(trajectory_data) >= 2000:
#             break

# # 保存CSV文件（只包含TRIP_ID和POLYLINE）
# trajectory_df = pd.DataFrame(trajectory_data)
# trajectory_df[['TRIP_ID', 'POLYLINE']].to_csv('./data/PortoTaxi/trajectory.csv', index=False)

# # 随机选20条绘图
# os.makedirs('results', exist_ok=True)
# sampled = trajectory_df.sample(n=min(20, len(trajectory_df)), random_state=42)

# for i, row in sampled.iterrows():
#     pts = ast.literal_eval(row['POLYLINE'])
#     xs, ys = zip(*pts)
#     plt.figure(figsize=(6, 6))
#     plt.plot(xs, ys, marker='o', markersize=2)
#     plt.title(f"Trip {row['TRIP_ID']}\nLength: {row['LENGTH_KM']:.1f}km")
#     plt.axis('equal')
#     plt.savefig(f'./results/png/traj_{i+1}.png', dpi=150, bbox_inches='tight')
#     plt.close()

# print("Done: trajectory.csv saved and 20 plots generated in 'results/' folder.")


def split_trajectories_to_segments(
    input_csv: str = './data/PortoTaxi/trajectory.csv',
    train_csv: str = './data/PortoTaxi/train_trajectory.csv',
    test_csv: str = './data/PortoTaxi/test_trajectory.csv',
    first_segment_km: float = 101.0,
    other_segment_km: float = 105.0,
):
    """将每条轨迹分割：第一段100km写入训练文件，其余按105km分割写入测试文件。"""

    df_all = pd.read_csv(input_csv)
    train_rows = []
    test_rows = []

    for _, row in df_all.iterrows():
        try:
            pts = ast.literal_eval(row['POLYLINE'])
        except Exception:
            continue

        if len(pts) < 2:
            continue

        # 计算轨迹总长度
        total_length = sum(great_circle(pts[i-1][::-1], pts[i][::-1]).km for i in range(1, len(pts)))
        if total_length < first_segment_km:
            continue  # 跳过短轨迹

        # 第一段：100km
        first_pts = [pts[0]]
        first_length = 0.0
        remaining_pts = pts[1:]

        for p in remaining_pts:
            d = great_circle(first_pts[-1][::-1], p[::-1]).km
            if first_length + d <= first_segment_km:
                first_pts.append(p)
                first_length += d
            else:
                break  # 停止第一段

        # 检查第一段长度
        if len(first_pts) < 2 or first_length < 99:  # 过滤太短的第一段
            continue

        # 剩余点
        remaining_pts = pts[len(first_pts)-1:]  # 从第一段最后一个点开始

        # 对剩余部分按105km分割
        segments = []
        if remaining_pts:
            current_pts = [remaining_pts[0]]
            current_length = 0.0
            for p in remaining_pts[1:]:
                d = great_circle(current_pts[-1][::-1], p[::-1]).km
                if current_length + d <= other_segment_km:
                    current_pts.append(p)
                    current_length += d
                else:
                    if len(current_pts) >= 2:
                        seg_length = sum(great_circle(current_pts[i-1][::-1], current_pts[i][::-1]).km for i in range(1, len(current_pts)))
                        if seg_length >= 100.0:
                            segments.append(current_pts)
                    current_pts = [current_pts[-1], p]
                    current_length = d
            # 最后一段
            if len(current_pts) >= 2:
                seg_length = sum(great_circle(current_pts[i-1][::-1], current_pts[i][::-1]).km for i in range(1, len(current_pts)))
                if seg_length >= 100.0:
                    segments.append(current_pts)

        # 第一段给训练
        train_rows.append({'TRIP_ID': row['TRIP_ID'], 'POLYLINE': str(first_pts)})
        # 其余段给测试
        for seg in segments:
            test_rows.append({'TRIP_ID': row['TRIP_ID'], 'POLYLINE': str(seg)})

    pd.DataFrame(train_rows)[['TRIP_ID', 'POLYLINE']].to_csv(train_csv, index=False)
    pd.DataFrame(test_rows)[['TRIP_ID', 'POLYLINE']].to_csv(test_csv, index=False)


if __name__ == '__main__':
    split_trajectories_to_segments()
    print('Split trajectories into train_trajectory.csv and test_trajectory.csv')
