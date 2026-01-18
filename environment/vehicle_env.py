import numpy as np
import torch
from config.parameters import Config
from config.paths import Paths
from collections import defaultdict
from pyproj import Transformer, CRS
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import pandas as pd
import random
import os


class Vehicle:
    """
    智能车辆类，代表车路协同系统中的分布式感知节点

    1. 维护车辆自身的状态信息（位置、连接、数据等）
    2. 执行本地模型推理和置信度计算
    3. 管理本地数据缓存
    4. 与基站和边缘服务器进行通信协调
    """

    def __init__(self, vehicle_id, position):
        self.id = vehicle_id  # 车辆唯一标识符
        self.position = position  # 车辆当前位置坐标
        self.bs_connection = None  # 当前连接的基站ID

        self.data_batches = []  # 车辆的实时数据批次，未标注
        self.uploaded_data = []  # 新上传的数据
        self.cache_data = []  # 车辆数据缓存,相当于在边缘服务器上的数据

        self.quality_scores_history = []  # 数据质量评分
        self.confidence_history = []  # 历史置信度记录
        self.test_loss_history = []  # 测试损失记录

    def set_bs_connection(self, bs_id):
        self.bs_connection = bs_id

    def add_data_batch(self, data_batch):
        """添加数据批次"""
        self.data_batches.append(data_batch)

    def set_uploaded_data(self, uploaded_batches):
        """设置新上传的数据"""
        self.uploaded_data = uploaded_batches

    def clear_uploaded_data(self):
        """清空已处理的上传数据"""
        self.uploaded_data = []

    def get_inference_confidence(self, global_model):
        """计算模型在本地数据上的推理置信度"""
        """
        dataloader:为车辆新采集的数据
        """
        if not self.data_batches or len(self.data_batches) == 0:
            return None

        global_model.eval()
        total_confidence = 0.0
        count = 0
        device = next(global_model.parameters()).device
        # 禁用梯度计算以提升效率
        with torch.no_grad():
            for batch in self.data_batches:
                if isinstance(batch, (list, tuple)):
                    inputs, _ = batch  # 忽略标签，只使用输入
                else:
                    inputs = batch

                if not isinstance(inputs, torch.Tensor):
                    inputs = torch.tensor(inputs, dtype=torch.float32)
                inputs = inputs.to(device)

                outputs = global_model(inputs)
                if hasattr(outputs, "logits"):
                    # 某些模型输出包含logits属性
                    outputs = outputs.logits
                # 计算置信度
                probabilities = torch.softmax(outputs, dim=1)  # 转换为概率分布
                batch_confidence = torch.max(probabilities, dim=1)[
                    0
                ]  # 获取每个样本的最大概率
                mean_confidence = batch_confidence.mean().item()  # 批次平均置信度

                total_confidence += mean_confidence
                count += 1

        avg_confidence = total_confidence / count if count > 0 else 0.0

        return avg_confidence

    def calculate_test_loss(self, global_model, gold_model):
        """计算模型在新上传数据上的测试损失"""
        """
        阶段s-1中新上传数据A_v^{s-1}在全局模型\omega_g^{s-1}上的测试损失L_{test,v}^s
        dataloader:为经过计算后车辆上传的数据
        """
        if not self.uploaded_data or len(self.uploaded_data) == 0:
            return None

        global_model.eval()
        gold_model.eval()
        criterion = torch.nn.CrossEntropyLoss()
        device = next(global_model.parameters()).device

        losses = []

        with torch.no_grad():
            for batch in self.uploaded_data:
                # 提取输入数据
                inputs = batch[0] if isinstance(batch, (list, tuple)) else batch
                inputs = inputs.to(device)

                if inputs.size(0) == 0:
                    continue

                # 生成伪标签
                gold_outputs = gold_model.model(inputs)
                targets = gold_outputs.argmax(dim=1)

                # 计算损失
                outputs = global_model(inputs)
                loss = criterion(outputs, targets)
                losses.append(loss.item())

        return np.mean(losses)

    def update_quality_scores(self, cache_manager):
        """从缓存管理器更新质量评分"""
        cache = cache_manager.get_vehicle_cache(self.id)
        if cache and "quality_scores" in cache and cache["quality_scores"]:
            # 使用最新的质量评分
            recent_scores = cache["quality_scores"][
                -min(Config.MAX_LOCAL_BATCHES, len(cache["quality_scores"])) :
            ]
            quality_score = np.mean(recent_scores)
            self.quality_scores_history.append(quality_score)
            return quality_score
        return 0  # 默认质量评分


class VehicleEnvironment:
    """
    车辆环境类

    1. 初始化环境。创建车辆和基站，并将车辆连接到最近的基站
    2. 更新车辆位置。模拟车辆移动，更新车辆与基站的连接
    3. 获取环境状态。为DRL智能体提供状态信息，包括每辆车的置信度、测试损失和数据质量评分
    4. 重置环境
    """

    def __init__(self, global_model, gold_model, cache_manager, data_simulator):
        # 实体集合
        self.vehicles = []  # 车辆对象列表
        self.base_stations = []  # 基站对象列表，字典
        self.current_session = 0  # 当前训练会话编号
        self.environment_time = 0.0  # 环境运行时间(s)

        self.trajectory_data = pd.read_csv(os.path.join(Paths.TRAJECTORY_DIR, Config.TRAJECTORY_FILE))
        self.trajectory_points = self._convert_to_cartesian(
            self.trajectory_data['lon'].values,
            self.trajectory_data['lat'].values
        )
        self.trajectory_length = len(self.trajectory_points)
        self.trajectory_index = 0  # 当前轨迹点索引
        self.direction = 1  # 移动方向：1前进，-1后退

        # PPP参数
        self.ppp_radius = 200  # PPP生成半径（米）
        self.ppp_lambda = 0.001  # 单位面积车辆密度（辆/平方米）

        # 初始化数据环境
        self.data_simulator = data_simulator
        self.global_model = global_model
        self.gold_model = gold_model
        self.cache_manager = cache_manager

        # 初始化物理环境
        self._initialize_environment()
        #self.plot_trajectory("./results/trajectory.png")

    def plot_trajectory(self, save_path=None):
        """
        可视化轨迹，主图显示完整轨迹和当前基站，小图显示初始车辆和PPP车辆。
        """
        if len(self.trajectory_points) == 0:
            print("轨迹点为空，无法绘图")
            return

        x, y = self.trajectory_points[:, 0], self.trajectory_points[:, 1]

        # 创建图形
        fig, main_ax = plt.subplots(figsize=(12, 8))

        # 绘制主图：完整轨迹
        main_ax.plot(x, y, '-', linewidth=2, color='tab:blue', alpha=0.7, label='Vehicle Trajectory')
        main_ax.scatter(x[0], y[0], color='green', s=100, label='Start')
        main_ax.scatter(x[-1], y[-1], color='red', s=100, label='End')

        # 绘制当前基站
        for bs in self.base_stations:
            bs_x, bs_y = bs['position']
            main_ax.scatter(bs_x, bs_y, color='black', s=100, marker='^', label='Base Station' if bs == self.base_stations[0] else "")

        # 添加小图：显示初始车辆和PPP车辆
        if self.vehicles:
            inset_ax = fig.add_axes([0.2, 0.6, 0.25, 0.25])  # 小图位置（大图右上角）

            for vehicle in self.vehicles:
                vx, vy = vehicle.position
                color = 'orange' if vehicle.id == 0 else 'purple'
                inset_ax.scatter(vx, vy, color=color, s=100 if vehicle.id == 0 else 80, marker='o', edgecolors='black', linewidth=1.5)
                inset_ax.text(vx, vy, f' {vehicle.id}', fontsize=8, verticalalignment='bottom', horizontalalignment='right')

            # 设置小图范围
            vehicle_positions = np.array([v.position for v in self.vehicles])
            min_x, max_x = vehicle_positions[:, 0].min(), vehicle_positions[:, 0].max()
            min_y, max_y = vehicle_positions[:, 1].min(), vehicle_positions[:, 1].max()
            margin = 50
            inset_ax.set_xlim(min_x - margin, max_x + margin)
            inset_ax.set_ylim(min_y - margin, max_y + margin)

            # 小图样式
            inset_ax.set_title('Vehicles', fontsize=10)
            inset_ax.axis('equal')
            inset_ax.tick_params(labelsize=8)
            inset_ax.grid(True, linestyle='--', alpha=0.5)

        # 添加新子图：显示主车辆前后各2.5km轨迹和连接的基站
        if self.vehicles and len(self.vehicles) > 0:
            main_vehicle = self.vehicles[0]  # 主车辆（ID=0）
            main_pos = main_vehicle.position
            distance_each_direction = 2500  # 前后各2.5km = 2500米
            current_idx = self.trajectory_index
            
            # 创建新子图（在当前子图下方，增大y轴方向距离）
            local_ax = fig.add_axes([0.2, 0.25, 0.25, 0.25])  # 位于第一个小图下方，距离更大
            
            # 沿轨迹向前（索引减小方向）查找2.5km的起始索引
            forward_distance = 0.0
            start_idx = current_idx
            for i in range(current_idx - 1, -1, -1):
                segment_dist = np.linalg.norm(self.trajectory_points[i+1] - self.trajectory_points[i])
                forward_distance += segment_dist
                if forward_distance > distance_each_direction:
                    start_idx = i + 1  # i+1是最后一个不超过距离的点
                    break
                start_idx = i
            
            # 沿轨迹向后（索引增大方向）查找2.5km的结束索引
            backward_distance = 0.0
            end_idx = current_idx
            for i in range(current_idx, self.trajectory_length - 1):
                segment_dist = np.linalg.norm(self.trajectory_points[i+1] - self.trajectory_points[i])
                backward_distance += segment_dist
                if backward_distance > distance_each_direction:
                    end_idx = i + 1  # i+1是第一个超过距离的点
                    break
                end_idx = i + 1
            
            # 获取轨迹段
            trajectory_segment = self.trajectory_points[start_idx:end_idx+1]
            
            # 绘制轨迹段
            if len(trajectory_segment) > 0:
                local_ax.plot(trajectory_segment[:, 0], trajectory_segment[:, 1], 
                             '-', linewidth=2, color='tab:blue', alpha=0.7, label='Trajectory (±2.5km)')
            
            # 绘制主车辆位置
            local_ax.scatter(main_pos[0], main_pos[1], color='orange', s=150, 
                           marker='o', edgecolors='black', linewidth=2, label='Main Vehicle', zorder=5)
            local_ax.text(main_pos[0], main_pos[1], ' Vehicle', fontsize=9, 
                         verticalalignment='center', horizontalalignment='left', fontweight='bold')
            
            # 绘制连接的基站
            if main_vehicle.bs_connection is not None:
                connected_bs = self._get_base_station_by_id(main_vehicle.bs_connection)
                if connected_bs:
                    bs_pos = connected_bs['position']
                    local_ax.scatter(bs_pos[0], bs_pos[1], color='red', s=150, 
                                   marker='^', edgecolors='black', linewidth=2, label='Connected BS', zorder=5)
                    local_ax.text(bs_pos[0], bs_pos[1], f' BS{connected_bs["id"]}', 
                                 fontsize=9, verticalalignment='bottom', horizontalalignment='left', fontweight='bold')
                    
                    # 绘制连接线
                    local_ax.plot([main_pos[0], bs_pos[0]], [main_pos[1], bs_pos[1]], 
                                '--', color='gray', linewidth=1.5, alpha=0.6, label='Connection')
            
            # 计算轨迹段的边界来确定显示范围
            if len(trajectory_segment) > 0:
                min_x, max_x = trajectory_segment[:, 0].min(), trajectory_segment[:, 0].max()
                min_y, max_y = trajectory_segment[:, 1].min(), trajectory_segment[:, 1].max()
                # 添加一些边距
                margin_ratio = 0.1
                x_range = max_x - min_x
                y_range = max_y - min_y
                margin_x = max(x_range * margin_ratio, 500)  # 至少500米
                margin_y = max(y_range * margin_ratio, 500)
                
                local_ax.set_xlim(min_x - margin_x, max_x + margin_x)
                local_ax.set_ylim(min_y - margin_y, max_y + margin_y)
            else:
                # 如果没有轨迹段，使用车辆位置为中心
                radius_5km = 5000
                local_ax.set_xlim(main_pos[0] - radius_5km, main_pos[0] + radius_5km)
                local_ax.set_ylim(main_pos[1] - radius_5km, main_pos[1] + radius_5km)
            
            # 子图样式
            local_ax.set_title('Vehicle & connected BS', fontsize=10)
            local_ax.set_xlabel("X (km)", fontsize=8)
            local_ax.set_ylabel("Y (km)", fontsize=8)
            
            # 将坐标轴刻度转换为km（使用Formatter避免警告）
            def format_km(x, pos):
                return f'{x/1000:.2f}'
            
            local_ax.xaxis.set_major_formatter(FuncFormatter(format_km))
            local_ax.yaxis.set_major_formatter(FuncFormatter(format_km))
            
            local_ax.axis('equal')
            local_ax.tick_params(labelsize=7)
            local_ax.grid(True, linestyle='--', alpha=0.5)
            local_ax.legend(fontsize=7, loc='upper left')

        # 主图设置
        main_ax.set_title("Trajectory with Current Base Stations", fontsize=14)
        main_ax.set_xlabel("X (km)")
        main_ax.set_ylabel("Y (km)")
        
        # 将主图坐标轴刻度转换为km（使用Formatter避免警告）
        def format_km_main(x, pos):
            return f'{x/1000:.1f}'
        
        main_ax.xaxis.set_major_formatter(FuncFormatter(format_km_main))
        main_ax.yaxis.set_major_formatter(FuncFormatter(format_km_main))
        main_ax.axis('equal')
        main_ax.grid(True, linestyle='--', alpha=0.6)
        main_ax.legend()

        # 保存或显示图像
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"轨迹图已保存至: {save_path}")

        plt.close()

    def _initialize_environment(self):
        """初始化车辆和基站环境"""
        # 1. 沿轨迹部署基站（每5km一个）
        self._initialize_base_stations_along_trajectory()
        # 2. 初始化车辆（主车辆+PPP车辆）
        self._initialize_vehicles_with_ppp()
        # 3. 建立连接
        self._establish_initial_connections()

    def _convert_to_cartesian(self, lon_array, lat_array):
        """
        使用 UTM 投影将经纬度转换为平面坐标（单位：米），
        并平移坐标系使得所有点的 x, y ≥ 0。
        """
        center_lon = np.mean(lon_array)
        center_lat = np.mean(lat_array)

        wgs84 = CRS("EPSG:4326")
        utm_crs = CRS(
            proj="utm",
            zone=int((center_lon + 180) // 6) + 1,
            south=center_lat < 0,
            ellps="WGS84"
        )

        transformer = Transformer.from_crs(wgs84, utm_crs, always_xy=True)
        x, y = transformer.transform(lon_array, lat_array)

        # 平移至非负
        return np.column_stack([x - x.min(), y - y.min()])

    def _initialize_base_stations_along_trajectory(self):
        """沿轨迹按 Y 轴均匀划分部署基站，X 坐标取对应轨迹点并东偏 50 米"""
        coverage_radius = Config.BASE_STATION_COVERAGE
        spacing = 10000  # 间距（单位：米）

        y_coords = self.trajectory_points[:, 1]
        y_min, y_max = np.min(y_coords), np.max(y_coords)
        total_y_span = y_max - y_min

        # 估算基站数量（基于 Y 跨度）
        num_bs = min(80, max(2, int(total_y_span / spacing) + 1))
        print(f"初始化 {num_bs} 个基站")

        # 均匀划分 Y 轴，并反转顺序
        target_ys = np.linspace(y_min, y_max, num_bs)[::-1]

        for i, target_y in enumerate(target_ys):
            # 找到轨迹中 Y 最接近 target_y 的点的索引
            idx = np.argmin(np.abs(y_coords - target_y))

            # 获取该点的 X，并向东偏移 50 米
            x_on_road = self.trajectory_points[idx, 0]
            bs_position = np.array([x_on_road + 500.0, target_y])

            self.base_stations.append({
                "id": i,
                "position": bs_position,
                "coverage": coverage_radius,
                "capacity": 50,
                "connected_vehicles": [],
                "utilization": 0.0,
            })

    def _initialize_vehicles_with_ppp(self):
        """初始化车辆：主车辆 + PPP生成的其他车辆"""
        print(f"初始化 {Config.NUM_VEHICLES} 辆智能车辆")

        # 1. 主车辆（ID 0）在轨迹起点
        main_vehicle = Vehicle(
            vehicle_id=0,
            position=self.trajectory_points[0]
        )
        self.vehicles.append(main_vehicle)

        # 2. PPP生成其他车辆
        self._generate_ppp_vehicles(main_vehicle.position)

    def _generate_ppp_vehicles(self, center_position):
        """泊松点过程生成车辆"""
        ppp_count = Config.NUM_VEHICLES - 1
        angles = np.random.uniform(0, 2 * np.pi, size=ppp_count)
        radii = np.random.uniform(0, self.ppp_radius, size=ppp_count)

        # 转换为笛卡尔偏移
        offsets = np.stack([
            radii * np.cos(angles),
            radii * np.sin(angles)
        ], axis=1)  # shape: (ppp_count, 2)

        # 所有位置 = 中心 + 偏移
        positions = center_position + offsets

        # 批量创建车辆
        for i in range(ppp_count):
            vehicle = Vehicle(
                vehicle_id=i + 1,
                position=positions[i]
            )
            self.vehicles.append(vehicle)

    def _establish_initial_connections(self):
        """建立车辆与基站的初始连接"""
        connection_success_count = 0

        for vehicle in self.vehicles:
            nearest_bs = self._find_nearest_base_station(vehicle.position)
            if nearest_bs:
                vehicle.set_bs_connection(nearest_bs["id"])
                nearest_bs["connected_vehicles"].append(vehicle.id)
                connection_success_count += 1
            else:
                vehicle.set_bs_connection(None)

        print(f"初始连接: {connection_success_count}/{len(self.vehicles)} 车辆成功连接")

    def _find_nearest_base_station(self, position):
        """
        找到距离指定位置最近的可用基站
        input:
            position: 车辆位置坐标 [x, y]
        return:
            dict: 最近的基站信息，如果没有可用基站则返回None
        """
        available_bs = [
            bs for bs in self.base_stations
            if np.linalg.norm(position - bs["position"]) <= bs["coverage"]
            and len(bs["connected_vehicles"]) < bs["capacity"]
        ]
        if not available_bs:
            return None

        return min(available_bs, key=lambda bs: np.linalg.norm(position - bs["position"]))

    def update_vehicle_positions(self, time_delta=1.0):
        """更新车辆位置：主车沿轨迹移动，PPP车辆在主车周围重新生成"""
        # 更新主车位置
        distance_moved = 15.0 * time_delta * self.direction  # 根据方向调整距离

        if distance_moved != 0:
            self.trajectory_index = self._find_next_trajectory_index(distance_moved)
            self.vehicles[0].position = self.trajectory_points[self.trajectory_index]

        # 检查是否到达边界并翻转方向
        if self.trajectory_index == 0 and self.direction == -1:
            self.direction = 1  # 到达起点，转为前进
        elif self.trajectory_index == self.trajectory_length - 1 and self.direction == 1:
            self.direction = -1  # 到达终点，转为后退

        # 重新生成PPP车辆并更新连接
        self.vehicles = [self.vehicles[0]]  # 保留主车
        self._generate_ppp_vehicles(self.vehicles[0].position)
        self._update_vehicle_connections()

        self.environment_time += time_delta

    def _find_next_trajectory_index(self, distance):
        """找到移动指定距离后的轨迹点索引，支持前进和后退"""
        dist_accumulated = 0.0
        
        if distance > 0:  # 前进
            for i in range(self.trajectory_index, self.trajectory_length - 1):
                segment_dist = np.linalg.norm(self.trajectory_points[i+1] - self.trajectory_points[i])
                dist_accumulated += segment_dist
                if dist_accumulated >= distance:
                    return i + 1
            return self.trajectory_length - 1  # 到达终点
        elif distance < 0:  # 后退
            abs_distance = abs(distance)
            for i in range(self.trajectory_index, 0, -1):
                segment_dist = np.linalg.norm(self.trajectory_points[i] - self.trajectory_points[i-1])
                dist_accumulated += segment_dist
                if dist_accumulated >= abs_distance:
                    return i - 1
            return 0  # 到达起点
        else:
            return self.trajectory_index  # 距离为0，不移动

    def _update_vehicle_connections(self):
        """更新所有车辆的基站连接"""
        for vehicle in self.vehicles:
            # 断开旧连接
            old_bs = self._get_base_station_by_id(vehicle.bs_connection)
            if old_bs and vehicle.id in old_bs["connected_vehicles"]:
                old_bs["connected_vehicles"].remove(vehicle.id)
                print(f"Vehicle {vehicle.id} disconnected from Base Station {old_bs['id']}")

            # 建立新连接
            nearest_bs = self._find_nearest_base_station(vehicle.position)
            if nearest_bs and len(nearest_bs["connected_vehicles"]) < nearest_bs["capacity"]:
                vehicle.set_bs_connection(nearest_bs["id"])
                nearest_bs["connected_vehicles"].append(vehicle.id)
                print(f"Vehicle {vehicle.id} connected to Base Station {nearest_bs['id']}")
            else:
                # 调试信息：打印车辆位置和最近基站信息
                print(f"Vehicle {vehicle.id} at position {vehicle.position} could not connect to any Base Station")
                if nearest_bs:
                    print(f"  Nearest BS {nearest_bs['id']} at {nearest_bs['position']}, distance: {np.linalg.norm(vehicle.position - nearest_bs['position'])}, capacity: {len(nearest_bs['connected_vehicles'])}/{nearest_bs['capacity']}")
                else:
                    # 计算到所有基站的距离
                    distances = [np.linalg.norm(vehicle.position - bs["position"]) for bs in self.base_stations]
                    min_dist = min(distances) if distances else float('inf')
                    print(f"  No available BS within range. Min distance to any BS: {min_dist} meters")
                vehicle.set_bs_connection(None)

    def _get_base_station_by_id(self, bs_id):
        """
        根据ID获取基站对象
        """
        for bs in self.base_stations:
            if bs["id"] == bs_id:
                return bs
        return None

    def _get_vehicle_by_id(self, vehicle_id):
        """
        根据ID获取车辆
        """
        for vehicle in self.vehicles:
            if vehicle.id == vehicle_id:
                return vehicle
        return None

    def get_environment_state(self):
        """获取真实的环境状态用于DRL"""
        state = []

        for vehicle in self.vehicles:
            try:
                # 获取当前置信度，从全局模型推理中得到
                if vehicle.data_batches:
                    confidence = vehicle.get_inference_confidence(self.global_model)
                else:
                    confidence = 0.5
                vehicle.confidence_history.append(confidence)

                # 2. 获取测试损失 - 从实际模型评估中获取
                if vehicle.uploaded_data:
                    raw_loss = vehicle.calculate_test_loss(
                        self.global_model, self.gold_model
                    )
                    test_loss = 1.0 / (1.0 + raw_loss)
                else:
                    test_loss = 1.0
                vehicle.test_loss_history.append(test_loss)

                # 3. 获取质量评分 - 从缓存管理器中获取
                quality_score = vehicle.update_quality_scores(self.cache_manager)

                # 添加到状态向量
                state.extend([confidence, test_loss, quality_score])

            except Exception as e:
                print(f"Error getting state for vehicle {vehicle.id}: {e}")
                state.extend([0.5, 1.0, 0])

        return np.array(state, dtype=np.float32)

    def reset(self):
        """重置环境"""
        print("重置车辆环境...")
        self.vehicles = []
        self.base_stations = []
        self.trajectory_index = 0
        self.direction = 1  # 重置为前进方向
        self.environment_time = 0.0
        self._initialize_environment()
        print("环境重置完成")

if __name__ == "__main__":
    """测试车辆与基站的连接切换及来回移动"""
    env = VehicleEnvironment(None, None, None, None)
    
    # 测试多次位置更新，观察方向变化和连接
    print("=== 初始状态 ===")
    print(f"轨迹长度: {env.trajectory_length}")
    print(f"初始轨迹索引: {env.trajectory_index}, 方向: {'前进' if env.direction == 1 else '后退'}")
    print(f"主车辆位置: {env.vehicles[0].position}")
    print(f"主车辆连接基站: {env.vehicles[0].bs_connection}")
    
    # 多次更新位置
    updates = [
        (750, "第一次更新"),
        (750, "第二次更新"),
        (750, "第三次更新"),
        (10750, "第四次更新"),
        (750, "第五次更新"),
        (750, "第六次更新"),
        (750, "第七次更新"),
        (10750, "第八次更新"),
        (750, "第九次更新（应开始前进）"),
        (750, "第十次更新（应开始后退）"),
        (750, "第十一次更新（应开始后退）"),
        (750, "第十二次更新（应开始后退）"),
    ]
    
    for time_delta, description in updates:
        print(f"\n=== {description} (time_delta={time_delta}) ===")
        env.update_vehicle_positions(time_delta=time_delta)
        print(f"轨迹索引: {env.trajectory_index}, 方向: {'前进' if env.direction == 1 else '后退'}")
        print(f"主车辆位置: {env.vehicles[0].position}")
        print(f"主车辆连接基站: {env.vehicles[0].bs_connection}")
        
        # 检查边界
        if env.trajectory_index == 0:
            print("✓ 已到达起点")
        elif env.trajectory_index == env.trajectory_length - 1:
            print("✓ 已到达终点")
    
    # 最终绘图
    env.plot_trajectory(save_path="./results/trajectory.png")
    print("\n轨迹图已保存至 ./results/trajectory.png")