import numpy as np
import torch
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.parameters import config
from environment.dataSimu_env import DomainIncrementalDataSimulator
import random
import matplotlib.pyplot as plt
from collections import defaultdict


class Vehicle:
    """
    智能车辆类，代表车路协同系统中的分布式感知节点

    1. 维护车辆自身的状态信息（位置、连接、数据等）
    2. 执行本地模型推理和置信度计算
    3. 管理本地数据缓存
    4. 与基站和边缘服务器进行通信协调
    """

    def __init__(self, vehicle_id, position):
        # 基本信息
        self.id = vehicle_id  # 车辆唯一标识符
        self.position = position  # 车辆当前位置坐标，用于确定与哪个基站连接，计算通信质量，模拟车辆运动
        # 通信状态
        self.bs_connection = None  # 当前连接的基站ID
        self.communication_quality = 1.0  # 通信质量因子 [0, 1]
        # 数据管理
        self.data_batches = []  # 车辆采集的实时数据批次，未标注
        self.cache_data = []  # 1
        self.data_quality_scores = []  # 数据质量评分
        # 模型状态
        self.local_model = None
        self.model_version = 0  # 当前模型版本
        # 性能监控
        self.confidence_history = []  # 历史置信度记录
        # 资源约束
        self.computation_capacity = 1.0  # 计算能力因子
        self.storage_capacity = 1000  # 存储容量（数据批次数）
        # 状态标志
        self.is_online = True

    def set_bs_connection(self, bs_id):
        old_bs = self.bs_connection
        self.bs_connection = bs_id
        # 记录基站切换事件
        if old_bs != bs_id:
            print(f"Vehicle {self.id}: BS connection changed from {old_bs} to {bs_id}")

    def add_data_batch(self, data_batch):
        """添加数据批次
        data_batch：数据批次，DataLoader对象
        """
        self.data_batches.append(data_batch)
        # 缓存管理逻辑（可选）：
        if len(self.data_batches) > config.MAX_LOCAL_BATCHES:
            # 移除最旧的数据批次（FIFO策略）
            self.data_batches.pop(0)

    def get_inference_confidence(self, model, data_loader):
        """计算模型在本地数据上的推理置信度
        1. 对于每个样本，模型输出一个向量logits，通过softmax函数转换为概率分布
        2. 取概率分布中最大的概率值作为该样本的置信度
        3. 对于一个批次的数据，计算所有样本置信度的平均值作为该批次的平均置信度
        """
        if not data_loader:
            return 0.0

        model.eval()
        total_confidence = 0.0
        count = 0
        # 禁用梯度计算以提升效率
        with torch.no_grad():
            for batch in data_loader:
                # 数据预处理：提取输入
                if isinstance(batch, (list, tuple)):
                    inputs, _ = batch  # 忽略标签，只使用输入
                else:
                    inputs = batch

                # 前向传播获取模型输出
                outputs = model(inputs)

                # 处理不同类型的模型输出
                if hasattr(outputs, "logits"):
                    # 某些模型（如 transformers）输出包含logits属性
                    outputs = outputs.logits
                # 计算置信度
                probabilities = torch.softmax(outputs, dim=1)  # 转换为概率分布
                batch_confidence = torch.max(probabilities, dim=1)[
                    0
                ]  # 获取每个样本的最大概率
                mean_confidence = batch_confidence.mean().item()  # 批次平均置信度

                total_confidence += mean_confidence
                count += 1
        # 计算整体平均置信度
        avg_confidence = total_confidence / count if count > 0 else 0.0

        # 记录历史置信度
        self.confidence_history.append(avg_confidence)

        # 历史记录管理（保持合理长度）
        if len(self.confidence_history) > config.MAX_CONFIDENCE_HISTORY:
            self.confidence_history.pop(0)

        return avg_confidence


class VehicleEnvironment:
    """
    车辆环境类

    1. 初始化环境。创建车辆和基站，并将车辆连接到最近的基站
    2. 更新车辆位置。模拟车辆移动，更新车辆与基站的连接
    3. 获取环境状态。为DRL智能体提供状态信息，包括每辆车的置信度、测试损失和数据质量评分
    4. 重置环境
    """

    def __init__(self):
        # 实体集合
        self.vehicles = []  # 车辆对象列表
        self.base_stations = []  # 基站对象列表，为基站信息字典
        self.current_session = 0  # 当前训练会话编号
        self.environment_time = 0.0  # 环境运行时间(s)
        self.road_length = 5000  # 道路长度(m）
        self.road_width = 20  # 道路宽度(m），-10到10
        self.num_lanes = 4  # 车道数量

        # 性能统计
        self.session_stats = {
            "total_communications": 0,
            "successful_uploads": 0,
            "connection_changes": 0,
        }

        # 初始化物理环境
        self._initialize_environment()
        # 初始化数据环境
        self.data_simulator = DomainIncrementalDataSimulator()

    def _initialize_environment(self):
        """
        初始化车辆和基站环境
        """
        # 初始化基站网络
        self._initialize_base_stations()
        # 初始化车辆集群
        self._initialize_vehicles()
        # 建立初始连接
        self._establish_initial_connections()
        # 记录初始状态
        self._log_initial_environment()

    def _initialize_base_stations(self):
        coverage_radius = config.BASE_STATION_COVERAGE
        optimal_bs_count = max(3, int(self.road_length / (coverage_radius * 0.8)) + 1)
        num_bs = min(optimal_bs_count, 5)  # 最多5个基站
        print(f"初始化 {num_bs} 个基站，覆盖半径: {coverage_radius}米")

        for i in range(num_bs):
            # 基站位置：沿着道路均匀分布
            bs_x = (
                i * (self.road_length / (num_bs - 1))
                if num_bs > 1
                else self.road_length / 2
            )
            bs_position = np.array([bs_x, -10])  # 假设基站位于道路边缘
            # 基站属性配置
            base_station = {
                "id": i,
                "position": bs_position,
                "coverage": coverage_radius,
                "capacity": 50,  # 最大连接车辆数
                "connected_vehicles": [],  # 当前连接的车辆ID列表
                "utilization": 0.0,  # 利用率
                "signal_strength": 1.0,  # 信号强度因子
            }
            self.base_stations.append(base_station)
            print(f"基站 {i} 创建于位置 {bs_position}")

    def _initialize_vehicles(self):
        print(f"初始化 {config.NUM_VEHICLES}辆智能车辆")
        for i in range(config.NUM_VEHICLES):
            position = self._generate_vehicle_position(i)
            # 创建车辆实例
            vehicle = Vehicle(
                vehicle_id=i,
                position=position,
            )
            self.vehicles.append(vehicle)
            print(f"车辆 {i} 创建于位置 {position}")

    def _generate_vehicle_position(self, vehicle_id):
        """
        为车辆生成初始位置的策略
        """
        # x轴，沿道路长度均匀分布
        road_segments = 5
        segment_length = self.road_length / road_segments
        segment_idx = vehicle_id % road_segments
        base_x = segment_idx * segment_length + np.random.uniform(
            0, segment_length * 0.8
        )

        # y轴，车道分配，模拟真实交通流
        lane_width = self.road_width / self.num_lanes
        lane_centers = [
            -(self.road_width / 2) + lane_width / 2 + i * lane_width
            for i in range(self.num_lanes)
        ]

        # 车辆ID决定初始车道，增加随机性
        preferred_lane = vehicle_id % self.num_lanes
        lane_variation = np.random.randint(-1, 2)  # -1, 0, 1
        actual_lane = max(0, min(self.num_lanes - 1, preferred_lane + lane_variation))
        y_position = lane_centers[actual_lane] + np.random.uniform(-0.5, 0.5)

        return np.array([base_x, y_position])

    def _establish_initial_connections(self):
        """
        建立车辆与基站的初始连接
        """
        connection_success_count = 0
        for vehicle in self.vehicles:
            nearest_bs = self._find_nearest_base_station(vehicle.position)
            if nearest_bs:
                # 检查基站容量
                if len(nearest_bs["connected_vehicles"]) < nearest_bs["capacity"]:
                    vehicle.set_bs_connection(nearest_bs["id"])
                    nearest_bs["connected_vehicles"].append(vehicle.id)
                    connection_success_count += 1

                    # 计算连接质量
                    distance = np.linalg.norm(vehicle.position - nearest_bs["position"])
                    quality = max(0, 1 - distance / nearest_bs["coverage"])
                    vehicle.communication_quality = quality
                else:
                    print(
                        f"警告: 基站 {nearest_bs['id']} 容量已满，车辆 {vehicle.id} 无法连接"
                    )
                    vehicle.set_bs_connection(None)
            else:
                print(f"警告: 车辆 {vehicle.id} 不在任何基站覆盖范围内")
                vehicle.set_bs_connection(None)

        print(
            f"初始连接建立完成: {connection_success_count}/{len(self.vehicles)} 车辆成功连接"
        )

    def _find_nearest_base_station(self, position, check_coverage=True):
        """
        找到距离指定位置最近的可用基站
        input:
            position: 车辆位置坐标 [x, y]
            check_coverage: 是否检查覆盖范围

        return:
            dict: 最近的基站信息，如果没有可用基站则返回None
        """
        min_distance = float("inf")
        nearest_bs = None

        for bs in self.base_stations:
            # 计算欧几里得距离
            distance = np.linalg.norm(position - bs["position"])
            # 检查是否在覆盖范围内
            within_coverage = (not check_coverage) or (distance <= bs["coverage"])
            # 检查基站容量
            has_capacity = len(bs["connected_vehicles"]) < bs["capacity"]

            if within_coverage and has_capacity and distance < min_distance:
                min_distance = distance
                nearest_bs = bs

        return nearest_bs

    def _log_initial_environment(self):
        """
        记录环境的初始状态信息
        """
        # 统计连接情况
        connected_vehicles = sum(
            1 for v in self.vehicles if v.bs_connection is not None
        )
        bs_utilization = []

        for bs in self.base_stations:
            utilization = len(bs["connected_vehicles"]) / bs["capacity"]
            bs_utilization.append(utilization)
            bs["utilization"] = utilization

        print("\n=== 物理环境初始化完成 ===")
        print(f"基站数量: {len(self.base_stations)}")
        print(f"车辆数量: {len(self.vehicles)}")
        print(f"已连接车辆: {connected_vehicles}")
        print(f"基站平均利用率: {np.mean(bs_utilization):.2f}")
        print("=====================\n")

    def update_vehicle_positions(self, time_delta=1.0):
        """
        更新车辆位置
        input:
            time_delta：时间步长(s)
        """
        connection_changes = 0
        for vehicle in self.vehicles:
            # 保持旧位置和连接状态
            old_position = vehicle.position.copy()
            old_bs_connection = vehicle.bs_connection
            # 更新车辆位置
            self._update_single_vehicle_position(vehicle, time_delta)

            # 检查是否需要更新基站连接
            if self._should_update_connection(vehicle, old_position):
                new_bs = self._find_nearest_base_station(vehicle.position)
                self._update_vehicle_connection(vehicle, old_bs_connection, new_bs)
                if old_bs_connection != (new_bs["id"] if new_bs else None):
                    connection_changes += 1

        # 更新环境时间
        self.environment_time += time_delta
        self.session_stats["connection_changes"] += connection_changes
        if connection_changes > 0:
            print(f"位置更新完成：{connection_changes} 个连接发生了变化")

    def _update_single_vehicle_position(self, vehicle, time_delta):
        """
        更新单个车辆的位置
        假设车辆速度在8-20 m/s(29-72 km/h)之间
        """
        base_speed = np.random.uniform(8, 20)
        # 车道偏好：车辆倾向于保持当前车道
        current_lane = self._get_vehicle_lane(vehicle)
        lane_keeping_factor = 0.8  # 80%概率保持车道

        if np.random.random() > lane_keeping_factor:
            # 换道逻辑
            lane_change = np.random.choice([-1, 1])  # 向左或者向右换道
            new_lane = max(0, min(self.num_lanes - 1, current_lane + lane_change))
            target_y = self._get_lane_center(new_lane)
        else:
            # 保持车道，但轻微横向波动
            target_y = self._get_lane_center(current_lane) + np.random.uniform(
                -0.2, 0.2
            )
        # 更新位置
        new_x = vehicle.position[0] + base_speed * time_delta
        # 边界处理：环形道路
        if new_x > self.road_length:
            new_x = new_x - self.road_length
            # 重新随机分配车道当车辆回到起点时
            target_y = self._get_lane_center(np.random.randint(0, self.num_lanes))
        # 平滑更新y坐标（模拟真实的车辆控制）
        current_y = vehicle.position[1]
        smooth_y = current_y + (target_y - current_y) * 0.3  # 平滑因子
        vehicle.position = np.array([new_x, smooth_y])

    def _update_vehicle_connection(self, vehicle, old_bs_id, new_bs):
        """
        更新车辆与基站的连接
        """
        # 从旧基站断开连接
        if old_bs_id is not None:
            old_bs = self._get_base_station_by_id(old_bs_id)
            if old_bs and vehicle.id in old_bs["connected_vehicles"]:
                old_bs["connected_vehicles"].remove(vehicle.id)

        # 连接到新基站
        if new_bs:
            vehicle.set_bs_connection(new_bs["id"])
            # 计算连接质量
            distance = np.linalg.norm(vehicle.position - new_bs["position"])
            quality = max(0.1, 1 - distance / new_bs["coverage"])  # 最低质量0.1
            vehicle.communication_quality = quality
            # 更新基站连接列表
            if vehicle.id not in new_bs["connected_vehicles"]:
                new_bs["connected_vehicles"].append(vehicle.id)
        else:
            # 无可用基站
            vehicle.set_bs_connection(None)
            vehicle.communication_quality = 0.0

    def _get_vehicle_lane(self, vehicle):
        """
        获取车辆当前所在车道编号
        """
        lane_width = self.road_width / self.num_lanes
        lane_centers = [
            -(self.road_width / 2) + lane_width / 2 + i * lane_width
            for i in range(self.num_lanes)
        ]
        # 找到最近的车道中心
        distances = [abs(vehicle.position[1] - center) for center in lane_centers]
        return np.argmin(distances)

    def _get_lane_center(self, lane_index):
        """
        获取指定车道的中心y坐标
        """
        lane_width = self.road_width / self.num_lanes
        return -(self.road_width / 2) + lane_width / 2 + lane_index * lane_width

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

    def _should_update_connection(self, vehicle, old_position):
        """
        判断是否需要更新车辆连接
        """
        if vehicle.bs_connection is None:
            return True  # 当前无连接，需要尝试连接
        # 计算移动距离
        movement = np.linalg.norm(vehicle.position - old_position)
        # 如果移动显著，检查连接
        if movement > 50:  # 移动超过50米
            current_bs = self._get_base_station_by_id(vehicle.bs_connection)
            if current_bs:
                current_distance = np.linalg.norm(
                    vehicle.position - current_bs["position"]
                )
                # 如果距离超过覆盖范围的80%，考虑切换
                return current_distance > current_bs["coverage"] * 0.8

        return False

    def reset(self):
        """
        重置环境到初始状态
        """
        print("重置车辆环境...")
        # 重置状态变量
        self.current_session = 0
        self.environment_time = 0.0
        self.session_stats = {
            "total_communications": 0,
            "successful_uploads": 0,
            "connection_changes": 0,
        }

        # 清空实体
        self.vehicles = []
        self.base_stations = []
        # 重新初始化
        self._initialize_environment()
        print("环境重置完成")

    def update_session(self, session_id):
        "更新训练会话"
        self.current_session = session_id
        # 更新车辆位置，此处的time_delta需要计算，待定
        self.update_vehicle_positions(time_delta=1)
        # 更新数据模拟器
        self.data_simulator.update_session(session_id)
        # 为车辆生成新数据
        self._refresh_vehicle_data()
        print("*****************************")
        print(f"*    Session {session_id} 更新完成     *")
        print("*****************************")

    def _refresh_vehicle_data(self):
        """为所有车辆刷新数据"""
        for vehicle in self.vehicles:
            # 生成新的数据批次
            new_data = self.data_simulator.generate_vehicle_data(vehicle.id)

            # 更新车辆数据
            vehicle.data_batches = new_data

            # 更新数据统计
            if new_data:
                total_samples = 0
                for loader in new_data:
                    if hasattr(loader, "dataset"):
                        total_samples += len(loader.dataset)

                vehicle.data_statistics = {
                    "total_samples": total_samples,
                    "num_batches": len(new_data),
                    "current_domain": self.data_simulator.get_current_domain(),
                }


# 使用示例
from matplotlib.patches import Rectangle


def plot_vehicle_trajectories_separate(env, duration=60, time_step=1):
    """
    分别绘制三个图：轨迹图、位置-时间图和热力图

    参数:
    env: VehicleEnvironment 实例
    duration: 模拟时间长度(秒)
    time_step: 时间步长(秒)
    """

    # 存储轨迹数据
    trajectories = {
        vehicle.id: {"x": [], "y": [], "times": []} for vehicle in env.vehicles
    }

    # 模拟60秒并记录位置
    current_time = 0
    while current_time <= duration:
        # 更新车辆位置
        env.update_vehicle_positions(time_step)
        current_time += time_step

        # 记录每辆车的位置
        for vehicle in env.vehicles:
            trajectories[vehicle.id]["x"].append(vehicle.position[0])
            trajectories[vehicle.id]["y"].append(vehicle.position[1])
            trajectories[vehicle.id]["times"].append(current_time)

    # 图1: 车辆轨迹图（调整宽高比）
    plt.figure(figsize=(26, 6))  # 增加宽度，减小高度
    ax1 = plt.gca()

    colors = plt.cm.tab10(np.linspace(0, 1, len(env.vehicles)))

    for i, (vehicle_id, trajectory) in enumerate(trajectories.items()):
        ax1.plot(
            trajectory["x"],
            trajectory["y"],
            color=colors[i],
            linewidth=2,
            alpha=0.7,
            label=f"Vehicle {vehicle_id}",
        )

        # 标记起点和终点
        ax1.scatter(
            trajectory["x"][0],
            trajectory["y"][0],
            color=colors[i],
            marker="o",
            s=50,
            zorder=5,
        )
        ax1.scatter(
            trajectory["x"][-1],
            trajectory["y"][-1],
            color=colors[i],
            marker="s",
            s=50,
            zorder=5,
        )

    # 添加基站位置
    if hasattr(env, "base_stations") and env.base_stations:
        for i, bs in enumerate(env.base_stations):
            # 假设基站有position属性，如果没有请根据实际情况调整
            if hasattr(bs, "position"):
                bs_x, bs_y = bs.position
            else:
                # 如果基站是字典形式
                bs_x, bs_y = bs.get(
                    "position", (i * env.road_length / len(env.base_stations), 0)
                )

            ax1.scatter(
                bs_x,
                bs_y,
                color="red",
                marker="^",
                s=150,
                label=f"Base Station {i+1}" if i == 0 else "",
                zorder=10,
            )
            ax1.text(
                bs_x, bs_y + 1, f"BS{i+1}", ha="center", va="bottom", fontweight="bold"
            )

    # 设置子图1属性
    ax1.set_xlabel("X Position (m)")
    ax1.set_ylabel("Y Position (m)")
    ax1.set_title("Vehicle Trajectories Over Time with Base Stations")
    ax1.grid(True, alpha=0.3)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

    # 绘制道路边界
    ax1.add_patch(
        Rectangle(
            (0, -env.road_width / 2),
            env.road_length,
            env.road_width,
            fill=False,
            edgecolor="black",
            linestyle="--",
            alpha=0.5,
        )
    )

    # 绘制车道线
    lane_width = env.road_width / env.num_lanes
    for i in range(1, env.num_lanes):
        y_pos = -env.road_width / 2 + i * lane_width
        ax1.axhline(y=y_pos, color="gray", linestyle=":", alpha=0.5)

    plt.tight_layout()
    plt.show()

    # 图2: X坐标随时间变化
    plt.figure(figsize=(12, 6))
    ax2 = plt.gca()

    for i, (vehicle_id, trajectory) in enumerate(trajectories.items()):
        ax2.plot(
            trajectory["times"],
            trajectory["x"],
            color=colors[i],
            linewidth=2,
            alpha=0.7,
            label=f"Vehicle {vehicle_id}",
        )

    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("X Position (m)")
    ax2.set_title("Longitudinal Position Over Time")
    ax2.grid(True, alpha=0.3)
    ax2.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

    plt.tight_layout()
    plt.show()

    # 图3: 位置热力图
    plt.figure(figsize=(12, 6))
    ax3 = plt.gca()

    # 收集所有位置点
    all_x = []
    all_y = []
    for trajectory in trajectories.values():
        all_x.extend(trajectory["x"])
        all_y.extend(trajectory["y"])

    # 创建热力图
    heatmap, xedges, yedges = np.histogram2d(
        all_x,
        all_y,
        bins=[50, 20],
        range=[[0, env.road_length], [-env.road_width / 2, env.road_width / 2]],
    )

    # 显示热力图
    im = ax3.imshow(
        heatmap.T,
        extent=[0, env.road_length, -env.road_width / 2, env.road_width / 2],
        origin="lower",
        cmap="hot",
        aspect="auto",
    )

    # 添加基站位置到热力图
    if hasattr(env, "base_stations") and env.base_stations:
        for i, bs in enumerate(env.base_stations):
            if hasattr(bs, "position"):
                bs_x, bs_y = bs.position
            else:
                bs_x, bs_y = bs.get(
                    "position", (i * env.road_length / len(env.base_stations), 0)
                )

            ax3.scatter(
                bs_x,
                bs_y,
                color="cyan",
                marker="^",
                s=150,
                label=f"Base Station {i+1}" if i == 0 else "",
                zorder=10,
            )
            ax3.text(
                bs_x,
                bs_y + 1,
                f"BS{i+1}",
                ha="center",
                va="bottom",
                fontweight="bold",
                color="white",
            )

    # 绘制道路边界
    ax3.add_patch(
        Rectangle(
            (0, -env.road_width / 2),
            env.road_length,
            env.road_width,
            fill=False,
            edgecolor="blue",
            linewidth=2,
        )
    )

    ax3.set_xlabel("X Position (m)")
    ax3.set_ylabel("Y Position (m)")
    ax3.set_title("Vehicle Position Heatmap with Base Stations")
    plt.colorbar(im, ax=ax3, label="Position Frequency")
    ax3.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

    plt.tight_layout()
    plt.show()

    return trajectories


def create_enhanced_trajectory_plot(env, trajectories):
    """
    创建一个增强版的轨迹图，包含更多细节
    """
    plt.figure(figsize=(18, 6))  # 更宽的图形

    colors = plt.cm.tab10(np.linspace(0, 1, len(env.vehicles)))

    # 绘制轨迹
    for i, (vehicle_id, trajectory) in enumerate(trajectories.items()):
        plt.plot(
            trajectory["x"],
            trajectory["y"],
            color=colors[i],
            linewidth=2,
            alpha=0.7,
            label=f"Vehicle {vehicle_id}",
        )

        # 标记起点和终点
        plt.scatter(
            trajectory["x"][0],
            trajectory["y"][0],
            color=colors[i],
            marker="o",
            s=80,
            zorder=5,
        )
        plt.scatter(
            trajectory["x"][-1],
            trajectory["y"][-1],
            color=colors[i],
            marker="s",
            s=80,
            zorder=5,
        )

    # 添加基站位置
    if hasattr(env, "base_stations") and env.base_stations:
        for i, bs in enumerate(env.base_stations):
            if hasattr(bs, "position"):
                bs_x, bs_y = bs.position
            else:
                bs_x, bs_y = bs.get(
                    "position", (i * env.road_length / len(env.base_stations), 0)
                )

            plt.scatter(
                bs_x,
                bs_y,
                color="red",
                marker="^",
                s=200,
                label=f"Base Station {i+1}",
                zorder=10,
            )
            plt.text(
                bs_x,
                bs_y + 1.5,
                f"BS{i+1}",
                ha="center",
                va="bottom",
                fontweight="bold",
                fontsize=12,
            )

    # 绘制道路边界
    plt.gca().add_patch(
        Rectangle(
            (0, -env.road_width / 2),
            env.road_length,
            env.road_width,
            fill=False,
            edgecolor="black",
            linewidth=2,
            alpha=0.7,
        )
    )

    # 绘制车道线
    lane_width = env.road_width / env.num_lanes
    for i in range(1, env.num_lanes):
        y_pos = -env.road_width / 2 + i * lane_width
        plt.axhline(y=y_pos, color="gray", linestyle="--", alpha=0.5)

    # 添加图例和标签
    plt.xlabel("X Position (m)", fontsize=12)
    plt.ylabel("Y Position (m)", fontsize=12)
    plt.title("Enhanced Vehicle Trajectories with Base Stations", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=10)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # 初始化环境
    env = VehicleEnvironment()
    env.update_session(0)
    env.update_session(1)
    # 绘制三个分开的图
    # trajectories = plot_vehicle_trajectories_separate(env, duration=60, time_step=0.5)
    # # 可选：创建增强版轨迹图
    # create_enhanced_trajectory_plot(env, trajectories)
