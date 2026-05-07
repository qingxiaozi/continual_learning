import logging
import numpy as np
import torch
from config.parameters import Config
from config.paths import Paths

logger = logging.getLogger(__name__)
from collections import defaultdict
from pyproj import Transformer, CRS
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.ticker import FuncFormatter
from scipy.spatial import cKDTree
import pandas as pd
import random
import os
import ast

fm._load_fontmanager(try_read_cache=False)


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

    def __init__(self, global_model, gold_model, cache_manager, data_simulator, mode="train"):
        # 实体集合
        self.vehicles = []  # 车辆对象列表
        self.base_stations = []  # 基站对象列表，字典
        self.current_session = 0  # 当前训练会话编号
        self.environment_time = 0.0  # 环境运行时间(s)

        # 根据模式选择轨迹文件
        trajectory_file = "train_trajectory.csv" if mode == "train" else "test_trajectory.csv"
        self.trajectory_data = pd.read_csv(os.path.join(Paths.TRAJECTORY_DIR, trajectory_file))
        self.trajectory_points = None  # 初始为空
        self.trajectory_index = 0  # 轨迹点索引
        self.base_point = [-9.374787, 37.088271]

        # PPP参数
        self.ppp_radius = Config.PPP_RADIUS
        self.ppp_lambda = Config.PPP_LAMBDA
        self.ppp_lambda_bs = Config.PPP_LAMBDA_BS

        # 初始化数据环境
        self.data_simulator = data_simulator
        self.global_model = global_model
        self.gold_model = gold_model
        self.cache_manager = cache_manager

        # 初始化物理环境
        self._initialize_base_stations()
        self._initialize_environment()
        # self.plot_all_trajectories_and_bs(Paths.ALL_TRAJECTORIES_PNG)

    def plot_all_trajectories_and_bs(self, save_path='all_trajectories.png', zoom_region=None):
        """
        绘制路网图，包含所有轨迹和基站，并可选地添加局部放大区域

        Args:
            save_path: 图片保存路径
            zoom_region: 元组 (lon_min, lon_max, lat_min, lat_max)，指定放大区域经纬度范围
        """
        import ast
        import matplotlib.pyplot as plt
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.serif'] = ['Nimbus Roman']
        fig, ax = plt.subplots(figsize=(14, 10))

        all_lons, all_lats = [], []
        for poly in self.trajectory_data['POLYLINE']:
            try:
                pts = ast.literal_eval(poly)
                if pts:
                    lons, lats = zip(*pts)
                    all_lons.extend(lons)
                    all_lats.extend(lats)
                    ax.plot(lons, lats, 'b-', linewidth=0.3, alpha=0.6)
            except:
                continue

        if self.base_stations:
            bs_lons = [bs['lonlat_position'][0] for bs in self.base_stations]
            bs_lats = [bs['lonlat_position'][1] for bs in self.base_stations]
            ax.scatter(bs_lons, bs_lats, c='red', s=15, marker='^', alpha=0.7,
                      label=f'Base Stations ({len(self.base_stations)})', zorder=5)

        ax.plot([], [], 'b-', linewidth=0.3, label=f'Trajectories ({len(self.trajectory_data)})')
        ax.set_xlabel("Longitude(°)", fontsize=38, fontname='Nimbus Roman')
        ax.set_ylabel("Latitude(°)", fontsize=38, fontname='Nimbus Roman')
        ax.set_title(f"Road Network: {len(self.trajectory_data)} Trajectories with {len(self.base_stations)} Base Stations", fontsize=38, fontname='Nimbus Roman')
        ax.legend(loc='upper right', prop={'family': 'Nimbus Roman', 'size': 38})

        if zoom_region is None and all_lons and all_lats:
            lon_margin = (max(all_lons) - min(all_lons)) * 0.05
            lat_margin = (max(all_lats) - min(all_lats)) * 0.05
            zoom_region = (
                min(all_lons) + lon_margin,
                min(all_lons) + (max(all_lons) - min(all_lons)) * 0.15,
                min(all_lats) + lat_margin,
                min(all_lats) + (max(all_lats) - min(all_lats)) * 0.15
            )

        if zoom_region:
            lon_min, lon_max, lat_min, lat_max = zoom_region

            ax_inset = fig.add_axes([0.55, 0.55, 0.35, 0.35])

            for poly in self.trajectory_data['POLYLINE']:
                try:
                    pts = ast.literal_eval(poly)
                    if pts:
                        ax_inset.plot(*zip(*pts), 'b-', linewidth=0.5, alpha=0.8)
                except:
                    continue

            bs_in_region = [
                (bs['lonlat_position'][0], bs['lonlat_position'][1])
                for bs in self.base_stations
                if lon_min <= bs['lonlat_position'][0] <= lon_max
                and lat_min <= bs['lonlat_position'][1] <= lat_max
            ]

            if bs_in_region:
                inset_bs_lons, inset_bs_lats = zip(*bs_in_region)
                ax_inset.scatter(inset_bs_lons, inset_bs_lats, c='red', s=50, marker='^',
                                alpha=0.9, label=f'{len(bs_in_region)} BS', zorder=5)

            ax_inset.set_xlim(lon_min, lon_max)
            ax_inset.set_ylim(lat_min, lat_max)
            ax_inset.set_title("Zoomed Region", fontsize=38, fontname='Nimbus Roman')
            ax_inset.tick_params(axis='both', labelsize=38)

            rect = plt.Rectangle((lon_min, lat_min), lon_max - lon_min, lat_max - lat_min,
                                 fill=False, edgecolor='green', linewidth=2, linestyle='--')
            ax.add_patch(rect)

            if bs_in_region:
                ax_inset.legend(loc='upper right', prop={'family': 'Nimbus Roman', 'size': 38})

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"Saved: {save_path}")

    def _initialize_environment(self):
        """初始化车辆"""
        self.vehicles = []
        self._load_random_trajectory()
        self._initialize_vehicles_with_ppp()

    def _load_random_trajectory(self):
        """随机加载一条轨迹"""
        if self.trajectory_data.empty: return
        try:
            pts = ast.literal_eval(self.trajectory_data.sample(1).iloc[0]['POLYLINE'])
            if len(pts) >= 2:
                lons, lats = zip(*pts)
                self.trajectory_points = self._convert_to_cartesian(lons, lats)
        except:
            self.trajectory_points = None

    def _convert_to_cartesian(self, lon_array, lat_array):
        """
        使用 UTM 投影将经纬度转换为平面坐标（单位：米），
        以基准点[-9.374787, 37.088271]为中心
        """
        lon_array = np.asarray(lon_array, dtype=np.float64)
        lat_array = np.asarray(lat_array, dtype=np.float64)

        # 固定基准点
        BASE_LON, BASE_LAT = -9.374787, 37.088271

        wgs84 = CRS("EPSG:4326")
        utm_crs = CRS(
            proj="utm",
            zone=int((BASE_LON + 180) // 6) + 1,
            south=BASE_LAT < 0,
            ellps="WGS84"
        )

        x, y = Transformer.from_crs(wgs84, utm_crs, always_xy=True).transform(lon_array, lat_array)

        return np.column_stack([x, y])

    def _initialize_base_stations(self):
        """
        城市级宏基站部署（沿轨迹缓冲区的 PPP）
        - 使用所有轨迹（train + test）生成基站，只生成一次并缓存
        - KDTree 去重，复杂度 O(N log N)
        """
        # 尝试加载已缓存的基站位置
        bs_cache_path = Paths.get_base_stations_cache_path()
        if os.path.exists(bs_cache_path):
            self.base_stations = np.load(bs_cache_path, allow_pickle=True).tolist()
            print(f"从缓存加载 {len(self.base_stations)} 个基站")
            return

        # 获取所有轨迹点（train + test）
        all_trajectory_data = []
        for traj_file in ["train_trajectory.csv", "test_trajectory.csv"]:
            traj_path = os.path.join(Paths.TRAJECTORY_DIR, traj_file)
            if os.path.exists(traj_path):
                df = pd.read_csv(traj_path)
                all_trajectory_data.append(df)

        if not all_trajectory_data:
            self.base_stations = []
            return

        # 合并所有轨迹
        combined_trajectory = pd.concat(all_trajectory_data, ignore_index=True)

        points = []
        for poly in combined_trajectory['POLYLINE']:
            try:
                pts = ast.literal_eval(poly) if isinstance(poly, str) else poly
                points.extend(pts)
            except Exception:
                continue

        if not points:
            self.base_stations = []
            return

        # ========= 1. 所有轨迹点 → UTM =========
        lons, lats = zip(*points)
        traj_utm = self._convert_to_cartesian(lons, lats)
        LAMBDA_MACRO = Config.PPP_LAMBDA_BS / 1e6       # 3 BS / km² → BS / m²
        MIN_BS_DISTANCE = Config.MIN_BS_DISTANCE

        # ========= 3. 沿轨迹缓冲区生成 PPP 候选基站 =========
        bs_candidates = []

        local_area = np.pi * Config.BASE_STATION_COVERAGE ** 2

        for p in traj_utm:
            n_bs = np.random.poisson(local_area * LAMBDA_MACRO)
            if n_bs == 0:
                continue

            angles = np.random.uniform(0, 2 * np.pi, n_bs)
            radii = np.random.uniform(0, Config.BASE_STATION_COVERAGE, n_bs)

            offsets = np.column_stack([
                radii * np.cos(angles),
                radii * np.sin(angles)
            ])

            bs_candidates.append(p + offsets)

        if not bs_candidates:
            self.base_stations = []
            return

        bs_candidates = np.vstack(bs_candidates)

        # 随机打乱，避免空间顺序偏置（很重要）
        np.random.shuffle(bs_candidates)

        # ========= 4. KDTree 去除过密基站（O(N log N)） =========
        filtered_bs = []
        kdtree = None

        for p in bs_candidates:
            if not filtered_bs:
                filtered_bs.append(p)
                kdtree = cKDTree(np.array(filtered_bs))
                continue

            # 查询最小间距内是否已有基站
            idxs = kdtree.query_ball_point(p, r=MIN_BS_DISTANCE)

            if len(idxs) == 0:
                filtered_bs.append(p)
                kdtree = cKDTree(np.array(filtered_bs))  # 动态更新

        filtered_bs = np.array(filtered_bs)

        # ========= 5. UTM → 经纬度 =========
        utm_zone = int((self.base_point[0] + 180) // 6) + 1
        utm_crs = CRS(
            proj="utm",
            zone=utm_zone,
            south=self.base_point[1] < 0,
            ellps="WGS84"
        )

        transformer = Transformer.from_crs(
            utm_crs,
            CRS("EPSG:4326"),
            always_xy=True
        )

        bs_lons, bs_lats = transformer.transform(
            filtered_bs[:, 0],
            filtered_bs[:, 1]
        )

        # ========= 6. 构建基站对象 =========
        self.base_stations = []
        for i, (lon, lat, utm) in enumerate(zip(bs_lons, bs_lats, filtered_bs)):
            self.base_stations.append({
                "id": i,
                "lonlat_position": np.array([lon, lat]),
                "utm_position": utm,
                "coverage": Config.BASE_STATION_COVERAGE,
                "capacity": 80,
                "connected_vehicles": []
            })

        print(f"初始化 {len(self.base_stations)} 个基站")

        # 保存基站位置到缓存文件
        bs_cache_path = Paths.get_base_stations_cache_path()
        np.save(bs_cache_path, self.base_stations)
        print(f"基站位置已缓存到: {bs_cache_path}")


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
        self._update_vehicle_connections()

    def _generate_ppp_vehicles(self, center_position):
        """在主车轨迹附近生成固定数量的从车"""
        num_vehicles = Config.NUM_VEHICLES - 1
        if num_vehicles <= 0:
            return

        if self.trajectory_points is not None and self.trajectory_index < len(self.trajectory_points) - 1:
            direction = self.trajectory_points[self.trajectory_index + 1] - self.trajectory_points[self.trajectory_index]
        elif self.trajectory_points is not None and self.trajectory_index > 0:
            direction = self.trajectory_points[self.trajectory_index] - self.trajectory_points[self.trajectory_index - 1]
        elif self.trajectory_points is not None and len(self.trajectory_points) > 1:
            direction = self.trajectory_points[-1] - self.trajectory_points[0]
        else:
            angle = np.random.uniform(0, 2 * np.pi)
            direction = np.array([np.cos(angle), np.sin(angle)])

        direction = direction / (np.linalg.norm(direction) + 1e-8)
        perpendicular = np.array([-direction[1], direction[0]])

        half_length = 100.0
        half_width = 10.0

        offsets_along = np.random.uniform(-half_length, half_length, num_vehicles)
        offsets_perp = np.random.uniform(-half_width, half_width, num_vehicles)

        offsets = np.column_stack([
            direction[0] * offsets_along + perpendicular[0] * offsets_perp,
            direction[1] * offsets_along + perpendicular[1] * offsets_perp
        ])

        positions = center_position + offsets

        for pos in positions:
            vehicle = Vehicle(
                vehicle_id=len(self.vehicles),
                position=pos
            )
            self.vehicles.append(vehicle)

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
            if np.linalg.norm(position - bs["utm_position"]) <= bs["coverage"]
            and len(bs["connected_vehicles"]) < bs["capacity"]
        ]
        if not available_bs:
            print(f"No available base station for position {position}")
            return None

        return min(available_bs, key=lambda bs: np.linalg.norm(position - bs["utm_position"]))

    def update_vehicle_positions(self, time_delta=20):
        """更新车辆位置"""
        if self.trajectory_index >= len(self.trajectory_points) - 1:
            print(f"车辆已行驶至终点")
            return

        distance = Config.VEHICLE_SPEED_FACTOR * time_delta
        idx = self.trajectory_index
        accumulated = 0.0

        while idx < len(self.trajectory_points) - 1 and accumulated + np.linalg.norm(self.trajectory_points[idx+1] - self.trajectory_points[idx]) <= distance:
            accumulated += np.linalg.norm(self.trajectory_points[idx+1] - self.trajectory_points[idx])
            idx += 1

        if idx > self.trajectory_index:
            self.trajectory_index = idx
            self.vehicles[0].position = self.trajectory_points[idx]
            self.vehicles = [self.vehicles[0]]
            self._generate_ppp_vehicles(self.vehicles[0].position)
        self._update_vehicle_connections()
        self.environment_time += time_delta

    def _update_vehicle_connections(self):
        """更新所有车辆的基站连接"""
        for bs in self.base_stations:
            bs["connected_vehicles"] = []

        for vehicle in self.vehicles:
            # 建立新连接
            nearest_bs = self._find_nearest_base_station(vehicle.position)
            if nearest_bs and len(nearest_bs["connected_vehicles"]) < nearest_bs["capacity"]:
                vehicle.set_bs_connection(nearest_bs["id"])
                nearest_bs["connected_vehicles"].append(vehicle.id)
                logger.info(f"Vehicle {vehicle.id} connected to Base Station {nearest_bs['id']}")
            else:
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
            if vehicle is None:
                state.extend([0.5, 1.0, 0.0, 1.0])
                continue
                
            try:
                confidence = 0.5
                if hasattr(vehicle, 'data_batches') and vehicle.data_batches:
                    confidence = vehicle.get_inference_confidence(self.global_model)
                if hasattr(vehicle, 'confidence_history'):
                    vehicle.confidence_history.append(confidence)

                test_loss = 1.0
                if hasattr(vehicle, 'uploaded_data') and vehicle.uploaded_data:
                    raw_loss = vehicle.calculate_test_loss(self.global_model, self.gold_model)
                    test_loss = 1.0 / (1.0 + raw_loss)
                if hasattr(vehicle, 'test_loss_history'):
                    vehicle.test_loss_history.append(test_loss)

                quality_score = 0.0
                if hasattr(vehicle, 'update_quality_scores'):
                    quality_score = vehicle.update_quality_scores(self.cache_manager)

                normalized_distance = 1.0
                if vehicle.bs_connection is not None:
                    bs = self._get_base_station_by_id(vehicle.bs_connection)
                    if bs:
                        distance = np.linalg.norm(vehicle.position - bs["utm_position"])
                        normalized_distance = min(distance / Config.BASE_STATION_COVERAGE, 1.0)

                state.extend([confidence, test_loss, quality_score, normalized_distance])

            except Exception as e:
                logger.debug(f"Error getting state for vehicle {getattr(vehicle, 'id', 'unknown')}: {e}")
                state.extend([0.5, 1.0, 0.0, 1.0])

        return np.array(state, dtype=np.float32)

    def reset(self):
        """重置环境"""
        logger.debug("重置车辆环境...")
        self.environment_time = 0.0
        self.trajectory_index = 0
        self._initialize_environment()
        logger.debug("环境重置完成")

if __name__ == "__main__":
    class MinimalEnv:
        def __init__(self):
            import pandas as pd
            import os
            from config.paths import Paths
            from config.parameters import Config
            from scipy.spatial import cKDTree
            from pyproj import Transformer, CRS
            import ast

            self.base_point = [-9.374787, 37.088271]

            train_data = pd.read_csv(os.path.join(Paths.TRAJECTORY_DIR, "train_trajectory.csv"))
            test_data = pd.read_csv(os.path.join(Paths.TRAJECTORY_DIR, "test_trajectory.csv"))
            self.trajectory_data = pd.concat([train_data, test_data], ignore_index=True)

            points = []
            for poly in self.trajectory_data['POLYLINE']:
                try:
                    pts = ast.literal_eval(poly) if isinstance(poly, str) else poly
                    points.extend(pts)
                except:
                    continue

            if points:
                lons, lats = zip(*points)
                BASE_LON, BASE_LAT = -9.374787, 37.088271
                wgs84 = CRS("EPSG:4326")
                utm_crs = CRS(proj="utm", zone=int((BASE_LON + 180) // 6) + 1, south=BASE_LAT < 0, ellps="WGS84")
                self.converter = Transformer.from_crs(wgs84, utm_crs, always_xy=True)
                traj_utm_x, traj_utm_y = self.converter.transform(lons, lats)
                traj_utm = np.column_stack([traj_utm_x, traj_utm_y])

                local_area = np.pi * Config.BASE_STATION_COVERAGE ** 2
                LAMBDA_MACRO = Config.PPP_LAMBDA_BS / 1e6

                bs_candidates = []
                for p in traj_utm:
                    n_bs = np.random.poisson(local_area * LAMBDA_MACRO)
                    if n_bs == 0:
                        continue
                    angles = np.random.uniform(0, 2 * np.pi, n_bs)
                    radii = np.random.uniform(0, Config.BASE_STATION_COVERAGE, n_bs)
                    offsets = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
                    bs_candidates.append(p + offsets)

                bs_candidates = np.vstack(bs_candidates) if bs_candidates else np.array([]).reshape(0, 2)
                np.random.shuffle(bs_candidates)

                filtered_bs = []
                kdtree = None
                for p in bs_candidates:
                    if not filtered_bs:
                        filtered_bs.append(p)
                        kdtree = cKDTree(np.array(filtered_bs))
                        continue
                    idxs = kdtree.query_ball_point(p, r=Config.MIN_BS_DISTANCE)
                    if len(idxs) == 0:
                        filtered_bs.append(p)
                        kdtree = cKDTree(np.array(filtered_bs))

                filtered_bs = np.array(filtered_bs)

                utm_zone = int((self.base_point[0] + 180) // 6) + 1
                utm_crs = CRS(proj="utm", zone=utm_zone, south=self.base_point[1] < 0, ellps="WGS84")
                transformer = Transformer.from_crs(utm_crs, CRS("EPSG:4326"), always_xy=True)
                bs_lons, bs_lats = transformer.transform(filtered_bs[:, 0], filtered_bs[:, 1])

                self.base_stations = []
                for i, (lon, lat, utm) in enumerate(zip(bs_lons, bs_lats, filtered_bs)):
                    self.base_stations.append({
                        "id": i,
                        "lonlat_position": np.array([lon, lat]),
                        "utm_position": utm,
                        "coverage": Config.BASE_STATION_COVERAGE,
                        "capacity": 80,
                        "connected_vehicles": []
                    })

                print(f"初始化 {len(self.base_stations)} 个基站")

        def plot_all_trajectories_and_bs(self, save_path='all_trajectories.png', zoom_region=None):
            import ast
            import matplotlib.pyplot as plt
            plt.rcParams['font.family'] = 'serif'
            plt.rcParams['font.serif'] = ['Nimbus Roman']
            fig, ax = plt.subplots(figsize=(16, 10))

            all_lons, all_lats = [], []
            for poly in self.trajectory_data['POLYLINE']:
                try:
                    pts = ast.literal_eval(poly) if isinstance(poly, str) else poly
                    if pts:
                        lons, lats = zip(*pts)
                        all_lons.extend(lons)
                        all_lats.extend(lats)
                        ax.plot(lons, lats, 'b-', linewidth=1, alpha=1)
                except:
                    continue

            ax.plot([], [], 'b-', linewidth=1, label='Trajectories')

            if all_lons and all_lats:
                lon_range = max(all_lons) - min(all_lons)
                ax.set_xlim(min(all_lons) - lon_range * 0.15, max(all_lons) + lon_range * 0.05)

            ax.set_xlabel("Longitude(°)", fontsize=38, fontname='Nimbus Roman')
            ax.set_ylabel("Latitude(°)", fontsize=38, fontname='Nimbus Roman')
            ax.legend(loc='lower left', prop={'family': 'Nimbus Roman', 'size': 38}, frameon=False)
            ax.tick_params(axis='both', labelsize=38)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            if zoom_region is None and all_lons and all_lats:
                zoom_region = (-8.62, -8.58, 39.98, 40.02)

            if zoom_region:
                lon_min, lon_max, lat_min, lat_max = zoom_region

                ax_inset = fig.add_axes([0.70, 0.35, 0.25, 0.25])

                for poly in self.trajectory_data['POLYLINE']:
                    try:
                        pts = ast.literal_eval(poly) if isinstance(poly, str) else poly
                        if pts:
                            ax_inset.plot(*zip(*pts), 'b-', linewidth=0.8, alpha=0.8)
                    except:
                        continue

                bs_in_region = [
                    (bs['lonlat_position'][0], bs['lonlat_position'][1])
                    for bs in self.base_stations
                    if lon_min <= bs['lonlat_position'][0] <= lon_max
                    and lat_min <= bs['lonlat_position'][1] <= lat_max
                ]

                if bs_in_region:
                    inset_bs_lons, inset_bs_lats = zip(*bs_in_region)
                    ax_inset.scatter(inset_bs_lons, inset_bs_lats, c='red', s=50, marker='^',
                                    alpha=0.9, zorder=5)
                    from matplotlib.lines import Line2D
                    legend_handle = Line2D([0], [0], marker='^', color='w', markerfacecolor='red',
                                          markersize=15, linestyle='None')
                    ax_inset.legend([legend_handle], ['BS'], loc='lower right', prop={'family': 'Nimbus Roman', 'size': 38}, frameon=False, handletextpad=0.2)

                ax_inset.set_xlim(lon_min, lon_max)
                ax_inset.set_ylim(lat_min, lat_max)
                # ax_inset.set_title("Zoomed Region", fontsize=32)
                ax_inset.tick_params(axis='both', labelsize=38, pad=5)
                ax_inset.spines['top'].set_visible(False)
                ax_inset.spines['right'].set_visible(False)

                rect = plt.Rectangle((lon_min, lat_min), lon_max - lon_min, lat_max - lat_min,
                                     fill=False, edgecolor='green', linewidth=2, linestyle='--')
                ax.add_patch(rect)

            plt.tight_layout()

            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')

            plt.close()
            print(f"Saved: {save_path}")

    env = MinimalEnv()
    env.plot_all_trajectories_and_bs(save_path='road_network.png')
