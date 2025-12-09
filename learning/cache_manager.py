import numpy as np
from config.parameters import Config


class CacheManager:
    """缓存管理器"""

    def __init__(self, max_size=Config.MAX_LOCAL_BATCHES):
        self.max_size = max_size
        self.caches = {}  # vehicle_id->cache_data

    def initialize_vehicle_cache(self, vehicle_id):
        """初始化车辆缓存"""
        self.caches[vehicle_id] = {
            "old_data": [],
            "new_data": [],
            "quality_scores": [],
            "batch_mapping": [],  # 记录批次在全局数据集中的索引
        }

    def update_cache(self, vehicle_id, new_data_batches, quality_scores=None):
        """更新车辆缓存"""
        if vehicle_id not in self.caches:
            self.initialize_vehicle_cache(vehicle_id)

        cache = self.caches[vehicle_id]

        # 清空新数据，避免重复添加
        cache["new_data"] = []
        # 添加新数据
        for batch in new_data_batches:
            cache["new_data"].append(batch)

        # 更新质量评分
        if quality_scores is not None:
            cache["quality_scores"] = quality_scores

        # 维护缓存大小
        self._maintain_cache_size(vehicle_id)

    def update_quality_scores(self, vehicle_id, quality_scores):
        """更新车辆缓存的质量评分"""
        if vehicle_id in self.caches:
            cache = self.caches[vehicle_id]
            # 确保质量评分数量与旧数据批次数量匹配
            if len(quality_scores) == len(cache["old_data"]):
                cache["quality_scores"] = quality_scores
            else:
                print(f"警告: 车辆 {vehicle_id} 质量评分数量不匹配")
                # 如果数量不匹配，只更新前min(len, len)个
                min_len = min(len(quality_scores), len(cache["old_data"]))
                cache["quality_scores"] = quality_scores[:min_len]

    def set_batch_mapping(self, vehicle_id, batch_mapping):
        """设置批次映射关系"""
        if vehicle_id in self.caches:
            self.caches[vehicle_id]["batch_mapping"] = batch_mapping

    def get_batch_mapping(self, vehicle_id):
        """获取批次映射关系"""
        if vehicle_id in self.caches:
            return self.caches[vehicle_id].get("batch_mapping", [])
        return []

    # def _maintain_cache_size(self, vehicle_id):
    #     """维护缓存大小"""
    #     cache = self.caches[vehicle_id]
    #     total_size = len(cache["old_data"]) + len(cache["new_data"])

    #     if total_size > self.max_size:
    #         # 移除质量最低的旧数据
    #         excess = total_size - self.max_size
    #         if cache["quality_scores"] and len(cache["old_data"]) > 0:
    #             # 基于质量评分移除
    #             quality_indices = np.argsort(cache["quality_scores"])[:excess]
    #             for idx in sorted(quality_indices, reverse=True):
    #                 if idx < len(cache["old_data"]):
    #                     cache["old_data"].pop(idx)
    #                     cache["quality_scores"].pop(idx)
    #         else:
    #             # 随机移除
    #             remove_count = min(excess, len(cache["old_data"]))
    #             cache["old_data"] = cache["old_data"][remove_count:]
    #             # 同时移除对应的质量评分
    #             if cache["quality_scores"]:
    #                 cache["quality_scores"] = cache["quality_scores"][remove_count:]

    #         print(
    #             f"车辆 {vehicle_id} 缓存维护: 移除了 {excess} 个批次, "
    #             f"剩余 {len(cache['old_data'])} 个旧批次, {len(cache['quality_scores'])} 个质量评分"
    #         )

    def _maintain_cache_size(self, vehicle_id):
        """维护缓存大小 - 从新旧数据中统一移除"""
        cache = self.caches[vehicle_id]
        total_size = len(cache["old_data"]) + len(cache["new_data"])

        if total_size > self.max_size:
            # 移除质量最低的数据（不区分新旧）
            excess = total_size - self.max_size

            if cache["quality_scores"] and len(cache["quality_scores"]) == total_size:
                # 基于质量评分移除（考虑所有数据）
                quality_indices = np.argsort(cache["quality_scores"])[:excess]

                # 按索引从大到小移除，避免索引错位
                for idx in sorted(quality_indices, reverse=True):
                    # 判断是旧数据还是新数据
                    if idx < len(cache["old_data"]):
                        cache["old_data"].pop(idx)
                    else:
                        # 调整索引到新数据的位置
                        new_idx = idx - len(cache["old_data"])
                        cache["new_data"].pop(new_idx)

                    # 移除对应的质量评分
                    cache["quality_scores"].pop(idx)

            else:
                # 随机移除：先移旧数据，不够再移新数据
                remove_count = min(excess, len(cache["old_data"]))
                cache["old_data"] = cache["old_data"][remove_count:]

                # 如果旧数据不够移，再移新数据
                if remove_count < excess:
                    remaining = excess - remove_count
                    cache["new_data"] = cache["new_data"][remaining:]

                # 同时移除对应的质量评分
                if cache["quality_scores"]:
                    cache["quality_scores"] = cache["quality_scores"][excess:]

            print(
                f"车辆 {vehicle_id} 缓存维护: 移除了 {excess} 个批次, "
                f"剩余 {len(cache['old_data'])} 个旧批次, {len(cache['new_data'])} 个新批次"
            )

    def get_vehicle_cache(self, vehicle_id):
        """获取车辆缓存"""
        return self.caches.get(
            vehicle_id,
            {"old_data": [], "new_data": [], "quality_scores": [], "batch_mapping": []},
        )

    def promote_new_to_old(self, vehicle_id):
        """将新数据提升为旧数据"""
        if vehicle_id not in self.caches:
            return

        cache = self.caches[vehicle_id]
        cache["old_data"].extend(cache["new_data"])
        cache["new_data"] = []
        self._maintain_cache_size(vehicle_id)

    def get_cache_stats(self):
        """获取缓存统计信息"""
        stats = {}
        for vehicle_id, cache in self.caches.items():
            stats[vehicle_id] = {
                "old_data_size": len(cache["old_data"]),
                "new_data_size": len(cache["new_data"]),
                "total_size": len(cache["old_data"]) + len(cache["new_data"]),
                "quality_scores": cache["quality_scores"],
            }
        return stats
