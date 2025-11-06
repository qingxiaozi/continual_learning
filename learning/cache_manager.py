import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.parameters import config


class cacheManager:
    """缓存管理器"""
    def __init__(self, max_size=config.MAX_LOCAL_BATCHES):
        self.max_size = max_size
        self.caches = {} # vehicle_id->cache_data

    def initialize_vehicle_cache(self, vehicle_id):
        """初始化车辆缓存"""
        self.caches[vehicle_id] = {
            'old_data': [],
            'new_data': [],
            'quality_scores': []
        }

    def update_cache(self, vehicle_id, new_data_batches, quality_scores=None):
        """更新车辆缓存"""
        if vehicle_id not in self.caches:
            self.initialize_vehicle_cache(vehicle_id)

        cache = self.caches[vehicle_id]

        # 添加新数据
        for batch in new_data_batches:
            cache['new_data'].append(batch)

        # 更新质量评分
        if quality_scores is not None:
            cache['quality_scores'] = quality_scores

        # 维护缓存大小
        self._maintain_cache_size(vehicle_id)

    def _maintain_cache_size(self, vehicle_id):
        """维护缓存大小"""
        cache = self.caches[vehicle_id]
        total_size = len(cache['old_data']) + len(cache['new_data'])

        if total_size > self.max_size:
            # 移除质量最低的旧数据
            excess = total_size - self.max_size
            if cache['quality_scores'] and len(cache['old_data']) > 0:
                # 基于质量评分移除
                quality_indices = np.argsort(cache['quality_scores'])[:excess]
                for idx in sorted(quality_indices, reverse=True):
                    if idx < len(cache['old_data']):
                        cache['old_data'].pop(idx)
            else:
                # 随机移除
                remove_count = min(excess, len(cache['old_data']))
                cache['old_data'] = cache['old_data'][remove_count:]

    def get_vehicle_cache(self, vehicle_id):
        """获取车辆缓存"""
        return self.caches.get(vehicle_id, {'old_data': [], 'new_data': []})

    def promote_new_to_old(self, vehicle_id):
        """将新数据提升为旧数据"""
        if vehicle_id not in self.caches:
            return

        cache = self.caches[vehicle_id]
        cache['old_data'].extend(cache['new_data'])
        cache['new_data'] = []

    def get_cache_stats(self):
        """获取缓存统计信息"""
        stats = {}
        for vehicle_id, cache in self.caches.items():
            stats[vehicle_id] = {
                'old_data_size': len(cache['old_data']),
                'new_data_size': len(cache['new_data']),
                'total_size': len(cache['old_data']) + len(cache['new_data'])
            }
        return stats