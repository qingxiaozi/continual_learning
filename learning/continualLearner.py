import torch
import torch.optim as optim
from config.parameters import Config
from learning.trainer import EpochTrainer


class ContinualLearner:
    """
    持续学习 orchestrator
    """
    def __init__(self, model, gold_model):
        self.model = model
        self.gold_model = gold_model
        self.trainer = EpochTrainer(model, gold_model)

    def train_with_cache(
        self,
        cache_manager,
        data_simulator,
        current_domain,
        num_epochs,
    ):
        batches, batch_mapping = self._collect_batches(cache_manager)

        if len(batches) == 0:
            return {
                "loss_before": 1.0,
                "loss_after": 1.0,
                "actual_epochs": 0,
            }

        val_dataset = data_simulator.get_val_dataset(current_domain)
        loss_before = self._compute_loss_on_batches(batches)

        result = self.trainer.train(
            batches=batches,
            num_epochs=num_epochs,
            init_epochs=Config.INIT_EPOCHS,
            val_dataset=val_dataset,
        )

        loss_after = self._compute_loss_on_batches(batches)
        self._update_cache_quality(
            cache_manager,
            batch_mapping,
            result["selector"],
        )
        return {
            "loss_before": loss_before,
            "loss_after": loss_after,
            "actual_epochs": result["actual_epochs"],
        }
    
    def _collect_batches(self, cache_manager):
        batches = []
        mapping = {}
        idx = 0
        for vid in range(Config.NUM_VEHICLES):
            cache = cache_manager.get_vehicle_cache(vid)
            for t in ["old", "new"]:
                for local_idx, batch in enumerate(cache[f"{t}_data"]):
                    batches.append(batch)
                    mapping[idx] = {
                        "vehicle_id": vid,
                        "data_type": t,
                        "local_idx": local_idx,
                    }
                    idx += 1
        return batches, mapping
    
    def _compute_loss_on_batches(self, batch_list):
        """
        计算模型在一组 batch 上的加权总损失
        等价于 main.py 中 _compute_weighted_loss_on_uploaded_data
        """
        total_loss = 0.0
        criterion = torch.nn.CrossEntropyLoss()

        self.model.eval()
        with torch.no_grad():
            for batch in batch_list:
                if isinstance(batch, (list, tuple)):
                    inputs, targets = batch
                else:
                    inputs = batch
                    targets = self.gold_model.model(inputs).argmax(dim=1)
                device = next(self.model.parameters()).device
                inputs = inputs.to(device)
                targets = targets.to(device)
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                #所有batch上的“样本级总交叉熵损失”
                total_loss += loss.item() * inputs.shape[0]

        self.model.train()
        return total_loss if total_loss > 0 else 1.0

    def _update_cache_quality(self, cache_manager, batch_mapping, mab_selector):
        quality_scores = mab_selector.get_quality_scores()

        vehicle_scores = {}
        for global_idx, score in enumerate(quality_scores):
            if global_idx not in batch_mapping:
                continue

            info = batch_mapping[global_idx]
            vid = info["vehicle_id"]
            dtype = info["data_type"]

            vehicle_scores.setdefault(vid, {"old": [], "new": []})
            vehicle_scores[vid][dtype].append(
                (info["local_idx"], score)
            )

        for vid, scores in vehicle_scores.items():
            cache = cache_manager.get_vehicle_cache(vid)

            scores["old"].sort()
            scores["new"].sort()

            cache["quality_scores"] = (
                [s for _, s in scores["old"]] +
                [s for _, s in scores["new"]]
            )  