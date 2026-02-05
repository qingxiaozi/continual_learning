import torch
from learning.losses import ElasticNetLoss
from learning.mab_selector import MABDataSelector
from config.parameters import Config
from learning.evaluator import ModelEvaluator
from torch.utils.data import DataLoader


class EpochTrainer:
    """训练器，负责模型step训练过程"""

    def __init__(self, model, gold_model):
        self.model = model
        self.gold_model = gold_model
        self.device = Config.DEVICE

        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=1e-3,
        )
        self.loss_fn = ElasticNetLoss(
            alpha=0.5,
            l1_lambda=1e-3,
        )

    def train(self, batches, num_epochs, init_epochs, val_dataset=None, patience=5):
        """训练模型"""
        selector = MABDataSelector(num_arms=len(batches))
        best_state = self.model.state_dict()
        best_val = float("inf")
        patience_cnt = 0
        actual_epochs = 0
        for epoch in range(num_epochs):
            actual_epochs += 1
            self._train_one_epoch(
                batches=batches,
                selector=selector,
                use_mab=(epoch >= init_epochs),
            )

            if val_dataset is not None:
                val_loader = DataLoader(
                    val_dataset,
                    batch_size=Config.BATCH_SIZE,
                    shuffle=False,
                )
                _, val_loss = ModelEvaluator().evaluate_model(
                    self.model, val_loader
                )
                if val_loss < best_val:
                    best_val = val_loss
                    best_state = self.model.state_dict()
                    patience_cnt = 0
                else:
                    patience_cnt += 1
                    if patience_cnt >= patience:
                        break
        self.model.load_state_dict(best_state)

        return {
            "actual_epochs": actual_epochs,
            "selector": selector,
        }

        
    def _train_one_epoch(self, batches, selector, use_mab):
        """训练一个epoch"""
        self.model.train()

        for step in range(len(batches)):
            if use_mab:
                idx = selector.select()
            else:
                idx = step % len(batches)

            batch = batches[idx]
            inputs, targets = self._prepare_batch(batch)

            # ===== loss before =====
            with torch.no_grad():
                outputs_before = self.model(inputs)
                loss_before = torch.nn.CrossEntropyLoss(
                    outputs_before, targets
                )
            # ===== update =====
            outputs = self.model(inputs)
            loss = self.loss_fn(outputs, targets, self.model)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # ===== loss after =====
            with torch.no_grad():
                outputs_after = self.model(inputs)
                loss_after = torch.nn.functional.cross_entropy(
                    outputs_after, targets
                )

            reward = loss_before.item() - loss_after.item()
            if use_mab:
                selector.update(idx, reward)

    def _prepare_batch(self, batch):
        if isinstance(batch, (list, tuple)):
            x, y = batch
        else:
            x = batch
            with torch.no_grad():
                y = self.gold_model(x).argmax(dim=1)
        return x.to(self.device), y.to(self.device)