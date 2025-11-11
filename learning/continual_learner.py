import torch
import torch.optim as optim
from config.parameters import Config

class ContinualLearner:
    """持续学习器"""
    def __init__(self, model, gold_model):
        self.model = model
        self.gold_model = gold_model
        self.optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
        self.criterion = torch.nn.CrossEntropyLoss()

    def train_on_dataset(self, dataloader, num_epochs=1):
        """在数据集上训练模型"""
        self.model.train()

        for epoch in range(num_epochs):
            epoch_loss = 0.0
            for batch_idx, batch in enumerate(dataloader):
                if isinstance(batch, (list, tuple)):
                    inputs, targets = batch
                else:
                    inputs = batch
                    # 使用黄金模型生成标签
                    with torch.no_grad():
                        targets = self.gold_model(inputs).argmax(dim=1)

                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(dataloader)
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

        return avg_loss

    def compute_test_loss(self, dataloader):
        """计算测试损失"""
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for batch in dataloader:
                if isinstance(batch, (list, tuple)):
                    inputs, targets = batch
                else:
                    inputs = batch
                    with torch.no_grad():
                        targets = self.gold_model(inputs).argmax(dim=1)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                total_loss += loss.item()

        return total_loss / len(dataloader)

    def update_model_parameters(self, new_params):
        """更新模型参数"""
        self.model.load_state_dict(new_params)

    def get_model_parameters(self):
        """获取模型参数"""
        return self.model.state_dict().copy()