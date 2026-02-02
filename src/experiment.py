import torch
import torch.nn as nn
import torch.optim as optim
import mlflow
import mlflow.pytorch

# 1. Define a simple model
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc = nn.Linear(10, 1)

    def forward(self, x):
        return self.fc(x)

# 2. Setup Data and Hyperparameters
model = SimpleNet()
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)
inputs = torch.randn(64, 10)
targets = torch.randn(64, 1)

# 3. Start MLflow Run
mlflow.set_experiment("PyTorch_Basic_Experiment")

with mlflow.start_run():
    # Log hyperparameters
    mlflow.log_param("lr", 0.01)
    mlflow.log_param("epochs", 100)

    for epoch in range(100):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        # Log metrics every 10 epochs
        if epoch % 10 == 0:
            mlflow.log_metric("loss", loss.item(), step=epoch)

    # 4. Log the model
    # This saves the model, the environment (conda.yaml), and the signature
    mlflow.pytorch.log_model(model, artifact_path="model")
    print(f"Run ID: {mlflow.active_run().info.run_id}")