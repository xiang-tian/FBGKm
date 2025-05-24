import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score
from sklearn.model_selection import train_test_split
import os
from mealpy import FloatVar, AGTO
from kan import KAN
from tqdm import tqdm 

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 加载数据
data = pd.read_excel(r"./data.xlsx")
features = data.columns[1:] 
target = data.columns[0] 

data[features] = data[features].apply(pd.to_numeric, errors='coerce')
data[target] = pd.to_numeric(data[target], errors='coerce')

def create_dataset(dataset, look_back=12):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back):
        a = dataset[i:(i + look_back), :-1]
        dataX.append(a)
        dataY.append(dataset[i + look_back, -1])
    return np.array(dataX), np.array(dataY)

class CNN_BiGRU_KAN(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout=0.1):
        super(CNN_BiGRU_KAN, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.conv1 = nn.Conv1d(input_dim, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.residual_fc = nn.Linear(128, hidden_dim * 2)
        self.bigru = nn.GRU(
            input_size=128, 
            hidden_size=hidden_dim, 
            num_layers=num_layers, 
            batch_first=True, 
            bidirectional=True, 
            dropout=dropout if num_layers > 1 else 0
        )
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = x.permute(0, 2, 1)
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_dim).to(x.device)
        gru_out, _ = self.bigru(x, h0)
        residual = self.residual_fc(x)
        
        out = gru_out + residual
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc(out[:, -1, :])
        return out

# 定义目标函数
def objective_function(solution):
    look_back, learning_rate, num_epochs, batch_size = solution
    look_back = int(round(look_back))
    num_epochs = int(round(num_epochs))
    batch_size = int(round(batch_size))

    l1_lambda = 1e-4  
    dropout = 0.2     
    X, y = create_dataset(data.values, look_back)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)
     # 转换为Tensor
    X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(device)
    X_val = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1).to(device)

    # 创建DataLoader
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size, shuffle=False)

    input_dim = X_train.shape[2]
    output_dim = 1
    hidden_dim = 64  
    num_layers = 2   

    model = CNN_BiGRU_KAN(input_dim, hidden_dim, num_layers, output_dim, dropout=dropout).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=l1_lambda)
    criterion = nn.MSELoss()

    for epoch in range(num_epochs):
        model.train()
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
    
    average_val_loss = val_loss / len(val_loader)
    return average_val_loss

problem_dict = {
    "bounds": FloatVar(lb=[12, 5e-5, 100, 16], ub=[64, 1e-3, 600, 64], name=["look_back", "learning_rate", "num_epochs", "batch_size"]),
    "obj_func": objective_function,
    "minmax": "min",
}

model = AGTO.MGTO(epoch=200, pop_size=50, pp=0.03,save_checkpoint=False)

# 运行优化
g_best = model.solve(problem_dict)
print(model.history.list_global_best)
print(model.history.list_current_best)
print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")

best_look_back, best_learning_rate, best_num_epochs, best_batch_size = g_best.solution

# 可视化结果
model.history.save_global_objectives_chart(filename="Result/goc")
model.history.save_local_objectives_chart(filename="Result/loc")
model.history.save_global_best_fitness_chart(filename="Result/gbfc")
model.history.save_local_best_fitness_chart(filename="Result/lbfc")
model.history.save_runtime_chart(filename="Result/rtc")
model.history.save_exploration_exploitation_chart(filename="Result/eec")
model.history.save_diversity_chart(filename="Result/dc")
