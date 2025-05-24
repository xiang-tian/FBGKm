import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tqdm import tqdm

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 读取数据
data = pd.read_excel(r"./data.xlsx")  
features = data.columns[1:] 
target = data.columns[0]  

# 将所有列转换为数值类型
data[features] = data[features].apply(pd.to_numeric, errors='coerce')
data[target] = pd.to_numeric(data[target], errors='coerce')

# 数据预处理
scaler = MinMaxScaler()
data[features] = scaler.fit_transform(data[features])  
data[target] = scaler.fit_transform(data[[target]])  

# 创建时间窗口数据集
def create_dataset(dataset, look_back=12):
    dataX, dataY = [],[]
    for i in range(len(dataset) - look_back):
        a = dataset[i:(i + look_back), :-1] 
        dataX.append(a)
        dataY.append(dataset[i + look_back, -1])  
    return np.array(dataX), np.array(dataY)

look_back = 19
learning_rate = 1.44306669e-03    
num_epochs = 303  
batch_size = 61
l2_lambda = 0.001  
dropout = 0.3

X, y = create_dataset(data.values, look_back)
# 划分数据集
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42) 
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# 转换为Tensor
X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(device)
X_val = torch.tensor(X_val, dtype=torch.float32).to(device)
y_val = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1).to(device)
X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1).to(device)

# 创建DataLoader
train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=False)
val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size, shuffle=False)
test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size, shuffle=False)

# 定义LSTM模型
class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout=0.2):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_dim, 
            hidden_size=hidden_dim, 
            num_layers=num_layers, 
            batch_first=True, 
            bidirectional=False, 
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.fc = nn.Linear(hidden_dim, output_dim)   
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        lstm_out, _ = self.lstm(x, (h0, c0)) 
        out = self.fc(lstm_out[:, -1, :])  
        return out

# 定义模型参数
input_dim = len(features) 
hidden_dim = 50
num_layers = 2
output_dim = 1

# 实例化LSTM模型
model = LSTM(input_dim, hidden_dim, num_layers, output_dim, dropout).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 定义计算 NSE 的函数
def calculate_nse(y_true, y_pred):
    numerator = np.sum((y_true - y_pred) ** 2)
    denominator = np.sum((y_true - np.mean(y_true)) ** 2)
    nse = 1 - (numerator / denominator)
    return nse

# 初始化变量
train_losses = []
val_losses = []
best_val_loss = float('inf')

for epoch in range(num_epochs):
    model.train()
    epoch_train_loss = 0.0
    train_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs} - Training', leave=False)
    for i, (inputs, labels) in enumerate(train_bar):
        optimizer.zero_grad()
        outputs = model(inputs)
        mse_loss = criterion(outputs, labels)
        l2_norm = sum(p.pow(2.0).sum() for p in model.parameters())
        loss = mse_loss + l2_lambda * l2_norm

        loss.backward()
        optimizer.step()
        epoch_train_loss += loss.item()
        train_bar.set_postfix({'Loss': f"{loss.item():.4f}"})
    
    # 计算训练集的平均损失
    average_train_loss = epoch_train_loss / len(train_loader)
    train_losses.append(average_train_loss)
    
    # 验证集
    model.eval()
    epoch_val_loss = 0.0
    val_bar = tqdm(val_loader, desc=f'Epoch {epoch + 1}/{num_epochs} - Validation', leave=False)
    with torch.no_grad():
        for inputs, labels in val_bar:
            outputs = model(inputs)
            mse_loss = criterion(outputs, labels)
            
            # L2 正则化
            l2_norm = sum(p.pow(2.0).sum() for p in model.parameters())
            loss = mse_loss + l2_lambda * l2_norm

            epoch_val_loss += loss.item()
            val_bar.set_postfix({'Loss': f"{loss.item():.4f}"})
    
    # 计算验证集的平均损失
    average_val_loss = epoch_val_loss / len(val_loader)
    val_losses.append(average_val_loss)
    
    # 在最后一个 epoch 计算训练集和验证集的 NSE、MSE 和 RMSE
    if epoch == num_epochs - 1:
        # 训练集
        model.train()
        train_preds, train_labels = [], []
        with torch.no_grad():
            for inputs, labels in train_loader:
                outputs = model(inputs)
                train_preds.extend(outputs.detach().cpu().numpy())
                train_labels.extend(labels.detach().cpu().numpy())
        
        train_preds = np.array(train_preds).flatten()
        train_labels = np.array(train_labels).flatten()
        train_nse = calculate_nse(train_labels, train_preds)
        train_mse = mean_squared_error(train_labels, train_preds)
        train_rmse = np.sqrt(train_mse)
        
        # 验证集
        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                val_preds.extend(outputs.detach().cpu().numpy())
                val_labels.extend(labels.detach().cpu().numpy())
        
        val_preds = np.array(val_preds).flatten()
        val_labels = np.array(val_labels).flatten()
        val_nse = calculate_nse(val_labels, val_preds)
        val_mse = mean_squared_error(val_labels, val_preds)
        val_rmse = np.sqrt(val_mse)
        
        # 输出结果
        print("\nFinal Training Metrics:")
        print(f"NSE: {train_nse:.4f}, MSE: {train_mse:.4f}, RMSE: {train_rmse:.4f}")
        
        print("\nFinal Validation Metrics:")
        print(f"NSE: {val_nse:.4f}, MSE: {val_mse:.4f}, RMSE: {val_rmse:.4f}")

# 测试模型
model.eval()
y_pred = []
y_true = []
test_bar = tqdm(test_loader, desc='Testing', leave=True)
with torch.no_grad():
    for inputs, labels in test_bar:
        outputs = model(inputs)
        y_pred.extend(outputs.cpu().numpy())
        y_true.extend(labels.cpu().numpy())

y_pred = np.array(y_pred)
y_test = np.array(y_true)

# 反归一化
y_pred = scaler.inverse_transform(y_pred)
y_test = scaler.inverse_transform(y_test)

combined_df = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_pred.flatten()})
combined_df.to_excel('LSTM.xlsx', index=False)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

# 计算 NSE
y_true_mean = np.mean(y_test)
numerator = np.sum((y_test - y_pred) ** 2)
denominator = np.sum((y_test - y_true_mean) ** 2)
nse = 1 - (numerator / denominator)

# 打印结果
print(f'MSE: {mse:.4f}')
print(f'RMSE: {rmse:.4f}')
print(f'NSE: {nse:.4f}')