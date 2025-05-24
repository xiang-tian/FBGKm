import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score
from sklearn.model_selection import train_test_split
import os
import matplotlib.pyplot as plt
from kan import KAN
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

data = pd.read_excel(r"./data.xlsx")

features = data.columns[1:]
target = data.columns[0]  

data[features] = data[features].apply(pd.to_numeric, errors='coerce')
data[target] = pd.to_numeric(data[target], errors='coerce')

scaler = MinMaxScaler()
data[features] = scaler.fit_transform(data[features]) 
data[target] = scaler.fit_transform(data[[target]])  

data_train_val = data[:-20]
data_predict = data[-20:]

def create_dataset(dataset, look_back=12):
    dataX, dataY = [],[]
    for i in range(len(dataset) - look_back):
        a = dataset[i:(i + look_back), :-1]  
        dataX.append(a)
        dataY.append(dataset[i + look_back, -1]) 
    return np.array(dataX), np.array(dataY)

look_back = 26
learning_rate = 8.44306669e-04    
num_epochs = 303  
batch_size = 61
l2_lambda = 1.0e-03
X, y = create_dataset(data_train_val.values, look_back)

n_samples = len(X)
train_size = int(n_samples * 0.7)  
remaining = n_samples - train_size
val_size = remaining // 2        
test_size = remaining - val_size

X_train, y_train = X[:train_size], y[:train_size]
X_val, y_val = X[train_size:train_size+val_size], y[train_size:train_size+val_size]
X_test, y_test = X[train_size+val_size:], y[train_size+val_size:]

X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(device)
X_val = torch.tensor(X_val, dtype=torch.float32).to(device)
y_val = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1).to(device)
X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1).to(device)

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size, shuffle=False)
test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size, shuffle=False)

class CNN_BiGRU_KAN(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim,dropout=0.2):
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
        self.fc = KAN(width=[hidden_dim * 2, output_dim])  
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

        h0 = torch.zeros(
            self.num_layers * 2, 
            x.size(0), 
            self.hidden_dim
        ).to(x.device)  
        gru_out, _ = self.bigru(x, h0)  
        residual = self.residual_fc(x)   
        out = gru_out + residual        
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc(out[:, -1, :])     
        return out

input_dim = len(features)  
hidden_dim = 50
num_layers = 2
output_dim = 1
dropout = 0.1
model = CNN_BiGRU_KAN(input_dim, hidden_dim, num_layers, output_dim).to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

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
    
    average_train_loss = epoch_train_loss / len(train_loader)
    train_losses.append(average_train_loss)
    
    model.eval()
    epoch_val_loss = 0.0
    val_bar = tqdm(val_loader, desc=f'Epoch {epoch + 1}/{num_epochs} - Validation', leave=False)
    with torch.no_grad():
        for inputs, labels in val_bar:
            outputs = model(inputs)
            mse_loss = criterion(outputs, labels)
            l2_norm = sum(p.pow(2.0).sum() for p in model.parameters())
            loss = mse_loss + l2_lambda * l2_norm 

            epoch_val_loss += loss.item()
            val_bar.set_postfix({'Loss': f"{loss.item():.4f}"})
    average_val_loss = epoch_val_loss / len(val_loader)
    val_losses.append(average_val_loss)

    if average_val_loss < best_val_loss:
        best_val_loss = average_val_loss
    
    if (epoch + 1) % 10 == 0 or epoch == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {average_train_loss:.4f}, Val Loss: {average_val_loss:.4f}')

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

y_pred = scaler.inverse_transform(y_pred)
y_test = scaler.inverse_transform(y_test)

os.makedirs('Result', exist_ok=True)
predictions_path = os.path.join('Result', 'predictions.csv')
pd.DataFrame(y_pred, columns=['BiGRU']).to_csv(predictions_path, index=False)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

y_true_mean = np.mean(y_test)
numerator = np.sum((y_test - y_pred) ** 2)
denominator = np.sum((y_test - y_true_mean) ** 2)
nse = 1 - (numerator / denominator)
print(f'MSE: {mse:.4f}')
print(f'RMSE: {rmse:.4f}')
print(f'NSE: {nse:.4f}')