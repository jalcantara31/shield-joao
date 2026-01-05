import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('compressor.csv', index_col=0, parse_dates=True)

df = df.dropna(axis=1, how='all').ffill() #remover colunas vazias e preencher lacunas

df.iloc[:, :3].plot(subplots=True, figsize=(12, 8), title="Sinais Temporais do Compressor")

plt.savefig("sinais.png")

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold

class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32) #entrada no tempo t
        self.y = torch.tensor(y, dtype=torch.float32) #saída no tempo t+1
    def __len__(self): return len(self.X) #tamanho da entrada
    def __getitem__(self, i): return self.X[i], self.y[i] #entradas e saidas "i"

class ModeloPreditivo(nn.Module): #padrao pytorch
    def __init__(self, input_dim): 
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        self.loss_fn = nn.MSELoss()

    def forward(self, x):
        return self.layer(x)

data = df.values #converte o dataframe para um array com apenas numeros
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

X, y = data_scaled[:-1], data_scaled[1:, 2] # Prever a 3ª coluna (ActShaft Power) no t+1

kf = KFold(n_splits=5, shuffle=False) # Shuffle=False é vital para séries temporais!

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
    train_set = TimeSeriesDataset(X[train_idx], y[train_idx])
    val_loader = DataLoader(TimeSeriesDataset(X[val_idx], y[val_idx]), batch_size=16)
    
    model = ModeloPreditivo(input_dim=X.shape[1]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Treinamento manual com 10 épocas
    epochs = 10
    for epoch in range(epochs):
        total_loss = 0
        for X_batch, y_batch in DataLoader(train_set, batch_size=16):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            y_hat = model(X_batch)
            loss = model.loss_fn(y_hat.squeeze(), y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        print(f"Fold {fold} - Epoch {epoch+1}/{epochs} - Loss: {total_loss/len(train_set):.4f}")
    
    print(f"Fold {fold} finalizado.")