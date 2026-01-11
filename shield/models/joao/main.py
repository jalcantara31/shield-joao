import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('compressor.csv', index_col=0, parse_dates=True) #carrega os dados utilizando a primeira coluna como indice e reconhece ela como data

df = df.dropna(axis=1, how='all').ffill() #remover colunas vazias e preencher lacunas

df.iloc[:, :3].plot(subplots=True, figsize=(12, 8), title="Sinais Temporais do Compressor") #plotagem apenas das 3 primeiras colunas

plt.savefig("sinais.png")


# O modelo de rede neural a seguir utiliza como entrada (X) todas as linhas, exceto a ultima, e a saida (y) pega a partir da segunda linha ate a ultima,
# porem apenas da segunda coluna. A partir disso o modelo esta prevendo apenas a coluna de indice 2
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

class ModeloPreditivo(nn.Module):       #padrao pytorch
    def __init__(self, input_dim): 
        super().__init__()
        self.layer = nn.Sequential(     #criacao da rede neural sequencial
            nn.Linear(input_dim, 64),   #criacao dos neuronios 
            nn.ReLU(),                  #remove valores negativos (->0)
            nn.Linear(64, 1)            #transforma os neuronios em 1 saida
        )
        self.loss_fn = nn.MSELoss()     #calcula o erro quadratico medio

    def forward(self, x):               #funcao que da a entrada X para passar pela rede e retorna o resultado
        return self.layer(x) 

data = df.values                        #converte o dataframe para um array com apenas numeros 
scaler = StandardScaler()               #adapta para media 0 e desvio padrao 1
data_scaled = scaler.fit_transform(data) #aplica a adaptacao dos dados da linha anterior nos nossos dados

X = data_scaled[:-1] #os dados de entrada (X) terao todas as linhas, menos a ultima
y = data_scaled[1:, 2] #os dados target (y) possui apenas a segunda coluna a partir da segunda linha

kf = KFold(n_splits=5, shuffle=False) #shufle=False é vital para séries temporais!!

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")       #define a GPU como dispositivo para treino ou entao a CPU caso a GPU
                                                                            #nao esteja disponivel
print(f"Dispositivo: {device}\n")

fig, axes = plt.subplots(kf.n_splits, 1, figsize=(10, 15), sharex=True)
fig.suptitle('Curva de Aprendizado (Loss) por Fold')

for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
    train_set = TimeSeriesDataset(X[train_idx], y[train_idx])               #cria o dataset de treino
    val_set = TimeSeriesDataset(X[val_idx], y[val_idx])

    train_loader = DataLoader(train_set, batch_size=16, shuffle=False)
    val_loader = DataLoader(TimeSeriesDataset(X[val_idx], y[val_idx]), batch_size=16)       #cria o dataloader para validacao com tamanho 16

    model = ModeloPreditivo(input_dim=X.shape[1]).to(device)        #define o modelo para entrada igual ao numero de colunas e move p/ o device
    optimizer = optim.Adam(model.parameters(), lr=0.001)            #define o otimizador Adam para atualizar os parametros do modelo
                                                                    #lr -> a taxa de aprendizado (learning rate)
    
    train_history = []
    val_history = []

    plt.figure(figsize=(10,6))

    epochs = 10            #numero de epochs
    for epoch in range(epochs):
        model.train()     
        total_train_loss = 0      
        for X_batch, y_batch in train_loader:   #divide os dados de treino em lotes (batchs) de 16
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)   #move os batches para o device
            optimizer.zero_grad()                                       #zera os gradientes anteriores para recomecar o treinamento
            y_hat = model(X_batch)                                      #passa o batch de entrada para o modelo ; y_hat -> previdicao
            loss = model.loss_fn(y_hat.squeeze(), y_batch)              #calcula a perda entre a previsao e o valor real ; 
                                                                        #.squeeze transforma as 16 saidas em um vetor com 16 valores
            loss.backward()         #calcula os gradientes por backpropagration
            optimizer.step()        #atualiza os pesos com os gradientes calculados
            total_train_loss += loss.item()       #calcula a perda media da epoch
        
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for X_val, y_val in val_loader:
                X_val, y_val = X_val.to(device), y_val.to(device)
                y_hat_val = model(X_val)
                loss_v = model.loss_fn(y_hat_val.squeeze(), y_val)
                total_val_loss+=loss_v.item()

        avg_train = total_train_loss/len(train_loader)
        avg_val = total_val_loss/len(val_loader)

        train_history.append(avg_train)
        val_history.append(avg_val)

        print(f"Fold {fold} | Ep {epoch+1} | Train Loss: {avg_train:.4f} | Val Loss: {avg_val:.4f}")    

    plt.plot(train_history, label='Treino', color='blue')
    plt.plot(val_history, label='Validação', color='red', linestyle='--')
    plt.title(f'Funcao Custo - Fold {fold}')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(alpha=0.3)
    nome_arquivo = f"funcao_custo_fold_{fold}.png"
    plt.savefig(nome_arquivo)

plt.close()


#plotar funcao custo 
#distancia media do valor real pro valor previsto
#avaliar se vale a pena salvar o modelo