import torch
import torch.nn as nn
import os
import numpy as np

# ==========================================
# CÓPIA DAS CLASSES DO AUTOENCODER (Para evitar correr o script de treino)
# ==========================================

class Encoder(nn.Module):
    def __init__(self, n_features, embedding_dim=64):
        super(Encoder, self).__init__()
        self.n_features = n_features
        self.embedding_dim = embedding_dim
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=embedding_dim,
            num_layers=1,
            batch_first=True
        )

    def forward(self, x):
        _, (hidden_n, _) = self.lstm(x)
        return hidden_n.squeeze(0)

class Decoder(nn.Module):
    def __init__(self, seq_len, embedding_dim=64, n_features=1):
        super(Decoder, self).__init__()
        self.seq_len = seq_len
        self.embedding_dim = embedding_dim
        self.n_features = n_features
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=embedding_dim,
            num_layers=1,
            batch_first=True
        )
        self.output_layer = nn.Linear(embedding_dim, n_features)

    def forward(self, x):
        x = x.unsqueeze(1).repeat(1, self.seq_len, 1)
        x, _ = self.lstm(x)
        x = self.output_layer(x)
        return x

class LSTMAutoencoder(nn.Module):
    def __init__(self, seq_len, n_features, embedding_dim=64):
        super(LSTMAutoencoder, self).__init__()
        self.encoder = Encoder(n_features, embedding_dim)
        self.decoder = Decoder(seq_len, embedding_dim, n_features)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# ==========================================
# MOTOR DE DECISÃO
# ==========================================

class DecisionEngine:
    def __init__(self, keyboard_ckpt, mouse_ckpt):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model_k = LSTMAutoencoder(seq_len=30, n_features=12, embedding_dim=64).to(self.device)
        self.model_m = LSTMAutoencoder(seq_len=120, n_features=15, embedding_dim=32).to(self.device)
        
        self._load_checkpoint(self.model_k, keyboard_ckpt)
        self._load_checkpoint(self.model_m, mouse_ckpt)
        
        self.criterion = nn.MSELoss(reduction='none')
        
        # Limiares de Erro (Thresholds) para considerar Anomalia (Ajustar com base nos testes do dataset)
        self.threshold_k = 0.5 
        self.threshold_m = 0.5 
        
        # Nível de Confiança
        self.trust_score = 100.0

    def _load_checkpoint(self, model, path):
        if os.path.exists(path):
            checkpoint = torch.load(path, map_location=self.device)
            # Verifica se o pth foi guardado com o dicionário completo (epoch, optimizer, etc)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            model.eval()
            print(f">> Modelo carregado de {path}")
        else:
            print(f">> AVISO: Checkpoint {path} não encontrado. Modelo usará pesos aleatórios!")

    def _compute_reconstruction_error(self, model, tensor_x):
        model.eval()
        mean_losses = []
        with torch.no_grad():
            # Dividir em mini-batches para evitar CUDA Out of Memory
            for batch in torch.split(tensor_x, 1024):
                batch = batch.to(self.device)
                reconstruction = model(batch)
                loss = self.criterion(reconstruction, batch)
                mean_losses.extend(torch.mean(loss, dim=[1, 2]).cpu().numpy())
        return np.mean(mean_losses)

    def fine_tune(self, model_type, tensor_x, epochs=5, lr=1e-4):
        """Treina o modelo com os dados de calibração do utilizador atual para ajustar os pesos base."""
        model = self.model_k if model_type == 'keyboard' else self.model_m
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        
        print(f">> A iniciar Fine-Tuning de 1 minuto para {model_type}...")
        for epoch in range(epochs):
            batch_losses = []
            # Dividir em mini-batches para não rebentar com a memória da placa gráfica (VRAM)
            for batch in torch.split(tensor_x, 1024):
                batch = batch.to(self.device)
                optimizer.zero_grad()
                reconstruction = model(batch)
                loss = nn.MSELoss()(reconstruction, batch)
                loss.backward()
                optimizer.step()
                batch_losses.append(loss.item())
                
            epoch_loss = np.mean(batch_losses)
            
        print(f">> Fine-Tuning Concluído! Loss final: {epoch_loss:.4f}")
        model.eval()
        
        # Ajustar o limiar de anomalia dinamicamente para cada utilizador (1.5x o erro base deles)
        if model_type == 'keyboard':
            self.threshold_k = epoch_loss * 1.5
            print(f">> Limiar do Teclado ajustado para {self.threshold_k:.4f}")
        else:
            self.threshold_m = epoch_loss * 1.5
            print(f">> Limiar do Rato ajustado para {self.threshold_m:.4f}")

    def evaluate(self, tensor_k, tensor_m):
        """Avalia os blocos e atualiza o Trust Score."""
        mse_k = 0
        mse_m = 0
        
        # O Trust Score cai se o erro for muito alto (anomalia)
        # Sobe devagarinho se o erro for baixo (confirmação do dono)
        
        penalty = 0
        bonus = 0

        if tensor_k is not None:
            mse_k = self._compute_reconstruction_error(self.model_k, tensor_k)
            if mse_k > self.threshold_k:
                penalty += 15 # Penalização por comportamento estranho
            else:
                bonus += 5

        if tensor_m is not None:
            mse_m = self._compute_reconstruction_error(self.model_m, tensor_m)
            if mse_m > self.threshold_m:
                penalty += 15
            else:
                bonus += 5
                
        # Atualização (limitada a 100)
        self.trust_score = max(0.0, min(100.0, self.trust_score - penalty + bonus))
        
        print(f"--- [AVALIAÇÃO] ---")
        print(f"MSE Teclado: {mse_k:.4f} | MSE Rato: {mse_m:.4f}")
        print(f"NÍVEL DE CONFIANÇA (TRUST SCORE): {self.trust_score:.1f}%")
        
        if self.trust_score < 50.0:
            print("🚨 [ALERTA] Trust Score abaixo de 50%! (Comportamento de Intruso Detetado)")
            
        return self.trust_score
