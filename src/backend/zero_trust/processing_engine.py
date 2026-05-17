import sys
import os
import pandas as pd
import numpy as np
import torch
import joblib
from sklearn.preprocessing import RobustScaler

# Configurar imports da pasta mãe (src/backend)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.abspath(os.path.join(BASE_DIR, '..'))
ROOT_DIR = os.path.abspath(os.path.join(BACKEND_DIR, '..', '..'))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

import metricsKeyboard
import metricsMouse

class ProcessingEngine:
    def __init__(self, dataset_dir):
        self.dataset_dir = dataset_dir
        self.scaler_teclado = RobustScaler()
        self.scaler_rato = RobustScaler()
        
        self.features_teclado = [
            'PRESS_TIME', 'RELEASE_TIME', 'DWELL_TIME', 'FLIGHT_TIME', 'OVERLAP_TIME', 
            'DIGRAPH_LATENCY', 'TRIGRAPH_LATENCY', 'IS_CORRECTION', 'IS_NAVIGATION', 
            'CTRL_BACKSPACE_PATTERN', 'ROLLING_WPM', 'RHYTHM_VARIANCE'
        ]
        
        self.features_rato = [
            'dt', 'dx', 'dy', 'distance', 'velocity', 'acceleration', 
            'angle', 'angle_change', 'curvature', 'is_idle', 
            'direction_bin', 'is_pressed', 'is_released', 'is_drag', 'is_overshoot'
        ]
        
        self.seq_len_teclado = 30
        self.seq_len_rato = 120
        
        self.scaler_teclado = None
        self.scaler_rato = None
        self._load_scalers()

    def _load_scalers(self):
        print(">> A carregar Scalers dos ficheiros .pkl...")
        k_pkl = os.path.join(BASE_DIR, 'scaler_teclado.pkl')
        if os.path.exists(k_pkl):
            self.scaler_teclado = joblib.load(k_pkl)
            print(">> scaler_teclado.pkl carregado!")
        else:
            print(f"AVISO: {k_pkl} não encontrado! O ML vai falhar.")

        m_pkl = os.path.join(BASE_DIR, 'scaler_rato.pkl')
        if os.path.exists(m_pkl):
            self.scaler_rato = joblib.load(m_pkl)
            print(">> scaler_rato.pkl carregado!")
        else:
            print(f"AVISO: {m_pkl} não encontrado! O ML vai falhar.")

    def _rotate_csv(self, filepath, max_size_mb=10):
        """Creates a new file if the current one exceeds max_size_mb."""
        if not os.path.exists(filepath):
            return filepath
            
        size_mb = os.path.getsize(filepath) / (1024 * 1024)
        if size_mb > max_size_mb:
            base, ext = os.path.splitext(filepath)
            i = 1
            while os.path.exists(f"{base}_part{i}{ext}"):
                i += 1
            new_path = f"{base}_part{i}{ext}"
            os.rename(filepath, new_path)
            print(f">> Rotação: {filepath} excedeu {max_size_mb}MB. Renomeado para {new_path}")
            
        return filepath

    def create_sequences(self, dados_numpy, seq_length):
        xs = []
        for i in range(len(dados_numpy) - seq_length + 1):
            xs.append(dados_numpy[i:(i + seq_length)])
        return np.array(xs)

    def process_keyboard(self, chunk, overlap, user_id, session_id, output_csv):
        df = pd.DataFrame(chunk)
        df['PARTICIPANT_ID'] = user_id
        df['TEST_SECTION_ID'] = session_id
        
        novos = len(df) - overlap if overlap > 0 else len(df)
        print(f"\n>> Processando lote de {novos} novos eventos de teclado (Contexto: {overlap})...")
        df_metrics_full = metricsKeyboard.getMetrics(df)
        
        if df_metrics_full.empty:
            return None, None
            
        df_metrics_full = df_metrics_full.dropna(subset=self.features_teclado)
        if df_metrics_full.empty:
            return None, None

        # Separar os novos eventos para gravar no CSV (sem duplicar o contexto/overlap)
        df_csv = df_metrics_full.iloc[overlap:] if overlap > 0 else df_metrics_full
        
        if not df_csv.empty:
            self._rotate_csv(output_csv)
            if os.path.exists(output_csv):
                df_csv.to_csv(output_csv, mode='a', index=False, header=False)
            else:
                df_csv.to_csv(output_csv, index=False)

        # Usar os dados TODOS (Full) para criar a janela de ML (porque precisamos de seq_len inteiras)
        dados_brutos = df_metrics_full[self.features_teclado].values
        dados_escalados = self.scaler_teclado.transform(dados_brutos)
        
        if len(dados_escalados) < self.seq_len_teclado:
            return None, df_csv # Sem dados suficientes para 1 sequência completa
            
        seqs = self.create_sequences(dados_escalados, self.seq_len_teclado)
        tensor_x = torch.tensor(seqs, dtype=torch.float32)
        return tensor_x, df_csv

    def process_mouse(self, chunk, overlap, user_id, session_id, output_csv):
        df = pd.DataFrame(chunk)
        
        novos = len(df) - overlap if overlap > 0 else len(df)
        print(f">> Processando lote de {novos} novos eventos de rato (Contexto: {overlap})...")
        df_metrics_full = metricsMouse.getMetrics(df)
        
        if df_metrics_full.empty:
            return None, None
            
        df_metrics_full = df_metrics_full.dropna(subset=self.features_rato)
        if df_metrics_full.empty:
            return None, None

        df_metrics_full.insert(0, 'USER_ID', user_id)
        df_metrics_full.insert(1, 'SESSION_ID', session_id)

        df_csv = df_metrics_full.iloc[overlap:] if overlap > 0 else df_metrics_full

        if not df_csv.empty:
            self._rotate_csv(output_csv)
            if os.path.exists(output_csv):
                df_csv.to_csv(output_csv, mode='a', index=False, header=False)
            else:
                df_csv.to_csv(output_csv, index=False)

        # Build Tensors (Com o contexto completo)
        dados_brutos = df_metrics_full[self.features_rato].values
        dados_escalados = self.scaler_rato.transform(dados_brutos)
        
        dados_escalados = np.clip(dados_escalados, a_min=-10, a_max=10)
        
        if len(dados_escalados) < self.seq_len_rato:
            return None, df_csv
            
        seqs = self.create_sequences(dados_escalados, self.seq_len_rato)
        tensor_x = torch.tensor(seqs, dtype=torch.float32)
        return tensor_x, df_csv
