import pandas as pd
import os
import joblib
from sklearn.preprocessing import RobustScaler

# Define paths
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
dataset_dir = os.path.join(ROOT_DIR, 'dataset')

features_teclado = [
    'PRESS_TIME', 'RELEASE_TIME', 'DWELL_TIME', 'FLIGHT_TIME', 'OVERLAP_TIME', 
    'DIGRAPH_LATENCY', 'TRIGRAPH_LATENCY', 'IS_CORRECTION', 'IS_NAVIGATION', 
    'CTRL_BACKSPACE_PATTERN', 'ROLLING_WPM', 'RHYTHM_VARIANCE'
]

features_rato = [
    'dt', 'dx', 'dy', 'distance', 'velocity', 'acceleration', 
    'angle', 'angle_change', 'curvature', 'is_idle', 
    'direction_bin', 'is_pressed', 'is_released', 'is_drag', 'is_overshoot'
]

print("=== GERAÇÃO DOS SCALERS (.pkl) ===")

# Teclado
k_path = os.path.join(dataset_dir, 'keystroke_dataset_processado.csv')
if os.path.exists(k_path):
    print(f">> A carregar Teclado (Isto pode demorar devido ao tamanho)...")
    # Carregamos apenas as colunas que nos importam para poupar imensa RAM!
    df_k = pd.read_csv(k_path, usecols=features_teclado)
    df_k = df_k.dropna(subset=features_teclado)
    scaler_teclado = RobustScaler()
    print(">> A ajustar o Scaler ao Teclado...")
    scaler_teclado.fit(df_k[features_teclado].values)
    joblib.dump(scaler_teclado, os.path.join(os.path.dirname(__file__), 'scaler_teclado.pkl'))
    print(">> scaler_teclado.pkl gravado com sucesso!")
    del df_k # libertar RAM

# Rato
m_path = os.path.join(dataset_dir, 'mouse_dataset_processado.parquet')
if os.path.exists(m_path):
    print(f">> A carregar Rato...")
    df_m = pd.read_parquet(m_path, columns=features_rato)
    df_m = df_m.dropna(subset=features_rato)
    scaler_rato = RobustScaler()
    print(">> A ajustar o Scaler ao Rato...")
    scaler_rato.fit(df_m[features_rato].values)
    joblib.dump(scaler_rato, os.path.join(os.path.dirname(__file__), 'scaler_rato.pkl'))
    print(">> scaler_rato.pkl gravado com sucesso!")
    del df_m

print("=== CONCLUÍDO ===")
