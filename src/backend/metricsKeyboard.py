import sys
import pandas as pd
import csv
import re
import os

# ======================================================= #
#                          PATHS                          #
# ======================================================= #

PATH = "../datasets/Keystrokes/AuthorizedUsers/"
PATH_CSV = "../datasets/ProcessedData/"

# ======================================================= #
#                     Aux - Functions                     #
# ======================================================= #

def saveCSV(df, userId):

    userCSV_path = PATH_CSV + f"keystroke_dataset_processado_user{userId}.csv"

    if os.path.exists(userCSV_path):
        df.to_csv(userCSV_path, mode='a', index = False, header = False)
    else:
        df.to_csv(userCSV_path, index=False)
    return

def removeSensitiveData(df):
    # Vamos remover alguma informação que possa comprometer a confiança do utilizador na aplicação
    #além de alguma que possa não ser relevante para a autenticação
    return df.drop(columns=['LETTER', 'USER'], errors = 'ignore')

def getMetrics(df):
    
    # 1. Ordenar por sessão (TEST_SECTION_ID) e depois por tempo de pressão
    # Isto é vital para não calcularmos o "Flight Time" entre o fim de uma frase e o início de outra
    df = df.sort_values(by=['TEST_SECTION_ID', 'PRESS_TIME'])
    
    # Agrupar os dados por sessão para cálculos sequenciais seguros
    grouped = df.groupby('TEST_SECTION_ID')
    
    # --- MÉTRICAS BASE ---
    
    # Tempo de Retenção (Dwell Time): Release - Press
    df['DWELL_TIME'] = df['RELEASE_TIME'] - df['PRESS_TIME']
    
    # Tempo de Voo (Flight Time): Press(atual) - Release(anterior)
    df['PREV_RELEASE_TIME'] = grouped['RELEASE_TIME'].shift(1)
    df['FLIGHT_TIME'] = df['PRESS_TIME'] - df['PREV_RELEASE_TIME']
    
    # Tempo de Sobreposição (Overlap Time)
    # Se o Flight Time é negativo, significa que a tecla atual foi pressionada antes da anterior ser solta
    df['OVERLAP_TIME'] = df['FLIGHT_TIME'].apply(lambda x: abs(x) if x < 0 else 0)
    
    # --- LATÊNCIAS DE SEQUÊNCIAS ---
    
    # Latência de Dígrafos: Tempo entre pressões de 2 teclas consecutivas
    df['DIGRAPH_LATENCY'] = grouped['PRESS_TIME'].diff(1)
    
    # Latência de Trígrafos: Tempo entre pressões num intervalo de 3 teclas
    df['TRIGRAPH_LATENCY'] = grouped['PRESS_TIME'].diff(2)
    
    # --- FREQUÊNCIAS E PADRÕES DE TECLAS ESPECIAIS ---
    
    # Frequência de Correção: 8 (Backspace), 46 (Delete)
    df['IS_CORRECTION'] = df['KEYCODE'].isin([8, 46]).astype(int)
    
    # Teclas de Navegação: 35 (End), 36 (Home), 37 a 40 (Setas)
    df['IS_NAVIGATION'] = df['KEYCODE'].isin([35, 36, 37, 38, 39, 40]).astype(int)
    
    # Padrão Especial (ex: Ctrl + Backspace)
    # Identifica se a tecla anterior foi Ctrl (17) e a atual é Backspace (8)
    df['PREV_KEYCODE'] = grouped['KEYCODE'].shift(1)
    df['CTRL_BACKSPACE_PATTERN'] = ((df['KEYCODE'] == 8) & (df['PREV_KEYCODE'] == 17)).astype(int)
    
    # --- CADÊNCIA E VARIÂNCIA (Métricas de Janela Deslizante / Rolling) ---
    
    # Velocidade Global (WPM - Words Per Minute):
    # Assumimos que 1 palavra = 5 caracteres. Usamos uma janela das últimas 25 teclas.
    df['TIME_LAST_25_KEYS'] = grouped['PRESS_TIME'].diff(25) # tempo em milissegundos
    # Cálculo: (5 palavras) / (Tempo_em_minutos)
    df['ROLLING_WPM'] = 5 / (df['TIME_LAST_25_KEYS'] / 60000.0)
    
    # Variância Rítmica (Rhythm Variance):
    # Desvio padrão do Flight Time numa janela das últimas 10 teclas
    df['RHYTHM_VARIANCE'] = grouped['FLIGHT_TIME'].transform(lambda x: x.rolling(10, min_periods=2).std())
    
    # --- LIMPEZA E EXPORTAÇÃO ---
    
    features = [
        'PARTICIPANT_ID', 'TEST_SECTION_ID', 'PRESS_TIME', 'RELEASE_TIME',
        'DWELL_TIME', 'FLIGHT_TIME', 'OVERLAP_TIME', 
        'DIGRAPH_LATENCY', 'TRIGRAPH_LATENCY', 
        'IS_CORRECTION', 'IS_NAVIGATION', 'CTRL_BACKSPACE_PATTERN', 
        'ROLLING_WPM', 'RHYTHM_VARIANCE'
    ]

    # Preencher valores nulos gerados pelos "shifts" (inícios de frases) com 0
    df = df.fillna(0)

    df = df[features]
    
    return df

# ======================================================= #
#                           Main                          #
# ======================================================= #

def main():

    arguments = sys.argv

    if len(arguments) != 2:
        print(">> Invalid number of arguments;")
        print(">> Follow the format: \"python metricsKeyboard.py <file>\"")

    fileName = arguments[1]

    if fileName[-4:] != ".txt":
        print(">> Invalid file format;")
        print(">> File must be a \".txt\"")

    try:
        df = pd.read_csv(
            PATH + fileName, 
            sep='\t', 
            encoding='latin-1',       # Ignora erros de caracteres estranhos do utf-8
            quoting=csv.QUOTE_NONE,   # Resolve o erro do "EOF inside string" (ignora aspas)
            on_bad_lines='skip',      # Resolve o erro do "saw 13 fields" (salta as linhas partidas)
            low_memory=False
            )
    except FileNotFoundError:
        print(">> Error: FileNotFound;")
        print(f">> The file <{arguments[1]}> does not exist or is not on the correct location")
    
    userId = int(re.search(r"id(\d)+", fileName).group(1))

    df_keystrokes = getMetrics(df)
    # df_keystrokes = removeSensitiveData(df_keystrokes)
    saveCSV(df_keystrokes, userId)

if __name__ == "__main__":
    main()