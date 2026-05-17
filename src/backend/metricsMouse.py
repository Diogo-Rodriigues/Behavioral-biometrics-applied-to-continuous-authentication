import sys
import pandas as pd
import json
import numpy as np
import random
import re
import os

# ======================================================= #
#                          PATHS                          #
# ======================================================= #

PATH = "../datasets/Mouse-Dynamics/AuthorizedUsers/"
PATH_CSV = "../datasets/ProcessedData/"

# ======================================================= #
#                     Aux - Functions                     #
# ======================================================= #

def saveCSV(df, userId, userList, sessionNum):

    # Como vamos usar a técnica "Impostor Selection" para garantir que os dados não ficam desiquilibrados 
    #vamos selecionar apenas um dos utilizadores além do original para fazer uma cópia
    #neste caso como apenas existem 2 utilizadores legítimos, a opção será sempre o utilizador não visado

    if userId in userList:
        userList.remove(userId)
        
    if not userList:
        impostorUser = userId + 1  # Fallback: cria um ID fictício para o impostor se não houver outros utilizadores
    else:
        impostorUser = random.choice(userList)

    df_impostor = df.copy()

    # Adicionamos a sample legítima ao dataset
    df.insert(0, 'USER_ID', userId)
    df.insert(1, 'SESSION_ID', sessionNum)
    df['is_illeggal'] = 0

    # Adicionamos a sample errada ao dataset
    df_impostor.insert(0, 'USER_ID', impostorUser)
    df_impostor.insert(1, 'SESSION_ID', sessionNum)
    df_impostor['is_illeggal'] = 1

    trueUserCSV_path = PATH_CSV + f"mouse_dataset_processado_user{userId}.csv"
    impostorUserCSV_path = PATH_CSV + f"mouse_dataset_processado_user{impostorUser}.csv"

    if os.path.exists(trueUserCSV_path):
        df.to_csv(trueUserCSV_path, mode='a', index = False, header = False)
    else:
        df.to_csv(trueUserCSV_path, index=False)

    if os.path.exists(impostorUserCSV_path):
        df_impostor.to_csv(impostorUserCSV_path, mode='a', index = False, header = False)
    else:
        df_impostor.to_csv(impostorUserCSV_path, index=False)

    return

def getMetrics(df, file_label_is_illegal = None):

    # Garantir nomes consistentes e remover espaços nos cabeçalhos, caso existam
    df.columns = df.columns.str.strip()

    # Ordenar cronologicamente
    df = df.sort_values(by='client timestamp')

    # 2. Calcular Derivadas Base (Tempo e Distância)
    # A diferença de tempo (dt) em segundos
    df['dt'] = df['client timestamp'].diff().fillna(0)
    
    # Evitar divisão por zero (Adicionamos um epsilon, ou consideramos dt = 0.001 se for muito pequeno)
    df['dt'] = np.where(df['dt'] < 0.001, 0.001, df['dt'])

    # Diferença espacial (dx, dy) e distância euclidiana percorrida
    df['dx'] = df['x'].diff().fillna(0)
    df['dy'] = df['y'].diff().fillna(0)
    df['distance'] = np.sqrt(df['dx']**2 + df['dy']**2)
    
    # --- 1. Velocidade e Aceleração ---
    df['velocity'] = df['distance'] / df['dt']
    df['acceleration'] = df['velocity'].diff().fillna(0) / df['dt']
    
    # --- 2. Curvatura e Ângulo de Movimento ---
    # Calcular o ângulo de movimento (em radianos) usando arctan2
    df['angle'] = np.arctan2(df['dy'], df['dx'])
    
    # Curvatura (Jitter) aproximada como a taxa de variação do ângulo sobre a distância
    # Quanto mais o ângulo muda a cada movimento, maior a curvatura.
    df['angle_change'] = df['angle'].diff().abs().fillna(0)
    df['curvature'] = df['angle_change'] / (df['distance'] + 1e-6)
    
    # --- 3. Segmentação por Ações (Strokes) ---
    # Um "stroke" de rato começa quando o rato se move e termina quando ele para.
    # Vamos definir um Idle Time (Pausa) de 200ms (0.2s) para dividir os movimentos
    df['is_idle'] = (df['dt'] > 0.2).astype(int)
    # Criar um ID único para cada "Action Stroke" contínua
    df['stroke_id'] = df['is_idle'].cumsum()
    
    # --- 4. Distribuição Direcional ---
    # Classificar o movimento em 8 direções baseadas no ângulo
    df['direction_bin'] = np.floor((df['angle'] + np.pi) / (np.pi / 4)) % 8
    
    # --- 5. Eventos de Botões (Click e Drag) ---
    # Filtrar apenas os cliques (esquerdos) para analisar retenção
    # Como o Balabit regista "Pressed" e "Released", podemos medir o Dwell Time do clique
    df['is_pressed'] = (df['state'] == 'Pressed').astype(int)
    df['is_released'] = (df['state'] == 'Released').astype(int)
    df['is_drag'] = (df['state'] == 'Drag').astype(int)
    
    # Para o projeto base, guardamos as labels do estado. O cálculo exato do Dwell  de um clique exige separar os Pressed/Released de cada stroke, o que fazemos  indiretamente analisando o estado ao longo do tempo na LSTM).

    # --- 6. Ultrapassagem do Alvo (Overshoot) ---
    # O overshoot acontece quando tu vais depressa numa direção e, logo a seguir (num dt pequeno), mudas drasticamente de direção (ângulo > 150 graus ou ~2.6 radianos) e a velocidade cai muito.
    df['is_overshoot'] = ((df['angle_change'] > 2.6) & (df['velocity'].shift(1) > df['velocity'])).astype(int)

    # Limpeza de infinitos (caso a divisão por zero tenha gerado infs)
    df = df.replace([np.inf, -np.inf], 0)
    
    # --- 7. Seleção de Features ---
    features = [
        'client timestamp', 'dt', 'dx', 'dy', 'distance', 
        'velocity', 'acceleration', 'angle', 'angle_change', 'curvature',
        'is_idle', 'stroke_id', 'direction_bin', 
        'is_pressed', 'is_released', 'is_drag', 'is_overshoot'
    ]

    df_features = df[features].copy()
    
    # Se passaste a tag (is_illegal) lida do ficheiro public_labels, nós guardamo-la!
    if file_label_is_illegal is not None:
        df_features['is_illegal'] = file_label_is_illegal
        
    return df_features

# ======================================================= #
#                           Main                          #
# ======================================================= #

def main():

    arguments = sys.argv

    if len(arguments) != 3:
        print(">> Invalid number of arguments;")
        print(">> Follow the format: \"python metricsMouse.py <file> <user Id>\"")
        return

    fileName = arguments[1]

    if fileName[-4:] != ".txt":
        print(">> Invalid file format;")
        print(">> File must be a \".txt\"")
        return

    try:
        num = int(arguments[2])
    except:
        print(">> Invalid userId format;")
        print(">> userId must be an integer")
        return

    with open("../datasets/sessionID.json", 'r') as jsonFile:
        data = json.load(jsonFile)

    userIdsList = list(map(int, data.keys()))

    if num not in userIdsList:
        print(">> Invalid userId;")
        print(f">> There is no user with the id {num}")
        return

    try:
        df = pd.read_csv(PATH + f"user{num}/" + fileName, sep=',')
    except FileNotFoundError:
        print(">> Error: FileNotFound;")
        print(f">> The file <{arguments[1]}> does not exist or is not on the correct location")

    sessionNum = int(re.search(r"_(\d)+", fileName).group(1))

    df_mouse = getMetrics(df)
    saveCSV(df_mouse, num, userIdsList, sessionNum)

    return

if __name__ == "__main__":
    main()