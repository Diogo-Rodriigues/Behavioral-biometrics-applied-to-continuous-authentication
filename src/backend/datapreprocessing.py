# %% [markdown]
# ## Análise dos dados

# %%
import os
import glob
import pandas as pd
import numpy as np
from tqdm import tqdm
import csv
import gc
import random

# %% [markdown]
# ### Keystrokes

# %%
df_keystrokes = pd.read_csv("./dataset/keystrokes/24_keystrokes.txt", sep='\t')
print(df_keystrokes.columns)

# %%
df_keystrokes.head()

# %%
df_keystrokes.tail()

# %%
print(df_keystrokes.shape)

# %%
print("\n--- ESTATÍSTICAS DESCRITIVAS (Numéricas) ---")
# Transpose (.T) facilita a leitura no relatório
display(df_keystrokes.describe().T) 

print("\n--- ESTATÍSTICAS (Categóricas) ---")
display(df_keystrokes.describe(include=['O']).T)

# %%
def process_keystroke(filepath):
    # 1. Ler o ficheiro
    df = pd.read_csv(
    filepath, 
    sep='\t', 
    encoding='latin-1',       # Ignora erros de caracteres estranhos do utf-8
    quoting=csv.QUOTE_NONE,   # Resolve o erro do "EOF inside string" (ignora aspas)
    on_bad_lines='skip',      # Resolve o erro do "saw 13 fields" (salta as linhas partidas)
    low_memory=False
    )
    
    # 2. Ordenar por sessão (TEST_SECTION_ID) e depois por tempo de pressão
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
    
    return df

# %%
df_teclado = process_keystroke('./dataset/keystrokes/24_keystrokes.txt')
df_teclado.head(15)

# %%
df_teclado.tail(15)

# %%
def agrupar_todos_keystrokes(path, caminho_guardar):
    padrao_pesquisa = os.path.join(path, '*_keystrokes.txt')
    lista_ficheiros = glob.glob(padrao_pesquisa)
    
    if not lista_ficheiros:
        print(f"Nenhum ficheiro encontrado na pasta: {path}")
        return False
    
    print(f"Encontrados {len(lista_ficheiros)} ficheiros. A iniciar processamento...")
    
    # 1. Baralhar a lista para não haver enviesamentos
    random.shuffle(lista_ficheiros)

    # 2. Cortar a lista para 1/4 do tamanho
    um_quarto = len(lista_ficheiros) // 4
    lista_ficheiros = lista_ficheiros[:um_quarto]
    
    print(f"A usar apenas 1/4 dos dados. Ficheiros a processar: {len(lista_ficheiros)}")
    
    lista_dataframes = []
    primeiro_lote = True

    # 3. Iterar sobre cada ficheiro
    for ficheiro in tqdm(lista_ficheiros, desc="A processar utilizadores"):
        try:
            # Processar o ficheiro
            df_utilizador = process_keystroke(ficheiro)
            
            # GARANTIA DE PRIVACIDADE E PREVENÇÃO DE ERROS:
            # Se a coluna LETTER ou USER_INPUT ainda existirem, removemos!
            colunas_a_remover = ['LETTER', 'USER_INPUT']
            df_utilizador = df_utilizador.drop(columns=[col for col in colunas_a_remover if col in df_utilizador.columns], errors='ignore')

            # Adicionar à lista
            lista_dataframes.append(df_utilizador)
            
        except Exception as e:
            pass # Ignoramos as linhas corrompidas

        # O SEGREDO: Escrever no disco a cada 5000 utilizadores
        if len(lista_dataframes) >= 5000:
            df_chunk = pd.concat(lista_dataframes, ignore_index=True)
            df_chunk.to_csv(caminho_guardar, mode='a', header=primeiro_lote, index=False)
            
            primeiro_lote = False 
            lista_dataframes = [] 
            gc.collect()          

    # Gravar os últimos utilizadores que sobraram
    if len(lista_dataframes) > 0:
        df_chunk = pd.concat(lista_dataframes, ignore_index=True)
        df_chunk.to_csv(caminho_guardar, mode='a', header=primeiro_lote, index=False)
            
    print(f"\nTodos os lotes foram gravados com sucesso em: {caminho_guardar}")
    return True # Já não devolvemos o DataFrame parcial, apenas confirmamos o sucesso

# %%
pasta_keystrokes = 'dataset/keystrokes/'
caminho_guardar = 'dataset/keystroke_dataset_processado.csv'

# MUITO IMPORTANTE: Se o ficheiro já existir de uma execução anterior falhada, apaga-o.
# Senão o 'mode="a"' vai adicionar dados ao ficheiro velho!
if os.path.exists(caminho_guardar):
    os.remove(caminho_guardar)

# Executar a função
sucesso = agrupar_todos_keystrokes(pasta_keystrokes, caminho_guardar)

if sucesso:
    print("\nO dataset completo (CSV) está pronto!")

# %%
df_test = pd.read_csv("./dataset/keystroke_dataset_processado.csv")
df_test.head()

# %%
print(df_test.shape)

# %% [markdown]
# ### Mouse

# %%
df_mouse = pd.read_csv("dataset/Mouse-Dynamics/training_files/user7/session_1060325796")
print(df_mouse.columns)

# %%
df_mouse.head()

# %%
df_mouse.tail()

# %%
print(df_mouse.shape)

# %%
print("\n--- ESTATÍSTICAS DESCRITIVAS (Numéricas) ---")
# Transpose (.T) facilita a leitura no relatório
display(df_mouse.describe().T) 

print("\n--- ESTATÍSTICAS (Categóricas) ---")
display(df_mouse.describe(include=['O']).T)

# %%
def process_mouse_file(filepath, file_label_is_illegal=None):
    # 1. Ler o ficheiro. Segundo o teu exemplo, as colunas são:
    # record timestamp, client timestamp, button, state, x, y
    df = pd.read_csv(filepath, sep=',')
    
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

# %%
df_rato = process_mouse_file('dataset/Mouse-Dynamics/training_files/user7/session_1060325796')
df_rato.head(10)

# %%
def construir_dataset_rato_global(diretorio_base):
    # 1. Ler o ficheiro de labels (public_labels.txt)
    caminho_labels = os.path.join(diretorio_base, 'public_labels.csv')
    
    print("A ler o mapeamento de labels...")
    # Assumimos separação por vírgula. Ajusta o 'sep' se for por tabulação ('\t')
    df_labels = pd.read_csv(caminho_labels, sep=',') 
    
    # Criar um dicionário para procura ultra-rápida. 
    # Exemplo: {'session_0061629194': 1, 'session_1060325796': 0}
    dicionario_labels = {}
    for index, row in df_labels.iterrows():
        # A primeira coluna costuma ser o filename. Extraímos apenas o nome da sessão final.
        nome_ficheiro = str(row.iloc[0]).split('/')[-1].replace('.csv', '') 
        label = int(row.iloc[1]) # 0 (Dono) ou 1 (Ilegal)
        dicionario_labels[nome_ficheiro] = label

    # 2. Procurar recursivamente por todas as sessões dentro da pasta Mouse-Dynamics
    # Isto vai procurar dentro de test_files/userX/ e training_files/userX/
    padrao_pesquisa = os.path.join(diretorio_base, '**', 'session_*')
    lista_ficheiros = glob.glob(padrao_pesquisa, recursive=True)
    
    # Filtrar apenas ficheiros (para o caso de haver pastas com nome parecido)
    lista_ficheiros = [f for f in lista_ficheiros if os.path.isfile(f)]
    
    if not lista_ficheiros:
        print("Erro: Nenhum ficheiro encontrado. Verifica o caminho da pasta.")
        return None
        
    print(f"Encontradas {len(lista_ficheiros)} sessões de rato. A iniciar extração cinemática...")
    
    lista_dataframes = []
    
    # 3. Iterar e extrair as features matemáticas com a barra de progresso
    for ficheiro in tqdm(lista_ficheiros, desc="A processar Rato"):
        try:
            # Extrair a sessão e o ID do utilizador do próprio caminho do ficheiro
            # Como estás em Windows, usamos os.sep para dividir o caminho ('\')
            partes_caminho = ficheiro.split(os.sep)
            nome_sessao = partes_caminho[-1]                  # Ex: 'session_0061629194'
            user_id = partes_caminho[-2].replace('user', '')  # Transforma 'user7' em '7'
            
            # 4. Ir buscar a label ao nosso dicionário
            # Se não encontrar a sessão no txt (dados incompletos), ignora e salta
            if nome_sessao not in dicionario_labels:
                continue 
                
            is_illegal_label = dicionario_labels[nome_sessao]
            
            # 5. Processar o ficheiro com a nossa função avançada
            df_sessao = process_mouse_file(ficheiro, file_label_is_illegal=is_illegal_label)
            
            # Adicionar as colunas de identificação no início do DataFrame
            df_sessao.insert(0, 'USER_ID', user_id)
            df_sessao.insert(1, 'SESSION_ID', nome_sessao)
            
            lista_dataframes.append(df_sessao)
            
        except Exception as e:
            # Tal como no teclado, se um ficheiro estiver corrompido, salta à frente.
            pass

    # 6. Fundir tudo num único ecrã
    print("\nA fundir todos os dados cinemáticos num único DataFrame mestre...")
    df_final = pd.concat(lista_dataframes, ignore_index=True)
    
    return df_final

# %%
# Define a pasta raiz do rato (onde estão as pastas test_files e training_files)
pasta_rato = 'dataset/Mouse-Dynamics/'

df_global_rato = construir_dataset_rato_global(pasta_rato)

if df_global_rato is not None:
    print(f"\nProcessamento de rato concluído!")
    print(f"Total de registos (linhas): {len(df_global_rato)}")
    print(f"Total de sessões processadas: {df_global_rato['SESSION_ID'].nunique()}")
    print(f"Distribuição Legaís (0) vs Ilegais (1):\n{df_global_rato['is_illegal'].value_counts()}")
    
    # Guardar em formato de alta performance (fortemente recomendado)
    caminho_guardar = 'dataset/mouse_dataset_processado.parquet'
    print(f"A guardar ficheiro final em: {caminho_guardar}")
    df_global_rato.to_parquet(caminho_guardar, index=False)
    
    # Se ainda não tiveres o pyarrow instalado, usa CSV (vai demorar mais a gravar)
    # df_global_rato.to_csv('dataset/mouse_dataset_processado.csv', index=False)
    print("Ficheiro guardado e pronto para a Inteligência Artificial!")

# %%
df_test_mouse = pd.read_parquet("./dataset/mouse_dataset_processado.parquet")
df_test_mouse.head()

# %%
print("\n--- ESTATÍSTICAS DESCRITIVAS (Numéricas) ---")
# Transpose (.T) facilita a leitura no relatório
display(df_test_mouse.describe().T) 

print("\n--- ESTATÍSTICAS (Categóricas) ---")
display(df_test_mouse.describe(include=['O']).T)


