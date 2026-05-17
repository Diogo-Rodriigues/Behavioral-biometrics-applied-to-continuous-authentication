import sys
import os
import json
import time
import re
import pandas as pd
from pynput import keyboard, mouse
from pynput.keyboard import Key

# =====================================================================
#                      ENVIRONMENT SETUP
# =====================================================================

# Change Current Working Directory to src/backend to respect relative paths
# of metricsKeyboard.py and metricsMouse.py ("../datasets/...")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.abspath(os.path.join(BASE_DIR, '..', 'src', 'backend'))

if not os.path.exists(BACKEND_DIR):
    print(f"ERRO: A diretoria {BACKEND_DIR} não foi encontrada!")
    sys.exit(1)

os.chdir(BACKEND_DIR)
sys.path.insert(0, BACKEND_DIR)

try:
    import metricsKeyboard
    import metricsMouse
except ModuleNotFoundError as e:
    print(f"Erro ao importar módulos. sys.path atual: {sys.path}")
    raise e

IDS_PATH = os.path.join(BASE_DIR, "ids.txt")
SESSION_JSON_PATH = "../datasets/sessionID.json"

# =====================================================================
#                      USER & SESSION MANAGEMENT
# =====================================================================

def update_session_json(user_id):
    """Safely updates sessionID.json and returns the current session ID for this user"""
    try:
        with open(SESSION_JSON_PATH, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        data = {}

    user_id_str = str(user_id)
    if user_id_str not in data:
        data[user_id_str] = 0

    # Increase session count
    data[user_id_str] += 1
    
    with open(SESSION_JSON_PATH, 'w') as f:
        json.dump(data, f, indent=4)
        
    return data[user_id_str], sum(list(map(int, data.values())))

def invalidUserOption(option, maxNum):
    return option < 0 or option > maxNum

def addUser(numUsers):
    with open(IDS_PATH, 'a') as usersFile:
        valid = False
        while not valid:
            userNickname = input(">> Introduce your nickname: ")
            print(f">> Are you sure you are \"{userNickname}\"? (Y/n/q)")
            confirmation = input()
            if re.search(r"(q|quit)", confirmation, flags=re.IGNORECASE):
                print(">> Closing Script ...")
                sys.exit()
            elif re.search(r"(no|n|não|nao)", confirmation, flags=re.IGNORECASE):
                continue
            else:
                valid = True
            
            usersFile.write(str(userNickname) + ' - ' + str(numUsers + 1) + '\n')
    return numUsers + 1

def selectUser():
    if not os.path.exists(IDS_PATH):
        with open(IDS_PATH, 'w') as f:
            pass # Create if not exists

    with open(IDS_PATH, 'r') as usersFile:
        lines = [line.strip() for line in usersFile.readlines() if line.strip()]
        
    print(">> Select your User by the number: \n")
    for i, line in enumerate(lines):
        print(f"[{i}] {line}")
    print(f"[{len(lines)}] -- Create New User --")

    numUsers = len(lines)
    userOption = None
    while userOption is None:
        try:
            userOption = int(input(">> Option: "))
            if invalidUserOption(userOption, numUsers):
                print(f">> {userOption} is not an available Option, Try again")
                userOption = None
        except ValueError:
            print(">> Error: The inserted value type is not permited")

    if userOption == numUsers:
        userId = addUser(numUsers)
    else:
        userNickName, userIdStr = lines[userOption].split(" - ")
        userId = int(userIdStr)
        print(f">> Are you sure you are {userNickName}? (Y/n)")
        confirmation = input()
        if re.search(r"(no|n|não|nao)", confirmation, flags=re.IGNORECASE):
            print(">> Closing Script ...")
            sys.exit()
            
    return userId

# =====================================================================
#                      KEYBOARD LISTENER
# =====================================================================

keyboard_events = []
pressed_keys = {}

special_keys = list(Key)
key_to_id = {str(k): 1000 + i for i, k in enumerate(special_keys)}

def get_keycode(key):
    try:
        if hasattr(key, 'char') and key.char is not None:
            if len(key.char) == 1:
                return ord(key.char)
        return key_to_id.get(str(key), 9999)
    except Exception:
        return 0

is_running = True

def on_press(key):
    global is_running
    if key == keyboard.Key.esc:
        print("\n>> Stopping Capture...")
        is_running = False
        return False
    
    k_str = str(key)
    if k_str not in pressed_keys:
        pressed_keys[k_str] = time.time()
        print(f"Key {k_str} pressed")

def on_release(key):
    k_str = str(key)
    if k_str in pressed_keys:
        press_time = pressed_keys.pop(k_str)
        release_time = time.time()
        
        # Append event to memory
        keyboard_events.append({
            'PRESS_TIME': int(press_time * 1000),     # Em milissegundos
            'RELEASE_TIME': int(release_time * 1000), # Em milissegundos
            'KEYCODE': get_keycode(key)
        })
        print(f"Key {k_str} released")

# =====================================================================
#                      MOUSE LISTENER
# =====================================================================

mouse_events = []
is_clicking = {"status": False}
start_time = None

def get_mouse_client_timestamp():
    return time.time() - start_time

def on_move(x, y):
    state = "Drag" if is_clicking["status"] else "Move"
    mouse_events.append({
        'client timestamp': get_mouse_client_timestamp(),
        'button': "NoButton",
        'state': state,
        'x': x,
        'y': y
    })

def on_scroll(x, y, dx, dy):
    mouse_events.append({
        'client timestamp': get_mouse_client_timestamp(),
        'button': "Scroll",
        'state': "Up" if dy > 0 else "Down",
        'x': x,
        'y': y
    })

def on_click(x, y, button, pressed):
    is_clicking["status"] = pressed
    state = "Pressed" if pressed else "Released"
    btn_str = "Left" if button == mouse.Button.left else "Right" if button == mouse.Button.right else "NoButton"
    
    mouse_events.append({
        'client timestamp': get_mouse_client_timestamp(),
        'button': btn_str,
        'state': state,
        'x': x,
        'y': y
    })

# =====================================================================
#                      PIPELINE EXECUTION
# =====================================================================

def process_keyboard(user_id, session_id, overlap=0):
    global keyboard_events
    
    if not keyboard_events:
        return
        
    # Extrair cópia e reter o overlap na memória
    chunk = keyboard_events.copy()
    if overlap > 0 and len(chunk) > overlap:
        keyboard_events = chunk[-overlap:]
    else:
        keyboard_events = []
        overlap = 0
        
    df = pd.DataFrame(chunk)
    # Fill required columns for getMetrics
    df['PARTICIPANT_ID'] = user_id
    df['TEST_SECTION_ID'] = session_id
    
    print(f"\n>> Processando lote de {len(df)} eventos de teclado...")
    df_metrics = metricsKeyboard.getMetrics(df)
    
    # Remover o contexto (overlap) calculado para não gravar duplicados no CSV
    if overlap > 0:
        df_metrics = df_metrics.iloc[overlap:]
        
    if not df_metrics.empty:
        metricsKeyboard.saveCSV(df_metrics, user_id)
        print(f">> Lote do Teclado Descarregado (Libertada memória)!")

def process_mouse(user_id, global_session_id, overlap=0):
    global mouse_events
    
    if not mouse_events:
        return
        
    chunk = mouse_events.copy()
    if overlap > 0 and len(chunk) > overlap:
        mouse_events = chunk[-overlap:]
    else:
        mouse_events = []
        overlap = 0
        
    df = pd.DataFrame(chunk)
    
    print(f">> Processando lote de {len(df)} eventos de rato...")
    df_metrics = metricsMouse.getMetrics(df)
    
    if overlap > 0:
        df_metrics = df_metrics.iloc[overlap:]
        
    if not df_metrics.empty:
        # Needs userIdsList for impostor selection
        with open(SESSION_JSON_PATH, 'r') as f:
            data = json.load(f)
        user_ids_list = list(map(int, data.keys()))
        
        metricsMouse.saveCSV(df_metrics, user_id, user_ids_list, global_session_id)
        print(f">> Lote do Rato Descarregado (Libertada memória)!")

def main():
    global start_time, is_running
    print("=== CONTINUOUS AUTHENTICATION PIPELINE ===")
    
    user_id = selectUser()
    session_id, global_session_id = update_session_json(user_id)
    
    print(f"\n>> Sessão Iniciada! User ID: {user_id} | Session: {session_id}")
    print(">> Pressione [ESC] para parar a gravação e desligar o sistema.")
    
    start_time = time.time()
    
    k_listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    m_listener = mouse.Listener(on_click=on_click, on_move=on_move, on_scroll=on_scroll)
    
    k_listener.start()
    m_listener.start()

    # O coração da Autenticação Contínua: Periodic Flushing (A cada 5 segundos)
    while is_running:
        time.sleep(5)
        
        # Descarregar teclado (Threshold de 500 eventos)
        if len(keyboard_events) > 500:
            process_keyboard(user_id, session_id, overlap=25)
            
        # Descarregar rato (Threshold de 1000 eventos)
        if len(mouse_events) > 1000:
            process_mouse(user_id, global_session_id, overlap=1)

    print("\n>> Captura terminada. A processar os eventos finais remanescentes...")
    
    # Flushing final (com overlap=0 porque vamos fechar)
    if len(keyboard_events) > 0:
        process_keyboard(user_id, session_id, overlap=0)
    
    if len(mouse_events) > 0:
        process_mouse(user_id, global_session_id, overlap=0)
        
    # Garantir que as threads fecham
    k_listener.stop()
    m_listener.stop()
    
    print("\n>> PIPELINE CONCLUÍDO E MEMÓRIA LIBERTADA COM SUCESSO! ✅")

if __name__ == "__main__":
    main()
