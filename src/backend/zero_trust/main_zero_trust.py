import time
import sys
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.abspath(os.path.join(BASE_DIR, '..'))
ROOT_DIR = os.path.abspath(os.path.join(BACKEND_DIR, '..', '..'))

sys.path.insert(0, BASE_DIR)

from sensor_agent import SensorAgent
from processing_engine import ProcessingEngine
from decision_engine import DecisionEngine

# ==========================================
# GESTÃO DE UTILIZADOR
# ==========================================
# (Simplificado para o Orchestrator - Em produção ligaria com o auth do Windows)

def select_user():
    # Simulando a mesma seleção do continuous_pipeline.py
    print(">> Identifique o seu ID de utilizador legítimo (Ex: 1)")
    user_id = int(input(">> ID: "))
    return user_id, 999 # session_id dummy para simplificar

def main():
    print("=====================================================")
    print("    ZERO-TRUST CONTINUOUS AUTHENTICATION ENGINE      ")
    print("=====================================================")
    
    user_id, session_id = select_user()
    
    dataset_dir = os.path.join(ROOT_DIR, 'dataset')
    
    print("\n[INIT] A carregar Processing Engine e Scalers...")
    processor = ProcessingEngine(dataset_dir)
    
    ckpt_k = os.path.join(ROOT_DIR, 'checkpoints_teclado', 'modelo_epoch_95.pth')
    ckpt_m = os.path.join(ROOT_DIR, 'checkpoints_rato', 'modelo_epoch_25.pth')
    
    print("\n[INIT] A carregar Decision Engine e Modelos Globais...")
    decision = DecisionEngine(ckpt_k, ckpt_m)
    
    sensor = SensorAgent()
    sensor.start()
    
    # Ficheiros onde gravar as novas sessões rotativas
    out_k = os.path.join(dataset_dir, f'session_live_keyboard_user{user_id}.csv')
    out_m = os.path.join(dataset_dir, f'session_live_mouse_user{user_id}.csv')
    
    # ==========================================
    # FASE 1: CALIBRAÇÃO (1 Minuto)
    # ==========================================
    print("\n=====================================================")
    print(" FASE 1: CALIBRAÇÃO (Fine-Tuning do Modelo)          ")
    print(" Continue a usar o PC normalmente durante 1 minuto.  ")
    print("=====================================================")
    
    time.sleep(60) # 1 Minuto de Calibração
    
    chunk_k, overlap_k = sensor.flush_keyboard(overlap=25)
    chunk_m, overlap_m = sensor.flush_mouse(overlap=1)
    
    tensor_k, _ = processor.process_keyboard(chunk_k, 0, user_id, session_id, out_k)
    tensor_m, _ = processor.process_mouse(chunk_m, 0, user_id, session_id, out_m)
    
    if tensor_k is not None:
        decision.fine_tune('keyboard', tensor_k, epochs=10)
    else:
        print(">> Sem dados de teclado suficientes para calibração.")
        
    if tensor_m is not None:
        decision.fine_tune('mouse', tensor_m, epochs=10)
    else:
        print(">> Sem dados de rato suficientes para calibração.")

    # ==========================================
    # FASE 2: AUTENTICAÇÃO CONTÍNUA
    # ==========================================
    print("\n=====================================================")
    print(" FASE 2: AUTENTICAÇÃO CONTÍNUA EM TEMPO REAL         ")
    print(" O sistema está agora a monitorizar o seu padrão!    ")
    print(" Pressione [ESC] para abortar.                       ")
    print("=====================================================")
    
    loop_time = 10 # Segundos entre cada inferência
    
    while sensor.is_running:
        time.sleep(loop_time)
        
        # O [ESC] pode ter sido pressionado durante o sleep
        if not sensor.is_running:
            break
            
        chunk_k, overlap_k = sensor.flush_keyboard(overlap=25)
        chunk_m, overlap_m = sensor.flush_mouse(overlap=1)
        
        tensor_k = None
        tensor_m = None
        
        if len(chunk_k) > 25:
            tensor_k, _ = processor.process_keyboard(chunk_k, overlap_k, user_id, session_id, out_k)
            
        if len(chunk_m) > 1:
            tensor_m, _ = processor.process_mouse(chunk_m, overlap_m, user_id, session_id, out_m)
            
        if tensor_k is not None or tensor_m is not None:
            decision.evaluate(tensor_k, tensor_m)

    print("\n>> Motor Zero-Trust Encerrado.")

if __name__ == "__main__":
    main()
