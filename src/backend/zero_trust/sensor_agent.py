import time
from pynput import keyboard, mouse
from pynput.keyboard import Key

class SensorAgent:
    def __init__(self):
        self.keyboard_events = []
        self.mouse_events = []
        
        self.pressed_keys = {}
        self.is_clicking = {"status": False}
        self.start_time = None
        self.is_running = False
        
        special_keys = list(Key)
        self.key_to_id = {str(k): 1000 + i for i, k in enumerate(special_keys)}
        
        self.k_listener = None
        self.m_listener = None

    def get_keycode(self, key):
        try:
            if hasattr(key, 'char') and key.char is not None:
                if len(key.char) == 1:
                    return ord(key.char)
            return self.key_to_id.get(str(key), 9999)
        except Exception:
            return 0

    def _on_press(self, key):
        if key == keyboard.Key.esc:
            print("\n>> Stopping Capture (ESC pressed)...")
            self.is_running = False
            return False
        
        k_str = str(key)
        if k_str not in self.pressed_keys:
            self.pressed_keys[k_str] = time.time()

    def _on_release(self, key):
        k_str = str(key)
        if k_str in self.pressed_keys:
            press_time = self.pressed_keys.pop(k_str)
            release_time = time.time()
            
            self.keyboard_events.append({
                'PRESS_TIME': int(press_time * 1000),
                'RELEASE_TIME': int(release_time * 1000),
                'KEYCODE': self.get_keycode(key)
            })

    def _get_mouse_client_timestamp(self):
        return time.time() - self.start_time

    def _on_move(self, x, y):
        state = "Drag" if self.is_clicking["status"] else "Move"
        self.mouse_events.append({
            'client timestamp': self._get_mouse_client_timestamp(),
            'button': "NoButton",
            'state': state,
            'x': x,
            'y': y
        })

    def _on_scroll(self, x, y, dx, dy):
        self.mouse_events.append({
            'client timestamp': self._get_mouse_client_timestamp(),
            'button': "Scroll",
            'state': "Up" if dy > 0 else "Down",
            'x': x,
            'y': y
        })

    def _on_click(self, x, y, button, pressed):
        self.is_clicking["status"] = pressed
        state = "Pressed" if pressed else "Released"
        btn_str = "Left" if button == mouse.Button.left else "Right" if button == mouse.Button.right else "NoButton"
        
        self.mouse_events.append({
            'client timestamp': self._get_mouse_client_timestamp(),
            'button': btn_str,
            'state': state,
            'x': x,
            'y': y
        })

    def start(self):
        self.start_time = time.time()
        self.is_running = True
        
        self.k_listener = keyboard.Listener(on_press=self._on_press, on_release=self._on_release)
        self.m_listener = mouse.Listener(on_click=self._on_click, on_move=self._on_move, on_scroll=self._on_scroll)
        
        self.k_listener.start()
        self.m_listener.start()
        print(">> Sensor Agent Started.")

    def stop(self):
        self.is_running = False
        if self.k_listener: self.k_listener.stop()
        if self.m_listener: self.m_listener.stop()
        print(">> Sensor Agent Stopped.")

    def flush_keyboard(self, overlap=25):
        chunk = self.keyboard_events.copy()
        if overlap > 0 and len(chunk) > overlap:
            self.keyboard_events = chunk[-overlap:]
        else:
            self.keyboard_events = []
            overlap = 0
        return chunk, overlap

    def flush_mouse(self, overlap=1):
        chunk = self.mouse_events.copy()
        if overlap > 0 and len(chunk) > overlap:
            self.mouse_events = chunk[-overlap:]
        else:
            self.mouse_events = []
            overlap = 0
        return chunk, overlap
