import cv2
import time
import sys
from pynput import keyboard
from src.aimbot import Aimbot

class AppController:
    def __init__(self, debug_mode: bool = False):
        self.aimbot = Aimbot()
        self.running = True
        self.aim_active = False
        self.debug_mode = debug_mode
        
        self.COLORS = {"BLUE": "\033[94m", "RED": "\033[91m", "GREEN": "\033[92m", "RESET": "\033[0m"}

    def update_terminal(self):
        state = f"{self.COLORS['GREEN']}[ON]{self.COLORS['RESET']}" if self.aim_active else f"{self.COLORS['RED']}[OFF]{self.COLORS['RESET']}"
        sys.stdout.write(f"\r{self.COLORS['BLUE']}STATUS:{self.COLORS['RESET']} {state} | F1: Toggle | F2: Exit | Right-Click: Aim    ")
        sys.stdout.flush()

    def on_press(self, key):
        if key == keyboard.Key.f1:
            self.aim_active = not self.aim_active
            self.update_terminal()
        elif key == keyboard.Key.f2:
            self.running = False
            return False

    def run(self):
        listener = keyboard.Listener(on_press=self.on_press)
        listener.start()

        print("System aktiv. TensorRT + Vectorized Logic.")
        self.update_terminal()

        prev_time = time.perf_counter() # Präzisere Zeitmessung
        
        try:
            while self.running:
                frame = self.aimbot.process_frame(self.aim_active, draw_debug=self.debug_mode)

                if self.debug_mode and frame is not None:
                    curr_time = time.perf_counter()
                    fps = int(1 / (curr_time - prev_time)) if curr_time > prev_time else 0
                    prev_time = curr_time

                    cv2.putText(frame, f"FPS: {fps}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.imshow("Debug", frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
        except KeyboardInterrupt:
            pass
        finally:
            self.aimbot.cleanup()
            cv2.destroyAllWindows()
            print(f"\n{self.COLORS['RED']}System offline.{self.COLORS['RESET']}")

if __name__ == "__main__":
    app = AppController(debug_mode=False) # In Produktion ZWINGEND False für Max FPS
    app.run()
