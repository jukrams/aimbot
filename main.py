import cv2
import time
import sys
from mss import mss
from pynput import keyboard
from src.aimbot import Aimbot

class AppController:
    def __init__(self, debug_mode=False):
        self.aimbot = Aimbot(width=416, height=416)
        self.running = True
        self.aim_active = False
        self.debug_mode = debug_mode # Debug-Ansicht deaktivierbar für max FPS
        
        self.C_BLUE = "\033[94m"
        self.C_RED = "\033[91m"
        self.C_GREEN = "\033[92m"
        self.C_RESET = "\033[0m"

    def update_terminal(self):
        status = f"{self.C_GREEN}[ON]{self.C_RESET}" if self.aim_active else f"{self.C_RED}[OFF]{self.C_RESET}"
        sys.stdout.write(f"\r{self.C_BLUE}STATUS:{self.C_RESET} {status} | F1: Toggle | F2: Exit | Right-Click: Aim    ")
        sys.stdout.flush()

    def on_press(self, key):
        if key == keyboard.Key.f1:
            self.aim_active = not self.aim_active
            self.update_terminal()
        if key == keyboard.Key.f2:
            self.running = False
            return False

    def run(self):
        listener = keyboard.Listener(on_press=self.on_press)
        listener.start()

        print("YOLO Aimbot gestartet. Optimierte Version.")
        self.update_terminal()

        with mss() as sct:
            prev_time = time.time()
            
            while self.running:
                frame = self.aimbot.process_frame(sct, self.aim_active, draw_debug=self.debug_mode)

                if self.debug_mode and frame is not None:
                    curr_time = time.time()
                    fps = 1 / (curr_time - prev_time) if curr_time - prev_time > 0 else 0
                    prev_time = curr_time

                    cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.imshow("Debug View", frame)
                    
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

        if self.debug_mode:
            cv2.destroyAllWindows()
        print(f"\n{self.C_RED}Beendet.{self.C_RESET}")

if __name__ == "__main__":
    # Setze debug_mode auf True, wenn du die Boxen sehen willst. Kostet Performance!
    app = AppController(debug_mode=False)
    app.run()
