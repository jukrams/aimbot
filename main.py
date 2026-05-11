import cv2
import time
import sys
from pynput import keyboard
from src.aimbot import Aimbot

class AppController:
    def __init__(self, debug_mode: bool = True):
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
        try:
            if key == keyboard.Key.f1:
                self.aim_active = not self.aim_active
                self.update_terminal()
            elif key == keyboard.Key.f2:
                self.running = False
                return False
        except AttributeError as e:
            # Spezifische Fehler fangen, nicht blind alles ignorieren
            pass

    def run(self):
        listener = keyboard.Listener(on_press=self.on_press)
        listener.daemon = True
        listener.start()

        if self.debug_mode:
            cv2.namedWindow("Debug", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Debug", 416, 416)

        print("System aktiv. Starte Prozess-Loop...")
        self.update_terminal()
        time.sleep(1)

        try:
            while self.running:
                start_time = time.perf_counter()
                
                frame = self.aimbot.process_frame(self.aim_active, draw_debug=self.debug_mode)

                if self.debug_mode and frame is not None:
                    fps = int(1.0 / (time.perf_counter() - start_time + 0.0001))
                    cv2.putText(frame, f"FPS: {fps}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                    cv2.circle(frame, (self.aimbot.center_x, self.aimbot.center_y), int(self.aimbot.fov_radius), (255, 0, 0), 2)
                    cv2.imshow("Debug", frame)
                
                # waitKey zwingend erforderlich, sonst blockiert die OpenCV GUI
                if self.debug_mode and cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        except KeyboardInterrupt:
            pass
        finally:
            self.aimbot.cleanup()
            cv2.destroyAllWindows()
            print(f"\n{self.COLORS['RED']}System offline.{self.COLORS['RESET']}")

if __name__ == "__main__":
    app = AppController(debug_mode=True)
    app.run()
