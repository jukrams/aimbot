import cv2
import time
import sys
from mss import mss
from pynput import keyboard
from src.aimbot import Aimbot

class AppController:
    def __init__(self):
        self.aimbot = Aimbot(width=416, height=416)
        self.running = True
        self.aim_active = False
        
        # Terminal Farben
        self.C_BLUE = "\033[94m"
        self.C_RED = "\033[91m"
        self.C_GREEN = "\033[92m"
        self.C_RESET = "\033[0m"

    def update_terminal(self):
        """Aktualisiert die Statuszeile im Terminal."""
        status_label = f"{self.C_BLUE}STATUS:{self.C_RESET}"
        status_val = f"{self.C_GREEN} [Started]{self.C_RESET}" if self.aim_active else f"{self.C_RED} [Stopped]{self.C_RESET}"
        sys.stdout.write(f"\r{status_label}{status_val} | F1: Toggle | F2: Exit | Right-Click: Aim    ")
        sys.stdout.flush()

    def on_press(self, key):
        if key == keyboard.Key.f1:
            self.aim_active = not self.aim_active
            self.update_terminal()
        if key == keyboard.Key.f2:
            self.running = False
            return False

    def run(self):
        # Keyboard Listener im Hintergrund
        listener = keyboard.Listener(on_press=self.on_press)
        listener.start()

        print(f"YOLO26 Aim-Assist gestartet. Modus: 416x416 Center-Crop.")
        self.update_terminal()

        with mss() as sct:
            prev_time = time.time()
            
            while self.running:
                # Frame verarbeiten via Aimbot Klasse
                frame = self.aimbot.process_frame(sct, self.aim_active)

                # FPS Berechnung
                curr_time = time.time()
                fps = 1 / (curr_time - prev_time)
                prev_time = curr_time

                # Overlay im Fenster
                cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                cv2.imshow("YOLO26 Debug View", frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        cv2.destroyAllWindows()
        print(f"\n{self.C_RED}Programm beendet.{self.C_RESET}")

if __name__ == "__main__":
    app = AppController()
    app.run()
