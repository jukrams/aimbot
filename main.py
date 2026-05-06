import cv2
import time
import sys
from pynput import keyboard
from src.aimbot import Aimbot

class AppController:
    def __init__(self, debug_mode=False):
        # Die Aimbot-Klasse verwaltet Kamera und Modell jetzt intern
        self.aimbot = Aimbot(width=416, height=416)
        self.running = True
        self.aim_active = False
        self.debug_mode = debug_mode
        
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

        print("System aktiv. Nutze DXCam + YOLO Vektorisierung.")
        self.update_terminal()

        prev_time = time.time()
        
        try:
            while self.running:
                # WICHTIG: Kein 'sct' mehr übergeben! 
                # Die Methode liefert nur im debug_mode ein Bild zurück.
                frame = self.aimbot.process_frame(self.aim_active, draw_debug=self.debug_mode)

                if self.debug_mode and frame is not None:
                    curr_time = time.time()
                    fps = 1 / (curr_time - prev_time) if curr_time - prev_time > 0 else 0
                    prev_time = curr_time

                    cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.imshow("Aimbot Debug", frame)
                    
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
        finally:
            self.aimbot.cleanup() # Kamera-Ressourcen sauber freigeben
            cv2.destroyAllWindows()
            print(f"\n{self.C_RED}Programm beendet.{self.C_RESET}")

if __name__ == "__main__":
    # Starte mit True, um zu sehen, ob es funktioniert. 
    # Später auf False für maximale Geschwindigkeit.
    app = AppController(debug_mode=True)
    app.run()
