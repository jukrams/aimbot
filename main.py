import sys
import os
from pynput import keyboard
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QTimer
from src.config import ConfigManager
from src.aimbot import Aimbot
from src.gui import OverlayGUI

class AppController:
    def __init__(self):
        self.config = ConfigManager()
        self.aimbot = Aimbot(self.config)
        
        self.app = QApplication(sys.argv)
        self.gui = OverlayGUI(self.aimbot, self.config)
        
        self.gui.toggle_requested.connect(self.set_aim_status)
        self.gui.exit_requested.connect(self.exit_program)
        self.gui.settings_changed.connect(self.reload_system)
        
        self.aim_active = False
        self.COLORS = {"BLUE": "\033[94m", "RED": "\033[91m", "GREEN": "\033[92m", "RESET": "\033[0m"}

        # Timer für die regelmäßige Abfrage der YOLO-Ergebnisse initialisieren
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.poll_aimbot_data)

    def update_terminal(self):
        state = f"{self.COLORS['GREEN']}[ON]{self.COLORS['RESET']}" if self.aim_active else f"{self.COLORS['RED']}[OFF]{self.COLORS['RESET']}"
        sys.stdout.write(f"\r{self.COLORS['BLUE']}STATUS:{self.COLORS['RESET']} {state} | F1: Toggle | F2: Exit    ")
        sys.stdout.flush()

    def set_aim_status(self, state: bool):
        self.aim_active = state
        self.gui.set_aim_state(self.aim_active)
        # Den Aimbot-Thread über den neuen Zustand informieren
        self.aimbot.aim_enabled = self.aim_active
        self.update_terminal()

    def reload_system(self):
        """Wird getriggert, wenn Einstellungen im GUI gespeichert wurden"""
        print(f"\n{self.COLORS['BLUE']}Lade Einstellungen neu...{self.COLORS['RESET']}")
        # Threads kurz stoppen, um Race Conditions bei Einstellungsänderungen zu vermeiden
        self.aimbot.stop()
        self.aimbot.apply_settings()
        self.aimbot.start(aim_enabled=self.aim_active)
        self.update_terminal()

    def poll_aimbot_data(self):
        """Fragt die Queue des Aimbots ab und sendet Daten an die GUI."""
        data = self.aimbot.get_latest_data()
        if data is not None:
            # Falls deine OverlayGUI eine Methode zum Zeichnen der Boxen besitzt,
            # wird sie hier aufgerufen. Beispiel:
            if hasattr(self.gui, "update_overlay"):
                self.gui.update_overlay(data)
            # Hinweis: Wenn die GUI-Aktualisierung direkt in der GUI-Klasse über einen eigenen
            # Timer gelöst ist, kann diese Methode hier komplett leer bleiben (pass).

    def exit_program(self):
        self.update_timer.stop()
        self.aimbot.stop()
        self.app.quit()

    def on_press(self, key):
        try:
            if key == keyboard.Key.f1:
                self.set_aim_status(not self.aim_active)
            elif key == keyboard.Key.f2:
                self.exit_program()
                return False
        except AttributeError:
            pass

    def run(self):
        listener = keyboard.Listener(on_press=self.on_press)
        listener.daemon = True
        listener.start()

        # Startet die Hintergrundthreads des Aimbots (Kamera-Erfassung & YOLO)
        self.aimbot.start(aim_enabled=self.aim_active)

        # Timer starten: Intervall 1ms sorgt für sofortige Verarbeitung,
        # sobald Daten in der Queue landen.
        self.update_timer.start(1)

        print("System aktiv. Öffne Interface...")
        self.update_terminal()
        self.gui.show()
        
        exit_code = 0
        try:
            exit_code = self.app.exec()
        finally:
            self.aimbot.cleanup()
            print(f"\n{self.COLORS['RED']}System offline.{self.COLORS['RESET']}")
            sys.exit(exit_code)

if __name__ == "__main__":
    os.environ["QT_ENABLE_HIGHDPI_SCALING"] = "1"
    
    app = AppController()
    app.run()
