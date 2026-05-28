import sys
from pynput import keyboard
from PyQt6.QtWidgets import QApplication
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

    def update_terminal(self):
        state = f"{self.COLORS['GREEN']}[ON]{self.COLORS['RESET']}" if self.aim_active else f"{self.COLORS['RED']}[OFF]{self.COLORS['RESET']}"
        sys.stdout.write(f"\r{self.COLORS['BLUE']}STATUS:{self.COLORS['RESET']} {state} | F1: Toggle | F2: Exit    ")
        sys.stdout.flush()

    def set_aim_status(self, state: bool):
        self.aim_active = state
        self.gui.set_aim_state(self.aim_active)
        self.update_terminal()

    def reload_system(self):
        """Wird getriggert, wenn Einstellungen im GUI gespeichert wurden"""
        print(f"\n{self.COLORS['BLUE']}Lade Einstellungen neu...{self.COLORS['RESET']}")
        self.aimbot.apply_settings()
        self.update_terminal()

    def exit_program(self):
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
    import os
    os.environ["QT_ENABLE_HIGHDPI_SCALING"] = "1"
    
    app = AppController()
    app.run()
