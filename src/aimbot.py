import cv2
import numpy as np
import pydirectinput
import win32api
from ultralytics import YOLO
from mss import mss

class Aimbot:
    def __init__(self, model_path="src/yolo26n.pt", width=416, height=416):
        # Initialisierung des Modells
        self.model = YOLO(model_path)
        
        # Bildschirm-Konfiguration
        self.width = width
        self.height = height
        screen_w, screen_h = pydirectinput.size()
        
        self.left = (screen_w - width) // 2
        self.top = (screen_h - height) // 2
        self.center_x = width // 2
        self.center_y = height // 2
        
        self.monitor = {
            "top": self.top, 
            "left": self.left, 
            "width": self.width, 
            "height": self.height
        }
        
        # Einstellungen
        self.conf_threshold = 0.80
        self.head_offset = 0.18  # Zielt auf das obere Fünftel der Box

    def is_right_click_pressed(self):
        """Prüft den Status der rechten Maustaste via Win32 API."""
        return win32api.GetAsyncKeyState(0x02) < 0

    def process_frame(self, sct, aim_enabled):
        """Erfasst ein Bild, führt Inferenz aus und bewegt ggf. die Maus."""
        # Screenshot
        img = np.array(sct.grab(self.monitor))
        frame = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        # Inferenz (Klasse 0 = Person)
        results = self.model.predict(frame, classes=[0], verbose=False)
        
        best_target = None
        current_max_conf = self.conf_threshold

        for r in results:
            boxes = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()

            for i, box in enumerate(boxes):
                conf_val = confs[i]
                x1, y1, x2, y2 = box

                # Zeichnen der Bounding Box
                is_valid = conf_val >= self.conf_threshold
                color = (0, 255, 0) if is_valid else (0, 0, 255)
                
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                cv2.putText(frame, f"{int(conf_val * 100)}%", (int(x1), int(y1) - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                # Auswahl des besten Ziels
                if conf_val > current_max_conf:
                    current_max_conf = conf_val
                    best_target = box

        # Mausbewegung ausführen
        if aim_enabled and self.is_right_click_pressed() and best_target is not None:
            self._move_mouse_to_target(best_target)

        return frame

    def _move_mouse_to_target(self, box):
        """Berechnet das Delta und sendet relativen Input."""
        x1, y1, x2, y2 = box
        target_x = x1 + (x2 - x1) / 2
        target_y = y1 + (y2 - y1) * self.head_offset

        offset_x = int(target_x - self.center_x)
        offset_y = int(target_y - self.center_y)

        pydirectinput.moveRel(offset_x, offset_y, relative=True)
