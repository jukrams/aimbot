import cv2
import numpy as np
import pydirectinput
import win32api
import math
from ultralytics import YOLO

# Deaktiviere pydirectinput Failsafe, falls die Maus an den Rand kommt
pydirectinput.FAILSAFE = False

class Aimbot:
    def __init__(self, model_path="src/yolo26n.pt", width=416, height=416):
        # Erzwinge GPU Nutzung falls verfügbar, deaktiviere unnötigen Output
        self.model = YOLO(model_path)
        
        self.width = width
        self.height = height
        screen_w, screen_h = pydirectinput.size()
        
        self.center_x = width // 2
        self.center_y = height // 2
        
        self.monitor = {
            "top": (screen_h - height) // 2, 
            "left": (screen_w - width) // 2, 
            "width": self.width, 
            "height": self.height
        }
        
        self.conf_threshold = 0.60 # Etwas niedriger ansetzen für schnellere Reaktion
        self.head_offset = 0.18
        self.smooth_factor = 0.4   # 1.0 = instant snap (schlecht), 0.1 = sehr langsam

    def is_right_click_pressed(self):
        return win32api.GetAsyncKeyState(0x02) < 0

    def process_frame(self, sct, aim_enabled, draw_debug=False):
        # BGRA ist direkt als Array verfügbar, Konvertierung nur nötig wenn wir in OpenCV zeichnen
        img_bgra = np.array(sct.grab(self.monitor))
        frame = cv2.cvtColor(img_bgra, cv2.COLOR_BGRA2BGR) # YOLO braucht BGR oder RGB

        results = self.model.predict(frame, classes=[0], verbose=False, device=0) # device=0 forciert GPU
        
        best_target = None
        min_distance = float('inf')

        for r in results:
            boxes = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()

            for i, box in enumerate(boxes):
                if confs[i] < self.conf_threshold:
                    continue

                x1, y1, x2, y2 = box
                
                # Berechne Zielpunkt (Kopf-Bereich)
                target_x = x1 + (x2 - x1) / 2
                target_y = y1 + (y2 - y1) * self.head_offset

                # Distanz zum Zentrum (Fadenkreuz) berechnen
                dist = math.hypot(target_x - self.center_x, target_y - self.center_y)

                # Nächstes Ziel auswählen, NICHT das mit der höchsten Confidence
                if dist < min_distance:
                    min_distance = dist
                    best_target = (target_x, target_y)

                if draw_debug:
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    cv2.circle(frame, (int(target_x), int(target_y)), 3, (0, 0, 255), -1)

        if aim_enabled and self.is_right_click_pressed() and best_target is not None:
            self._move_mouse_to_target(best_target[0], best_target[1])

        return frame if draw_debug else None

    def _move_mouse_to_target(self, target_x, target_y):
        offset_x = target_x - self.center_x
        offset_y = target_y - self.center_y

        # Smoothing anwenden
        move_x = int(offset_x * self.smooth_factor)
        move_y = int(offset_y * self.smooth_factor)

        # Nur bewegen, wenn auch eine spürbare Bewegung vorliegt (verhindert Micro-Jitter)
        if abs(move_x) > 0 or abs(move_y) > 0:
            pydirectinput.moveRel(move_x, move_y, relative=True)
