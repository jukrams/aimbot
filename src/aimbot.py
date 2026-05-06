import cv2
import numpy as np
import pydirectinput
import win32api
import math
import dxcam
from ultralytics import YOLO

pydirectinput.FAILSAFE = False

class Aimbot:
    def __init__(self, model_path="src/yolo26n.pt", width=416, height=416):
        # Falls du eine .engine Datei hast, Pfad hier anpassen für 3x FPS
        self.model = YOLO(model_path)
        
        self.width = width
        self.height = height
        
        # Kamera-Setup via DXCam (schnellste Methode unter Windows)
        self.camera = dxcam.create(output_color="BGR")
        # Wir erfassen nur den Center-Crop für maximale Performance
        screen_w, screen_h = pydirectinput.size()
        left = (screen_w - width) // 2
        top = (screen_h - height) // 2
        right = left + width
        bottom = top + height
        self.region = (left, top, right, bottom)
        
        self.center_x = width // 2
        self.center_y = height // 2
        
        # Einstellungen
        self.conf_threshold = 0.60
        self.head_offset = 0.18
        self.fov_radius = 150
        self.smooth_factor = 0.4

    def is_right_click_pressed(self):
        return win32api.GetAsyncKeyState(0x02) < 0

    def process_frame(self, aim_enabled, draw_debug=False):
        """
        WICHTIG: Diese Signatur muss exakt so aussehen!
        self, aim_enabled, draw_debug=False
        """
        # Frame capture
        frame = self.camera.grab(region=self.region)
        if frame is None:
            return None

        # Inferenz
        results = self.model.predict(frame, classes=[0], verbose=False)
        
        best_target = None
        highest_score = -float('inf')

        # Vektorisierte Logik (vereinfacht für Stabilität)
        for r in results:
            boxes = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()

            for i, box in enumerate(boxes):
                conf = confs[i]
                if conf < self.conf_threshold: continue

                x1, y1, x2, y2 = box
                w, h = x2 - x1, y2 - y1
                
                # Aspekt-Ratio Filter
                if w > h * 1.2: continue

                t_x, t_y = x1 + w / 2, y1 + h * self.head_offset
                dist = math.hypot(t_x - self.center_x, t_y - self.center_y)

                if dist > self.fov_radius: continue

                # Score-Berechnung
                score = (conf * 100) - (dist * 0.5)

                if score > highest_score:
                    highest_score = score
                    best_target = (t_x, t_y)

                if draw_debug:
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

        if draw_debug:
            cv2.circle(frame, (self.center_x, self.center_y), self.fov_radius, (255, 0, 0), 1)

        # Bewegung ausführen
        if aim_enabled and self.is_right_click_pressed() and best_target:
            self._move_mouse_to_target(best_target[0], best_target[1])

        return frame if draw_debug else None

    def _move_mouse_to_target(self, target_x, target_y):
        offset_x = int((target_x - self.center_x) * self.smooth_factor)
        offset_y = int((target_y - self.center_y) * self.smooth_factor)
        if abs(offset_x) > 0 or abs(offset_y) > 0:
            pydirectinput.moveRel(offset_x, offset_y, relative=True)

    def cleanup(self):
        # Wichtig, um die Kamera-Ressource freizugeben
        if hasattr(self, 'camera'):
            del self.camera
