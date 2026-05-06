import cv2
import numpy as np
import pydirectinput
import win32api
import math
import dxcam
from ultralytics import YOLO

# Deaktiviere Failsafe für unterbrechungsfreie Mausbewegungen
pydirectinput.FAILSAFE = False

class Aimbot:
    def __init__(self, model_path="src/yolo11s.engine", width=416, height=416):
        print("Initialisiere High-End Aimbot...")
        # Lade Modell (idealerweise .engine für TensorRT)
        self.model = YOLO(model_path)
        
        self.width = width
        self.height = height
        
        # DXCam Setup (GPU Frame Buffer Access)
        self.camera = dxcam.create(output_color="BGR")
        
        screen_w, screen_h = pydirectinput.size()
        left = (screen_w - width) // 2
        top = (screen_h - height) // 2
        self.region = (left, top, (left + width), (top + height))
        
        self.center_x = width // 2
        self.center_y = height // 2
        
        # --- BASIS EINSTELLUNGEN ---
        self.conf_threshold = 0.60
        self.fov_radius = 60
        self.head_offset = 0.18
        self.base_smooth = 0.35       # Basis-Glättung
        
        # --- GEWICHTUNG (3-WEGE-TAUZIEHEN) ---
        self.weight_dist = 0.60
        self.weight_conf = 0.25
        self.weight_size = 0.15
        
        # --- UPGRADE: PRÄDIKTION (VORSICHTIG) ---
        self.prev_target = None
        self.velocity = [0, 0]
        # 0.4 ist ein konservativer Wert, der ca. 10-15ms Latenz kompensiert
        self.prediction_strength = 0.4 
        
        # --- STICKY AIM STATE ---
        self.last_target_point = None

    def is_right_click_pressed(self):
        return win32api.GetAsyncKeyState(0x02) < 0

    def process_frame(self, aim_enabled, draw_debug=False):
        frame = self.camera.grab(region=self.region)
        if frame is None:
            return None

        results = self.model.predict(frame, classes=[0], verbose=False, half=True)
        
        best_target = None
        highest_score = -float('inf')
        target_conf = 0

        for r in results:
            boxes = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()

            for i, box in enumerate(boxes):
                conf = confs[i]
                if conf < self.conf_threshold: continue

                x1, y1, x2, y2 = box
                w, h = x2 - x1, y2 - y1
                
                # Aspekt-Ratio Filter (Leichenschutz)
                if w > h * 1.1: continue

                t_x = x1 + w / 2
                t_y = y1 + h * self.head_offset
                
                dist = math.hypot(t_x - self.center_x, t_y - self.center_y)
                if dist > self.fov_radius: continue

                # Normalisierungen
                norm_dist = max(0.0, 1.0 - (dist / self.fov_radius))
                norm_conf = conf
                norm_size = min((w * h) / 80000.0, 1.0)
                
                # Score Berechnung
                base_score = ((norm_dist ** 2) * self.weight_dist) + \
                             (norm_conf * self.weight_conf) + \
                             (norm_size * self.weight_size)

                # Hysterese (Sticky Bonus)
                if self.last_target_point is not None:
                    d_last = math.hypot(t_x - self.last_target_point[0], t_y - self.last_target_point[1])
                    if d_last < 25: base_score += 0.15

                if base_score > highest_score:
                    highest_score = base_score
                    best_target = (t_x, t_y)
                    target_conf = conf

        # --- UPGRADE: KALMAN-PRÄDIKTION ---
        if best_target:
            final_x, final_y = best_target
            
            if self.prev_target:
                # Berechne Bewegungsvektor des Gegners
                vx = final_x - self.prev_target[0]
                vy = final_y - self.prev_target[1]
                
                # Prädiziere Position (Vorsichtige Extrapolation)
                final_x += vx * self.prediction_strength
                final_y += vy * self.prediction_strength
            
            self.prev_target = best_target # Speichere reale Position für nächsten Frame
            self.last_target_point = best_target

            if aim_enabled and self.is_right_click_pressed():
                self._move_mouse_to_target(final_x, final_y, target_conf)
        else:
            self.prev_target = None
            self.last_target_point = None

        return frame if draw_debug else None

    def _move_mouse_to_target(self, target_x, target_y, confidence):
        """Bewegt die Maus mit dynamischem Smoothing basierend auf KI-Sicherheit."""
        offset_x = target_x - self.center_x
        offset_y = target_y - self.center_y

        # UPGRADE: DYNAMISCHES SMOOTHING (Weight-Drop)
        # Wenn Confidence hoch (0.9), ist der Faktor aggressiver.
        # Wenn Confidence niedrig (0.6), wird die Bewegung extrem weich/langsam.
        dynamic_smooth = self.base_smooth * (confidence ** 2)

        move_x = int(offset_x * dynamic_smooth)
        move_y = int(offset_y * dynamic_smooth)

        if abs(move_x) > 0 or abs(move_y) > 0:
            pydirectinput.moveRel(move_x, move_y, relative=True)

    def cleanup(self):
        if hasattr(self, 'camera'):
            self.camera.stop()
