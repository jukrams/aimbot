import cv2
import numpy as np
import pydirectinput
import win32api
import math
import dxcam
from ultralytics import YOLO

# Deaktiviere Failsafe, falls die Maus an den Bildschirmrand stößt
pydirectinput.FAILSAFE = False

class Aimbot:
    def __init__(self, model_path="src/yolo26n.pt", width=416, height=416):
        print("Initialisiere KI-Modell...")
        self.model = YOLO(model_path)
        
        self.width = width
        self.height = height
        
        # DXCam Setup für direkte Speicher-Erfassung (schnellstes Capture)
        print("Initialisiere DXCam...")
        self.camera = dxcam.create(output_color="BGR")
        
        screen_w, screen_h = pydirectinput.size()
        left = (screen_w - width) // 2
        top = (screen_h - height) // 2
        right = left + width
        bottom = top + height
        self.region = (left, top, right, bottom)
        
        # Geometrisches Zentrum
        self.center_x = width // 2
        self.center_y = height // 2
        
        # --- HAUPTEINSTELLUNGEN ---
        self.conf_threshold = 0.60    # 60% reicht, da das Tauziehen False Positives bestraft
        self.fov_radius = 200          # Legit-Radius (Kreis)
        self.head_offset = 0.18       # Zielt auf das obere Fünftel (Kopf/Brust)
        self.smooth_factor = 0.4      # Smoothing für menschliche Bewegungen
        
        # --- GEWICHTUNG FÜR DAS TAUZIEHEN (Muss in Summe 1.0 ergeben) ---
        self.weight_dist = 0.60       # Fadenkreuz-Magnet (Wichtigster Wert)
        self.weight_conf = 0.25       # KI-Sicherheit
        self.weight_size = 0.15       # Threat Prioritization (Nahkampf)
        
        # --- STICKY AIM STATE ---
        self.last_target_point = None

    def is_right_click_pressed(self):
        # Asynchrone Abfrage der rechten Maustaste
        return win32api.GetAsyncKeyState(0x02) < 0

    def process_frame(self, aim_enabled, draw_debug=False):
        """Haupt-Pipeline: Wird bis zu 144 Mal pro Sekunde aufgerufen."""
        frame = self.camera.grab(region=self.region)
        if frame is None:
            return None

        # KI-Inferenz (nur Klasse 0 = Personen)
        results = self.model.predict(frame, classes=[0], verbose=False)
        
        best_target = None
        highest_score = -float('inf')

        for r in results:
            boxes = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()

            for i, box in enumerate(boxes):
                conf = confs[i]
                
                # --- FILTER 1: Confidence ---
                if conf < self.conf_threshold: 
                    continue

                x1, y1, x2, y2 = box
                w, h = x2 - x1, y2 - y1
                
                # --- FILTER 2: Aspekt-Ratio (Leichen-Schutz) ---
                if w > h * 1.2: 
                    continue

                # Ziel-Koordinate berechnen
                t_x = x1 + w / 2
                t_y = y1 + h * self.head_offset
                
                dist = math.hypot(t_x - self.center_x, t_y - self.center_y)

                # --- FILTER 3: Hard-FOV (Kreis-Geometrie) ---
                if dist > self.fov_radius: 
                    continue

                # === MATHEMATIK: DAS 3-WEGE-TAUZIEHEN ===
                
                # A) Normalisierte Distanz (0 bis 1)
                norm_dist = max(0.0, 1.0 - (dist / self.fov_radius))
                
                # B) Normalisierte KI-Sicherheit
                norm_conf = conf
                
                # C) Normalisierte Objektgröße (Gedeckelt bei 100.000 Pixeln)
                area = w * h
                norm_size = min(area / 100000.0, 1.0)
                
                # Basis-Score berechnen (Die Distanz wird quadriert für den Exponentiellen Magneten)
                base_score = ((norm_dist ** 2) * self.weight_dist) + \
                             (norm_conf * self.weight_conf) + \
                             (norm_size * self.weight_size)

                # --- STICKY AIM (Hysterese) ---
                # Wenn das Ziel nah an dem ist, was wir im Frame davor anvisiert haben,
                # vergeben wir einen Treue-Bonus von 15%.
                sticky_bonus = 0.0
                if self.last_target_point is not None:
                    dist_to_last = math.hypot(t_x - self.last_target_point[0], t_y - self.last_target_point[1])
                    if dist_to_last < 20:  # Toleranz für Bewegung zwischen Frames
                        sticky_bonus = 0.15

                final_score = base_score + sticky_bonus

                # Champion-Check
                if final_score > highest_score:
                    highest_score = final_score
                    best_target = (t_x, t_y)

                # Debug-Ansicht: Zeichne alle gültigen Ziele im FOV
                if draw_debug:
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    # Zeige den berechneten final_score auf 2 Nachkommastellen genau
                    score_text = f"S:{final_score:.2f}"
                    cv2.putText(frame, score_text, (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,255), 1)

        # --- ABSCHLUSS ---
        
        # Speicher das beste Ziel für den nächsten Frame (für Sticky Aim)
        self.last_target_point = best_target

        if draw_debug:
            # Zeichne FOV und Fadenkreuz-Zentrum
            cv2.circle(frame, (self.center_x, self.center_y), self.fov_radius, (255, 0, 0), 1)
            cv2.circle(frame, (self.center_x, self.center_y), 2, (0, 255, 0), -1)
            
            # Markiere das aktuell anvisierte Ziel
            if best_target:
                cv2.circle(frame, (int(best_target[0]), int(best_target[1])), 4, (0, 0, 255), -1)

        # Ausführung
        if aim_enabled and self.is_right_click_pressed() and best_target:
            self._move_mouse_to_target(best_target[0], best_target[1])

        return frame if draw_debug else None

    def _move_mouse_to_target(self, target_x, target_y):
        """Kalkuliert den Vektor und führt die physikalische Mausbewegung aus."""
        offset_x = target_x - self.center_x
        offset_y = target_y - self.center_y

        move_x = int(offset_x * self.smooth_factor)
        move_y = int(offset_y * self.smooth_factor)

        if abs(move_x) > 0 or abs(move_y) > 0:
            pydirectinput.moveRel(move_x, move_y, relative=True)

    def cleanup(self):
        """Gibt Ressourcen frei, wichtig bei Programmende."""
        if hasattr(self, 'camera'):
            self.camera.stop()
            del self.camera
