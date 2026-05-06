import cv2
import numpy as np
import pydirectinput
import win32api
import math
from ultralytics import YOLO

# Deaktiviere pydirectinput Failsafe, falls die Maus bei schnellen Bewegungen an den Rand kommt
pydirectinput.FAILSAFE = False

class Aimbot:
    def __init__(self, model_path="src/yolo26n.pt", width=416, height=416):
        # Erzwinge GPU Nutzung, deaktiviere Konsolen-Spam
        self.model = YOLO(model_path)
        
        # Bildschirm-Konfiguration (Performance-Crop)
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
        
        # Basis-Einstellungen
        self.conf_threshold = 0.60 # Grundlimit (alles darunter wird sofort ignoriert)
        self.head_offset = 0.18    # Zielt auf das obere Fünftel der Bounding Box
        self.smooth_factor = 0.4   # 1.0 = Instant Snap, 0.1 = langsames Nachziehen

    def is_right_click_pressed(self):
        """Prüft asynchron den Status der rechten Maustaste."""
        return win32api.GetAsyncKeyState(0x02) < 0

    def process_frame(self, sct, aim_enabled, draw_debug=False):
        """Haupt-Loop: Bild erfassen, analysieren und Maus steuern."""
        # Screenshot ziehen und konvertieren
        img_bgra = np.array(sct.grab(self.monitor))
        frame = cv2.cvtColor(img_bgra, cv2.COLOR_BGRA2BGR)

        # Inferenz (Nur Klasse 0: Person)
        results = self.model.predict(frame, classes=[0], verbose=False, device=0)
        
        best_target = None
        highest_score = -float('inf')
        
        # --- LOGIK-FILTER PARAMETER ---
        fov_radius = 150         # Ignoriere strikt alles außerhalb dieses Pixel-Radius
        dist_weight = 0.5        # Bestrafungsfaktor für Distanz zum Zentrum
        # ------------------------------

        for r in results:
            boxes = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()

            for i, box in enumerate(boxes):
                conf = confs[i]
                
                # 1. Confidence-Check
                if conf < self.conf_threshold:
                    continue

                x1, y1, x2, y2 = box
                width = x2 - x1
                height = y2 - y1

                # 2. Aspekt-Ratio Filter (Ausschluss von liegenden Objekten/Leichen)
                if width > height * 1.2: 
                    continue

                # Zielpunkt auf dem Modell berechnen
                target_x = x1 + width / 2
                target_y = y1 + height * self.head_offset

                # Distanz zum Zentrum berechnen
                dist = math.hypot(target_x - self.center_x, target_y - self.center_y)

                # 3. Hard-FOV Filter (Legitbot-Verhalten)
                if dist > fov_radius:
                    continue

                # 4. Score-Berechnung
                score = (conf * 100) - (dist * dist_weight)

                # Bestes Ziel anhand des Scores identifizieren
                if score > highest_score:
                    highest_score = score
                    best_target = (target_x, target_y)

                # Debug-Ansicht: Zeichnet Boxen und Scores für ALLE gültigen Ziele im FOV
                if draw_debug:
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    cv2.putText(frame, f"S:{int(score)}", (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,0), 1)

        # Debug-Ansicht: Fadenkreuz-Zentrum, Logik-FOV und finaler Zielpunkt
        if draw_debug:
            # Zeigt den aktiven FOV-Bereich
            cv2.circle(frame, (self.center_x, self.center_y), fov_radius, (255, 0, 0), 1)
            if best_target:
                # Markiert das aktuell anvisierte Ziel rot
                cv2.circle(frame, (int(best_target[0]), int(best_target[1])), 4, (0, 0, 255), -1)

        # Ausführung der Bewegung
        if aim_enabled and self.is_right_click_pressed() and best_target is not None:
            self._move_mouse_to_target(best_target[0], best_target[1])

        return frame if draw_debug else None

    def _move_mouse_to_target(self, target_x, target_y):
        """Übersetzt Pixel-Deltas in simulierte, geglättete Maus-Inputs."""
        offset_x = target_x - self.center_x
        offset_y = target_y - self.center_y

        # Smoothing zur Humanisierung der Bewegung
        move_x = int(offset_x * self.smooth_factor)
        move_y = int(offset_y * self.smooth_factor)

        # Filter gegen Micro-Jitter (Zittern im Sub-Pixel Bereich)
        if abs(move_x) > 0 or abs(move_y) > 0:
            pydirectinput.moveRel(move_x, move_y, relative=True)
