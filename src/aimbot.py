import numpy as np
import math
import pydirectinput
from ultralytics import YOLO
import dxcam # Deutlich schneller als mss

pydirectinput.FAILSAFE = False

class AimbotOpt:
    def __init__(self, model_path="src/yolo26n.engine", width=416, height=416):
        # AI: Nutzung von TensorRT (.engine) statt .pt für max FPS
        self.model = YOLO(model_path, task='detect')
        
        self.width = width
        self.height = height
        self.center = np.array([width / 2, height / 2])
        
        # Performance: DXcam Setup (DirectX Frame Capturing)
        self.camera = dxcam.create(output_color="BGR")
        self.camera.start(target_fps=144, video_mode=True)
        
        # Parameter
        self.conf_threshold = 0.60
        self.head_offset = 0.18
        self.fov_radius = 150
        self.smooth_factor = 0.4

    def process_frame(self, aim_enabled):
        # Performance: Frame direkt aus dem Puffer lesen
        frame = self.camera.get_latest_frame()
        if frame is None:
            return

        # Wir schneiden den Center-Crop aus dem Vollbild
        h, w = frame.shape[:2]
        crop_y = (h - self.height) // 2
        crop_x = (w - self.width) // 2
        frame_cropped = frame[crop_y:crop_y+self.height, crop_x:crop_x+self.width]

        # Inferenz (Half-Precision wird von TensorRT automatisch gehandhabt)
        results = self.model.predict(frame_cropped, classes=[0], verbose=False)
        r = results[0]

        # Vektorisierung (Python/Math): Keine For-Schleifen mehr!
        boxes = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()

        if len(boxes) == 0:
            return

        # 1. Filtern nach Confidence
        valid_mask = confs >= self.conf_threshold
        boxes = boxes[valid_mask]
        confs = confs[valid_mask]

        if len(boxes) == 0:
            return

        # 2. Koordinaten und Dimensionen als Vektoren berechnen
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        widths = x2 - x1
        heights = y2 - y1

        # 3. Aspekt-Ratio Filter (Maske aktualisieren)
        ratio_mask = widths <= heights * 1.2
        boxes = boxes[ratio_mask]
        confs = confs[ratio_mask]
        widths = widths[ratio_mask]
        heights = heights[ratio_mask]
        x1, y1 = x1[ratio_mask], y1[ratio_mask]

        if len(boxes) == 0:
            return

        # 4. Zielpunkte berechnen
        target_x = x1 + widths / 2
        target_y = y1 + heights * self.head_offset
        targets = np.column_stack((target_x, target_y))

        # 5. Distanzen zum Zentrum vektorisiert berechnen
        distances = np.linalg.norm(targets - self.center, axis=1)

        # 6. Hard-FOV Filter
        fov_mask = distances <= self.fov_radius
        confs = confs[fov_mask]
        targets = targets[fov_mask]
        distances = distances[fov_mask]

        if len(targets) == 0:
            return

        # 7. Math: Normalisierter Score (0.0 bis 1.0)
        # Distanz umkehren: je kleiner die Distanz, desto näher an 1
        normalized_distances = 1.0 - (distances / self.fov_radius)
        
        # Gewichtung: 70% Fokus auf Distanz (Fadenkreuznähe), 30% auf Modell-Confidence
        scores = (confs * 0.3) + (normalized_distances * 0.7)

        # 8. Bestes Ziel finden (argmax gibt den Index des höchsten Scores)
        best_idx = np.argmax(scores)
        best_target = targets[best_idx]

        # Input ausführen
        if aim_enabled: # Hier fehlt noch der Hardware-Input-Check
            self._move_mouse_to_target(best_target[0], best_target[1])

    def _move_mouse_to_target(self, target_x, target_y):
        offset = np.array([target_x, target_y]) - self.center
        move = (offset * self.smooth_factor).astype(int)

        if np.any(np.abs(move) > 0):
            # Gaming Engineer Info: pydirectinput ist unsafe. 
            # In einem echten Produkt würde hier serielle Kommunikation zu einem Arduino stattfinden.
            pydirectinput.moveRel(move[0], move[1], relative=True)

    def cleanup(self):
        self.camera.stop()
