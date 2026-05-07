import math
import ctypes
import dxcam
import numpy as np
from ultralytics import YOLO
from typing import Tuple, Optional

# C-Types Setup für minimalen Input-Lag (Umgeht pydirectinput Overhead)
MOUSEEVENTF_MOVE = 0x0001
mouse_event = ctypes.windll.user32.mouse_event

class Aimbot:
    def __init__(self, model_path: str = "src/yolo11s.engine", width: int = 416, height: int = 416):
        self.width = width
        self.height = height
        self.center_x = width // 2
        self.center_y = height // 2
        
        # Kamera-Setup
        self.camera = dxcam.create(output_color="BGR")
        screen_w = ctypes.windll.user32.GetSystemMetrics(0)
        screen_h = ctypes.windll.user32.GetSystemMetrics(1)
        left, top = (screen_w - width) // 2, (screen_h - height) // 2
        self.region = (left, top, left + width, top + height)
        
        # Modell-Laden (Half-Precision erzwingen)
        self.model = YOLO(model_path, task='detect')
        
        # --- HYPERPARAMETER ---
        self.conf_threshold = 0.60
        self.fov_radius = 60.0
        self.head_offset = 0.18
        self.base_smooth = 0.35
        self.area_norm = 80000.0
        
        self.weights = np.array([0.60, 0.25, 0.15]) # Dist, Conf, Size
        
        # --- STATE (EWMA Prädiktion) ---
        self.prev_target: Optional[Tuple[float, float]] = None
        self.velocity = np.array([0.0, 0.0])
        self.alpha = 0.6 # EWMA Glättungsfaktor für Geschwindigkeit
        self.prediction_frames = 1.5 # Wieviele Frames in die Zukunft

    @staticmethod
    def is_aim_key_pressed() -> bool:
        """Prüft asynchron den Status der rechten Maustaste (VK_RBUTTON = 0x02)."""
        return ctypes.windll.user32.GetAsyncKeyState(0x02) != 0

    def process_frame(self, aim_enabled: bool, draw_debug: bool = False) -> Optional[np.ndarray]:
        frame = self.camera.grab(region=self.region)
        if frame is None:
            return None

        # Inferenz ohne Gradientenberechnung
        results = self.model.predict(frame, classes=[0], verbose=False, half=True, max_det=10)
        
        if not results or len(results[0].boxes) == 0:
            self._reset_state()
            return frame if draw_debug else None

        # Block-Transfer auf CPU für vektorisierte Operationen
        boxes = results[0].boxes.xyxy.cpu().numpy()
        confs = results[0].boxes.conf.cpu().numpy()
        
        target = self._evaluate_targets(boxes, confs)
        
        if target is not None:
            final_x, final_y, best_conf = target
            
            if aim_enabled and self.is_aim_key_pressed():
                self._apply_mouse_movement(final_x, final_y, best_conf)
        else:
            self._reset_state()

        return frame if draw_debug else None

    def _evaluate_targets(self, boxes: np.ndarray, confs: np.ndarray) -> Optional[Tuple[float, float, float]]:
        """Vektorisierte Berechnung der Ziel-Scores."""
        # Breiten und Höhen
        w = boxes[:, 2] - boxes[:, 0]
        h = boxes[:, 3] - boxes[:, 1]
        
        # Leichenschutz: Aspekt-Ratio Filter (Breite > Höhe * 1.1)
        valid_mask = (w <= h * 1.1) & (confs >= self.conf_threshold)
        if not np.any(valid_mask):
            return None
            
        boxes, confs, w, h = boxes[valid_mask], confs[valid_mask], w[valid_mask], h[valid_mask]
        
        # Zielkoordinaten berechnen
        t_x = boxes[:, 0] + w / 2
        t_y = boxes[:, 1] + h * self.head_offset
        
        # Distanzen zum Zentrum (Vektorisiert)
        dist = np.hypot(t_x - self.center_x, t_y - self.center_y)
        fov_mask = dist <= self.fov_radius
        if not np.any(fov_mask):
            return None
            
        t_x, t_y, confs, w, h, dist = t_x[fov_mask], t_y[fov_mask], confs[fov_mask], w[fov_mask], h[fov_mask], dist[fov_mask]

        # Normalisierte Metriken berechnen
        norm_dist = np.maximum(0.0, 1.0 - (dist / self.fov_radius)) ** 2
        norm_size = np.minimum((w * h) / self.area_norm, 1.0)
        
        # Score-Matrix Multiplikation
        metrics = np.column_stack((norm_dist, confs, norm_size))
        scores = np.dot(metrics, self.weights)
        
        # Hysterese: Sticky Target Bonus (Vermeidet Target-Flickering)
        if self.prev_target is not None:
            d_last = np.hypot(t_x - self.prev_target[0], t_y - self.prev_target[1])
            scores[d_last < 25] += 0.15
            
        best_idx = np.argmax(scores)
        best_t_x, best_t_y = t_x[best_idx], t_y[best_idx]
        
        # Prädiktion via EWMA
        if self.prev_target is not None:
            curr_velocity = np.array([best_t_x - self.prev_target[0], best_t_y - self.prev_target[1]])
            self.velocity = self.alpha * curr_velocity + (1 - self.alpha) * self.velocity
            
            # Prädizierte Position
            best_t_x += self.velocity[0] * self.prediction_frames
            best_t_y += self.velocity[1] * self.prediction_frames

        self.prev_target = (best_t_x, best_t_y)
        return best_t_x, best_t_y, confs[best_idx]

    def _apply_mouse_movement(self, target_x: float, target_y: float, confidence: float):
        """Bewegt die Maus via ctypes (Raw Input)."""
        offset_x = target_x - self.center_x
        offset_y = target_y - self.center_y

        dynamic_smooth = self.base_smooth * (confidence ** 2)
        move_x = int(offset_x * dynamic_smooth)
        move_y = int(offset_y * dynamic_smooth)

        if move_x or move_y:
            mouse_event(MOUSEEVENTF_MOVE, move_x, move_y, 0, 0)

    def _reset_state(self):
        self.prev_target = None
        self.velocity = np.array([0.0, 0.0])

    def cleanup(self):
        if hasattr(self, 'camera'):
            self.camera.stop()
