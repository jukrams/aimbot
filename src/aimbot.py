import math
import ctypes
import dxcam
import numpy as np
import torch
import cv2
from ultralytics import YOLO
from typing import Tuple, Optional

MOUSEEVENTF_MOVE = 0x0001
mouse_event = ctypes.windll.user32.mouse_event

class Aimbot:
    def __init__(self, model_path: str = "src/yolov8n.pt", width: int = 416, height: int = 416):
        self.width = width
        self.height = height
        self.center_x = width // 2
        self.center_y = height // 2
        
        screen_w = ctypes.windll.user32.GetSystemMetrics(0)
        screen_h = ctypes.windll.user32.GetSystemMetrics(1)
        
        left, top = (screen_w - width) // 2, (screen_h - height) // 2
        self.region = (left, top, left + width, top + height)

        self.camera = dxcam.create(output_color="BGR", max_buffer_len=1)
        self.camera.start(region=self.region, target_fps=60)
        
        # Hardware-Check für YOLO
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.use_half = self.device == 'cuda'
        self.model = YOLO(model_path, task='detect')
        
        self.conf_threshold = 0.60
        self.fov_radius = 200.0
        self.head_offset = 0.18
        self.base_smooth = 0.35
        self.area_norm = 80000.0
        
        self.weights = np.array([0.60, 0.25, 0.15]) 
        self.prev_target: Optional[Tuple[float, float]] = None

    @staticmethod
    def is_aim_key_pressed() -> bool:
        # 0x8000 prüft, ob die Taste aktuell gehalten wird
        return (ctypes.windll.user32.GetAsyncKeyState(0x02) & 0x8000) != 0

    def process_frame(self, aim_enabled: bool, draw_debug: bool = False) -> Optional[np.ndarray]:
        frame = self.camera.get_latest_frame()
        
        if frame is None:
            return None

        # Dynamische Präzision basierend auf der Hardware
        results = self.model.predict(frame, classes=[0], verbose=False, device=self.device, half=self.use_half, max_det=5)
        
        if not results or len(results[0].boxes) == 0:
            self._reset_state()
            return frame if draw_debug else None

        # Koordinaten auslesen
        boxes = results[0].boxes.xyxy.cpu().numpy()
        confs = results[0].boxes.conf.cpu().numpy()
        
        # --- VISUALISIERUNG: Alle erkannten Ziele (Grün) ---
        if draw_debug:
            for i in range(len(boxes)):
                x1, y1, x2, y2 = map(int, boxes[i])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Bestes Ziel evaluieren
        target = self._evaluate_targets(boxes, confs)
        
        if target is not None:
            final_x, final_y, best_conf = target
            
            # --- VISUALISIERUNG: Das anvisierte Ziel (Rot) ---
            if draw_debug:
                cv2.circle(frame, (int(final_x), int(final_y)), 4, (0, 0, 255), -1)
            
            if aim_enabled and self.is_aim_key_pressed():
                self._apply_mouse_movement(final_x, final_y, best_conf)
        else:
            self._reset_state()

        return frame if draw_debug else None

    def _evaluate_targets(self, boxes: np.ndarray, confs: np.ndarray) -> Optional[Tuple[float, float, float]]:
        w = boxes[:, 2] - boxes[:, 0]
        h = boxes[:, 3] - boxes[:, 1]
        
        valid_mask = (w <= h * 1.1) & (confs >= self.conf_threshold)
        if not np.any(valid_mask):
            return None
            
        boxes, confs, w, h = boxes[valid_mask], confs[valid_mask], w[valid_mask], h[valid_mask]
        
        t_x = boxes[:, 0] + w / 2
        t_y = boxes[:, 1] + h * self.head_offset
        
        dist = np.hypot(t_x - self.center_x, t_y - self.center_y)
        
        fov_mask = dist <= self.fov_radius
        if not np.any(fov_mask):
            return None
            
        t_x, t_y, confs, w, h, dist = t_x[fov_mask], t_y[fov_mask], confs[fov_mask], w[fov_mask], h[fov_mask], dist[fov_mask]

        norm_dist = np.maximum(0.0, 1.0 - (dist / self.fov_radius)) ** 2
        norm_size = np.minimum((w * h) / self.area_norm, 1.0)
        
        metrics = np.column_stack((norm_dist, confs, norm_size))
        scores = np.dot(metrics, self.weights)
        
        if self.prev_target is not None:
            d_last = np.hypot(t_x - self.prev_target[0], t_y - self.prev_target[1])
            scores[d_last < 30] += 0.15
            
        best_idx = np.argmax(scores)
        best_t_x, best_t_y = t_x[best_idx], t_y[best_idx]

        self.prev_target = (best_t_x, best_t_y)
        return best_t_x, best_t_y, confs[best_idx]

    def _apply_mouse_movement(self, target_x: float, target_y: float, confidence: float):
        offset_x = target_x - self.center_x
        offset_y = target_y - self.center_y

        dynamic_smooth = self.base_smooth * (confidence ** 2)
        
        move_x = int(offset_x * dynamic_smooth)
        move_y = int(offset_y * dynamic_smooth)

        if move_x or move_y:
            mouse_event(MOUSEEVENTF_MOVE, move_x, move_y, 0, 0)

    def _reset_state(self):
        self.prev_target = None

    def cleanup(self):
        if hasattr(self, 'camera'):
            self.camera.stop()
            del self.camera
