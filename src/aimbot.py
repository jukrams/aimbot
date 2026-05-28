import ctypes
import dxcam
import numpy as np
import torch
from ultralytics import YOLO
from typing import Tuple, Optional, Dict

MOUSEEVENTF_MOVE = 0x0001
mouse_event = ctypes.windll.user32.mouse_event

class Aimbot:
    def __init__(self, config):
        self.config = config
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.use_half = self.device == 'cuda'
        
        self.camera = None
        self.model = None
        self.prev_target: Optional[Tuple[float, float]] = None
        
        self.apply_settings()

    def apply_settings(self):
        """Wird beim Start und nach Einstellungs-Updates aufgerufen."""
        self.width = int(self.config.get("resolution"))
        self.height = self.width
        self.center_x = self.width // 2
        self.center_y = self.height // 2
        
        self.fov_radius = float(self.config.get("fov"))
        self.conf_threshold = float(self.config.get("confidence"))
        self.base_smooth = float(self.config.get("smoothness"))
        self.head_offset = float(self.config.get("target_offset"))
        
        self.area_norm = 80000.0

        self._init_camera()
        self._init_model()

    def _init_camera(self):
        if self.camera:
            self.camera.stop()
            del self.camera
            
        screen_w = ctypes.windll.user32.GetSystemMetrics(0)
        screen_h = ctypes.windll.user32.GetSystemMetrics(1)
        left, top = (screen_w - self.width) // 2, (screen_h - self.height) // 2
        self.region = (left, top, left + self.width, top + self.height)

        self.camera = dxcam.create(output_color="BGR", max_buffer_len=1)
        self.camera.start(region=self.region, target_fps=60)

    def _init_model(self):
        model_path = self.config.get("model_type")
        self.model = YOLO(model_path, task='detect')

    @staticmethod
    def is_aim_key_pressed() -> bool:
        return (ctypes.windll.user32.GetAsyncKeyState(0x02) & 0x8000) != 0

    def process_frame(self, aim_enabled: bool) -> Optional[Dict]:
        frame = self.camera.get_latest_frame()
        if frame is None:
            return None

        # Klassen 0 (Person) und 1 (Kopf für Custom-Modelle) zulassen
        results = self.model.predict(frame, classes=[0, 1], verbose=False, device=self.device, half=self.use_half, max_det=5)
        
        data = {
            "frame": frame,
            "boxes": [], 
            "target": None,
            "fov": self.fov_radius,
            "center": (self.center_x, self.center_y)
        }

        if not results or len(results[0].boxes) == 0:
            self._reset_state()
            return data

        boxes = results[0].boxes.xyxy.cpu().numpy()
        confs = results[0].boxes.conf.cpu().numpy()
        classes = results[0].boxes.cls.cpu().numpy()

        target_data, box_render_data = self._evaluate_targets(boxes, confs, classes)
        data["boxes"] = box_render_data
        
        if target_data is not None:
            final_x, final_y, best_conf = target_data
            data["target"] = (final_x, final_y)
            
            if aim_enabled and self.is_aim_key_pressed():
                self._apply_mouse_movement(final_x, final_y, best_conf)
        else:
            self._reset_state()

        return data

    def _evaluate_targets(self, boxes: np.ndarray, confs: np.ndarray, classes: np.ndarray):
        w = boxes[:, 2] - boxes[:, 0]
        h = boxes[:, 3] - boxes[:, 1]
        
        # 1. Zielpunkt definieren
        t_x = boxes[:, 0] + w / 2
        t_y = np.where(classes == 1, boxes[:, 1] + h * 0.5, boxes[:, 1] + h * self.head_offset)
        
        dist = np.hypot(t_x - self.center_x, t_y - self.center_y)
        
        # Maskierung
        valid_mask = (dist <= self.fov_radius) & (confs >= self.conf_threshold) & (w <= h * 1.5)
        if not np.any(valid_mask):
            return None, []
            
        boxes, confs, classes = boxes[valid_mask], confs[valid_mask], classes[valid_mask]
        w, h, t_x, t_y, dist = w[valid_mask], h[valid_mask], t_x[valid_mask], t_y[valid_mask], dist[valid_mask]

        # 2. Normalisierung
        norm_dist = np.maximum(0.0, 1.0 - (dist / self.fov_radius))
        norm_size = np.minimum((w * h) / self.area_norm, 1.0)
        
        # 3. Scoring Matrix
        metrics = np.column_stack((norm_dist, confs, norm_size))
        scores = np.dot(metrics, np.array([0.60, 0.25, 0.15]))
        
        # 4. Target Lock Bonus
        if self.prev_target is not None:
            d_last = np.hypot(t_x - self.prev_target[0], t_y - self.prev_target[1])
            scores[d_last < 30] += 0.25 
            
        # 5. Head-Priorisierung
        scores[classes == 1] += 2.0

        # Render-Daten für GUI
        box_render_data = []
        for i in range(len(boxes)):
            box_render_data.append({
                "coords": boxes[i].tolist(),
                "score": round(float(scores[i] * 100), 1),
                "is_head": bool(classes[i] == 1)
            })
            
        best_idx = np.argmax(scores)
        best_t_x, best_t_y = t_x[best_idx], t_y[best_idx]
        
        self.prev_target = (best_t_x, best_t_y)
        return (best_t_x, best_t_y, confs[best_idx]), box_render_data

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
        if self.camera:
            self.camera.stop()
