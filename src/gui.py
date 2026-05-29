import time
from PyQt6.QtCore import QThread, pyqtSignal, Qt
from PyQt6.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QFont
from PyQt6.QtWidgets import (QLabel, QWidget, QVBoxLayout, QHBoxLayout, 
                             QPushButton, QDialog, QTabWidget, QComboBox, 
                             QSlider, QFormLayout, QInputDialog, QMessageBox, QCheckBox)

# --- WORKER THREAD ---
class InferenceThread(QThread):
    frame_ready = pyqtSignal(dict)

    def __init__(self, aimbot):
        super().__init__()
        self.aimbot = aimbot
        self.running = True
        self.aim_active = False
        self.paused = False
        self.last_frame_id = None

    def run(self):
        while self.running:
            if self.paused:
                time.sleep(0.1)
                continue
                
            data = self.aimbot.get_latest_data()
            
            # Nur senden, wenn Daten da sind und es sich um einen neuen Frame handelt
            if data is not None and id(data["frame"]) != self.last_frame_id:
                self.last_frame_id = id(data["frame"])
                self.frame_ready.emit(data)
            else:
                # Schont die CPU, wenn die KI noch kein neues Bild geliefert hat
                time.sleep(0.001)

    def stop(self):
        self.running = False
        self.wait()

# --- SETTINGS DIALOG ---
class SettingsDialog(QDialog):
    def __init__(self, config, parent=None):
        super().__init__(parent)
        self.config = config
        self.setWindowTitle("System Einstellungen")
        self.setFixedSize(400, 400)
        self.setStyleSheet("background-color: #121620; color: white;")
        
        layout = QVBoxLayout()
        
        # --- PROFIL-MANAGEMENT ---
        profile_layout = QHBoxLayout()
        self.combo_profile = QComboBox()
        self.combo_profile.addItems(self.config.get_profile_names())
        self.combo_profile.setCurrentText(self.config.data["active_profile"])
        self.combo_profile.currentTextChanged.connect(self._on_profile_changed)

        btn_new = QPushButton("Neu")
        btn_new.setStyleSheet("background-color: #4169E1; font-weight: bold;")
        btn_new.clicked.connect(self._create_profile)
        
        btn_delete = QPushButton("Löschen")
        btn_delete.setStyleSheet("background-color: #d9534f; font-weight: bold;")
        btn_delete.clicked.connect(self._delete_profile)

        profile_layout.addWidget(QLabel("Profil:"))
        profile_layout.addWidget(self.combo_profile, stretch=1)
        profile_layout.addWidget(btn_new)
        profile_layout.addWidget(btn_delete)
        layout.addLayout(profile_layout)

        # --- EINSTELLUNGS-TABS ---
        tabs = QTabWidget()
        tabs.setStyleSheet("""
            QTabBar::tab { background: #282e3d; padding: 8px 15px; margin-right: 2px; }
            QTabBar::tab:selected { background: #4169E1; }
            QTabWidget::pane { border: 1px solid #4169E1; }
        """)

        # Tab 1: System
        tab_sys = QWidget()
        sys_layout = QFormLayout()
        self.combo_res = QComboBox()
        self.combo_res.addItems(["320", "416", "640"])
        self.combo_model = QComboBox()
        self.combo_model.addItems(["yolov8n.pt", "yolov8s.pt", "src/custom_model.pt"])
        
        self.chk_visuals = QCheckBox("Visuals anzeigen (Boxen/FOV)")
        self.chk_visuals.setChecked(True)

        sys_layout.addRow("Erkennungs-Auflösung:", self.combo_res)
        sys_layout.addRow("YOLO Modell:", self.combo_model)
        sys_layout.addRow("", self.chk_visuals)
        tab_sys.setLayout(sys_layout)

        # Tab 2: Aim Logik
        tab_aim = QWidget()
        aim_layout = QFormLayout()
        
        self.fov_container, self.slider_fov, self.lbl_fov = self._create_slider_with_label(64, 320, " px")
        self.conf_container, self.slider_conf, self.lbl_conf = self._create_slider_with_label(60, 95, " %")
        self.smooth_container, self.slider_smooth, self.lbl_smooth = self._create_slider_with_label(1, 100, " %")
        self.offset_container, self.slider_offset, self.lbl_offset = self._create_slider_with_label(0, 50, " %")
        
        aim_layout.addRow("FOV Radius:", self.fov_container)
        aim_layout.addRow("Confidence:", self.conf_container)
        aim_layout.addRow("Smoothness:", self.smooth_container)
        aim_layout.addRow("Trefferzone (Offset):", self.offset_container)
        tab_aim.setLayout(aim_layout)

        tabs.addTab(tab_sys, "Modell & Bild")
        tabs.addTab(tab_aim, "Aim-Logik")
        layout.addWidget(tabs)

        self.combo_res.currentTextChanged.connect(self._update_fov_limits)
        self._sync_ui_with_config()

        # --- SPEICHERN ---
        btn_save = QPushButton("Speichern & Anwenden")
        btn_save.setStyleSheet("background-color: #4169E1; padding: 10px; font-weight: bold; margin-top: 10px;")
        btn_save.clicked.connect(self.save_and_close)
        layout.addWidget(btn_save)
        
        self.setLayout(layout)

    def _create_slider_with_label(self, min_val, max_val, suffix=""):
        container = QWidget()
        layout = QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setMinimum(min_val)
        slider.setMaximum(max_val)
        
        label = QLabel()
        label.setMinimumWidth(45)
        label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        label.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
        
        slider.valueChanged.connect(lambda v: label.setText(f"{v}{suffix}"))
        layout.addWidget(slider)
        layout.addWidget(label)
        
        return container, slider, label

    def _update_fov_limits(self, _=None):
        res = int(self.combo_res.currentText())
        min_fov = int(res * 0.20)
        max_fov = res
        
        self.slider_fov.setMinimum(min_fov)
        self.slider_fov.setMaximum(max_fov)
        
        current_val = self.slider_fov.value()
        if current_val < min_fov:
            self.slider_fov.setValue(min_fov)
        elif current_val > max_fov:
            self.slider_fov.setValue(max_fov)

    def _on_profile_changed(self, name):
        if name:
            self.config.set_active_profile(name)
            self._sync_ui_with_config()

    def _create_profile(self):
        name, ok = QInputDialog.getText(self, "Neues Profil", "Profilname:")
        if ok and name.strip():
            name = name.strip()
            if name in self.config.get_profile_names():
                QMessageBox.warning(self, "Fehler", "Dieser Profilname existiert bereits.")
                return
            
            self._save_ui_to_config() 
            self.config.create_profile(name)
            
            self.combo_profile.blockSignals(True)
            self.combo_profile.addItem(name)
            self.combo_profile.setCurrentText(name)
            self.combo_profile.blockSignals(False)

    def _delete_profile(self):
        name = self.combo_profile.currentText()
        if name == "Standard":
            QMessageBox.warning(self, "Fehler", "Das Standard-Profil kann nicht gelöscht werden.")
            return
            
        reply = QMessageBox.question(self, "Profil löschen", f"Profil '{name}' wirklich löschen?",
                                     QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        
        if reply == QMessageBox.StandardButton.Yes:
            self.config.delete_profile(name)
            self.combo_profile.blockSignals(True)
            self.combo_profile.removeItem(self.combo_profile.findText(name))
            self.combo_profile.setCurrentText(self.config.data["active_profile"])
            self.combo_profile.blockSignals(False)
            self._sync_ui_with_config()

    def _sync_ui_with_config(self):
        self.combo_res.blockSignals(True)
        self.combo_res.setCurrentText(str(self.config.get("resolution")))
        self.combo_res.blockSignals(False)
        
        self.combo_model.setCurrentText(self.config.get("model_type"))
        self._update_fov_limits()
        self.slider_fov.setValue(int(self.config.get("fov")))
        
        conf_val = int(self.config.get("confidence") * 100)
        self.slider_conf.setValue(max(60, min(95, conf_val)))
        
        smooth_val = int(self.config.get("smoothness") * 100)
        self.slider_smooth.setValue(max(1, min(100, smooth_val)))

        self.slider_fov.valueChanged.emit(self.slider_fov.value())
        self.slider_conf.valueChanged.emit(self.slider_conf.value())
        self.slider_smooth.valueChanged.emit(self.slider_smooth.value())

        self.chk_visuals.setChecked(self.config.get("show_visuals"))
        
        offset_val = int(self.config.get("target_offset") * 100)
        self.slider_offset.setValue(max(0, min(50, offset_val)))
        self.slider_offset.valueChanged.emit(self.slider_offset.value())

    def _save_ui_to_config(self):
        self.config.set("resolution", int(self.combo_res.currentText()))
        self.config.set("model_type", self.combo_model.currentText())
        self.config.set("fov", self.slider_fov.value())
        self.config.set("confidence", self.slider_conf.value() / 100.0)
        self.config.set("smoothness", self.slider_smooth.value() / 100.0)
        self.config.set("show_visuals", self.chk_visuals.isChecked())
        self.config.set("target_offset", self.slider_offset.value() / 100.0)

    def save_and_close(self):
        self._save_ui_to_config()
        self.config.save()
        self.accept()

# --- MAIN GUI ---
class OverlayGUI(QWidget):
    toggle_requested = pyqtSignal(bool)
    exit_requested = pyqtSignal()
    settings_changed = pyqtSignal()

    def __init__(self, aimbot, config):
        super().__init__()
        self.aimbot = aimbot
        self.config = config
        self.setWindowTitle("System Debug View")
        self.aim_active = False

        self.DISPLAY_SIZE = 640
        self.setFixedSize(self.DISPLAY_SIZE + 30, self.DISPLAY_SIZE + 110)

        self.setStyleSheet("""
            QWidget { background-color: #121620; color: #ffffff; font-family: 'Segoe UI'; }
            QPushButton { background-color: #4169E1; border: none; padding: 10px; font-weight: bold; border-radius: 6px; }
            QPushButton:hover { background-color: #5a7ff3; }
            QPushButton:pressed { background-color: #2b4ec2; }
        """)
        
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(15, 15, 15, 15)

        self.label = QLabel(self)
        self.label.setFixedSize(self.DISPLAY_SIZE, self.DISPLAY_SIZE)
        self.label.setStyleSheet("border: 2px solid #4169E1; background-color: #090b11;")
        main_layout.addWidget(self.label, alignment=Qt.AlignmentFlag.AlignCenter)

        btn_layout = QHBoxLayout()
        self.btn_toggle = QPushButton("Start (F1)")
        self.btn_settings = QPushButton("Einstellungen")
        self.btn_exit = QPushButton("Beenden (F2)")
        self.btn_exit.setStyleSheet("background-color: #282e3d; color: #a0a5b5;")
        
        self.btn_toggle.clicked.connect(lambda: self.toggle_requested.emit(not self.aim_active))
        self.btn_settings.clicked.connect(self.open_settings)
        self.btn_exit.clicked.connect(self.exit_requested.emit)

        btn_layout.addWidget(self.btn_toggle)
        btn_layout.addWidget(self.btn_settings)
        btn_layout.addWidget(self.btn_exit)
        main_layout.addLayout(btn_layout)
        self.setLayout(main_layout)

        self.thread = InferenceThread(aimbot)
        self.thread.frame_ready.connect(self.draw_overlays)
        self.thread.start()

    def set_aim_state(self, state: bool):
        self.aim_active = state
        self.thread.aim_active = state
        if state:
            self.btn_toggle.setText("Stopp (F1)")
            self.btn_toggle.setStyleSheet("background-color: #d9534f;")
        else:
            self.btn_toggle.setText("Start (F1)")
            self.btn_toggle.setStyleSheet("")

    def open_settings(self):
        self.thread.paused = True
        dialog = SettingsDialog(self.config, self)
        if dialog.exec():
            self.settings_changed.emit()
        self.thread.paused = False

    def draw_overlays(self, data: dict):
        frame = data["frame"]
        rgb_image = frame[:, :, ::-1]
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        
        q_img = QImage(rgb_image.tobytes(), w, h, bytes_per_line, QImage.Format.Format_RGB888)
        q_img_scaled = q_img.scaled(self.DISPLAY_SIZE, self.DISPLAY_SIZE, Qt.AspectRatioMode.KeepAspectRatio)
        pixmap = QPixmap.fromImage(q_img_scaled)
        
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        painter.setPen(QPen(QColor(255, 255, 0), 2))
        painter.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
        painter.drawText(10, 25, f"FPS: {data['fps']}")

        if self.config.get("show_visuals"):
            scale_ratio = self.DISPLAY_SIZE / w
            
            c_x, c_y = data["center"][0] * scale_ratio, data["center"][1] * scale_ratio
            fov = data["fov"] * scale_ratio
            painter.setPen(QPen(QColor(65, 105, 225), 2)) 
            painter.drawEllipse(int(c_x - fov), int(c_y - fov), int(fov * 2), int(fov * 2))

            painter.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
            for box_data in data["boxes"]:
                x1, y1, x2, y2 = [int(v * scale_ratio) for v in box_data["coords"]]
                score = box_data["score"]
                is_head = box_data["is_head"]
                
                color = QColor(255, 105, 180) if is_head else QColor(0, 255, 64)
                painter.setPen(QPen(color, 2))
                
                painter.drawRect(x1, y1, x2 - x1, y2 - y1)
                
                text_str = f"S: {score}"
                text_rect = painter.fontMetrics().boundingRect(text_str)
                text_rect.moveTo(x1, y1 - 15)
                text_rect.adjust(-2, -2, 2, 2)
                
                painter.fillRect(text_rect, QColor(0, 0, 0, 150))
                painter.setPen(QPen(QColor(255, 255, 255)))
                painter.drawText(x1, y1 - 4, text_str)

            if data["target"]:
                t_x, t_y = int(data["target"][0] * scale_ratio), int(data["target"][1] * scale_ratio)
                painter.setPen(QPen(QColor(255, 50, 50), 2))
                painter.setBrush(QColor(255, 50, 50))
                painter.drawEllipse(t_x - 4, t_y - 4, 8, 8)

        painter.end()
        self.label.setPixmap(pixmap)

    def closeEvent(self, event):
        self.thread.stop()
        super().closeEvent(event)
