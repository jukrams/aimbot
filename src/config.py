import json
import os

class ConfigManager:
    # Das Basis-Profil, von dem neue Profile erben
    DEFAULT_SETTINGS = {
        "resolution": 320,
        "model_type": "yolov8n.pt",
        "fov": 200,
        "confidence": 0.60,
        "smoothness": 0.35,
        "target_offset": 0.18,  # NEU: 0.0 = Kopf, 0.5 = Mitte
        "show_visuals": True    # NEU: True = Zeichnen, False = Unsichtbar
    }

    def __init__(self, filename="config.json"):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.filepath = os.path.join(base_dir, filename)
        
        # Neue Datenstruktur: Profile und ein aktiver Zeiger
        self.data = {
            "profiles": {
                "Standard": self.DEFAULT_SETTINGS.copy()
            },
            "active_profile": "Standard"
        }
        self.load()

    def load(self):
        """Lädt die Profildaten. Repariert fehlende oder fehlerhafte Strukturen."""
        if os.path.exists(self.filepath):
            try:
                with open(self.filepath, "r") as f:
                    loaded_data = json.load(f)
                    # Validierung der neuen Struktur
                    if "profiles" in loaded_data and "active_profile" in loaded_data:
                        self.data = loaded_data
            except json.JSONDecodeError:
                pass 

        # Sicherheitsnetz: Das 'Standard'-Profil muss immer existieren
        if "Standard" not in self.data["profiles"]:
            self.data["profiles"]["Standard"] = self.DEFAULT_SETTINGS.copy()

        # Sicherheitsnetz: Fallback, falls das aktive Profil gelöscht wurde
        if self.data["active_profile"] not in self.data["profiles"]:
            self.data["active_profile"] = "Standard"

    def save(self):
        """Sichert die gesamte Struktur auf die Festplatte."""
        with open(self.filepath, "w") as f:
            json.dump(self.data, f, indent=4)
            
    @property
    def active_settings(self):
        """Liefert das Dictionary des aktuell ausgewählten Profils."""
        return self.data["profiles"][self.data["active_profile"]]

    # --- Datenzugriff für aimbot.py ---
    def get(self, key):
        return self.active_settings.get(key, self.DEFAULT_SETTINGS.get(key))

    def set(self, key, value):
        self.active_settings[key] = value

    # --- Profil-Management für gui.py ---
    def set_active_profile(self, name):
        if name in self.data["profiles"]:
            self.data["active_profile"] = name

    def create_profile(self, name):
        if name and name not in self.data["profiles"]:
            # Neues Profil erbt die Werte des gerade aktiven Profils
            self.data["profiles"][name] = self.active_settings.copy()
            self.data["active_profile"] = name

    def delete_profile(self, name):
        if name in self.data["profiles"] and name != "Standard":
            del self.data["profiles"][name]
            if self.data["active_profile"] == name:
                self.data["active_profile"] = "Standard"

    def get_profile_names(self):
        return list(self.data["profiles"].keys())