# YOLO26 Aim-Assist System

Ein hocheffizientes Objekterkennungssystem zur automatisierten Zielerfassung von Personen in Echtzeit. Basierend auf der modernsten **YOLO26-Architektur** und optimiert für minimale Latenz durch einen zentrierten 416x416 Screen-Crop.

---

## 🚀 Features

*   **YOLO26 Integration:** Nutzt das aktuellste NMS-free Modell für maximale Inferenzgeschwindigkeit.
*   **Center-Crop (416x416):** Reduziert die Prozessorlast durch Fokus auf die Bildschirmmitte.
*   **Relative Mausbewegung:** Kompatibel mit Game-Engines durch `pydirectinput`.
*   **Intelligentes Targeting:** Automatischer Fokus auf den Kopfbereich (oberes Fünftel der Bounding Box).
*   **Duales Kontrollsystem:** 
    *   **F1:** System aktivieren/deaktivieren.
    *   **Rechtsklick:** Aim-Assist nur bei gedrückter Taste auslösen.
*   **Live-Visualisierung:** Anzeige von FPS, Bounding Boxes und Konfidenz-Werten in Prozent.

---

## 📂 Projektstruktur

```text
/projekt-ordner
├── main.py              # Programmstart, UI und Steuerung
├── requirements.txt     # Abhängigkeiten
├── README.md            # Dokumentation
└── src/
    ├── aimbot.py        # Kernlogik (Detektion & Maussteuerung)
    └── yolo26n.pt       # YOLO-Modell (wird automatisch geladen)
