# Skytouch

Skytouch is a real-time computer vision application that turns hand gestures into a digital canvas, image editor, and video filter interface using OpenCV and MediaPipe.

---

## Features & Controls

### Hand Gestures
* **Freehand Draw:** Raise **Index Finger** only.
* **Drag Bounding Shape:** **Pinch** (Thumb + Index Finger) and drag across the screen.
* **Cycle Color:** Raise **Index + Pinky** fingers (Hold).
* **Cycle Shape Mode:** Raise **Index + Middle** fingers (Hold).
* **Cycle Brush Style:** Raise **Index + Middle + Ring** fingers (Hold).
* **Undo Action:** **Pinky + Thumb** out (Surfer / Shaka gesture).
* **Clear Canvas:** Hold a closed **Fist** (fills clear progress bar).
* **Viewport Filter Box:** Use **Two Hands** (index finger and thumb on each hand) to create a boundary region for applying live visual filters.

### Keyboard Shortcuts
* `r` – Toggle video recording.
* `s` – Save current canvas artwork to disk.
* `q` – Quit application.

---

## Prerequisites

* Python 3.10+ (Tested on Python 3.12)
* Web camera
* [`uv`](https://docs.astral.sh/uv/) (Python package installer and runner)

---

## Installation & Usage

1. **Clone the repository:**
   ```bash
   git clone https://github.com/kekcodenoob/skytouch.git
   cd skytouch
   ```

2. **Install `uv` (if not already installed):**

   *macOS / Linux:*
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

   *Windows (PowerShell):*
   ```powershell
   powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
   ```

3. **Install dependencies from `uv.lock`:**
   ```bash
   uv sync
   ```

4. **Check that your camera works then run the application:**
   ```bash
   uv run main.py
   ```

---

## Future Improvements

* **On-Screen UI Toolbar:** Add a virtual button panel so users can select colors, shapes, filters, and brush sizes directly with their index finger instead of relying solely on gesture holds.
* **Custom Color Palette:** Allow users to pick custom RGB colors via a virtual color wheel.
* **Dynamic Eraser Mode:** Add a dedicated eraser gesture or mode to clear precise sections of the canvas without resetting the entire frame.
* **Multi-Hand Canvas Collaboration:** Support simultaneous drawing and shape manipulation for two hands or two separate users.
* **Image Upload and Editing:** Allow users to place their own images onto the canvas, add drawings and filters, then download the completed graphic.