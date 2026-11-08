# Skytouch

Skytouch is a real-time computer vision application that turns hand gestures into a digital canvas, image editor, and video filter interface using OpenCV and MediaPipe.

---

## Features & Controls

### Hand Gestures
* **Freehand Draw:** Raise **Index Finger** only.
* **Drag Bounding Shape:** **Pinch** (Thumb + Pinky Finger), drag, and release to place.
* **Drag Uploaded Image:** **OK Sign** (Thumb + Index touching) and drag across the screen (only while an image is loaded).
* **Scale Uploaded Image:** **OK Sign with both hands**. Move your hands closer together to shrink the image, or farther apart to enlarge it.
* **Cycle Color:** Hold **Index + Pinky** fingers up.
* **Cycle Shape Mode:** Hold **Index + Middle** fingers up (Peace sign).
* **Undo Action:** Hold **Pinky + Thumb** out (Shaka).
* **Clear Canvas:** Hold a closed **Fist** for ~1.5s (progress bar fills).
* **Viewport Filter Box:** Make an **"L"** (thumb + index) with **both hands** to frame a region. Each new frame cycles the live filter.
* **Idle:** An open palm does nothing. Held gestures fire once per hold, so release and hold again to repeat. A ring around your fingertip shows hold progress, and the current gesture is shown in the bottom-left.

### Keyboard Shortcuts
* `b` – Cycle brush style.
* `c` – Clear canvas.
* `r` – Toggle video recording.
* `u` – Upload image from computer for editing.
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
* **Multi-Hand Canvas Collaboration:** Support simultaneous drawing and shape manipulation for two hands or two separate users.
* **Canvas Toggle On/Off:** Allow for all or some drawing features to be turned on and off, giving users freedom to move around without messing up their canvas.