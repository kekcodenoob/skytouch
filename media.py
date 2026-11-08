"""Save locations and helpers for Skytouch artwork and recordings."""

import os
import subprocess
import sys
from datetime import datetime

SAVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "saves")
IMAGE_EXTS = (".png", ".jpg", ".jpeg")
VIDEO_EXTS = (".mp4",)


def new_media_path(ext):
    """Returns a fresh timestamped path like saves/skytouch_20261107_153210.png."""
    os.makedirs(SAVE_DIR, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(SAVE_DIR, f"skytouch_{stamp}{ext}")
    n = 2
    while os.path.exists(path):  # two saves in the same second
        path = os.path.join(SAVE_DIR, f"skytouch_{stamp}_{n}{ext}")
        n += 1
    return path


def list_media():
    """All saved images and videos, newest first."""
    if not os.path.isdir(SAVE_DIR):
        return []
    files = [
        os.path.join(SAVE_DIR, f)
        for f in os.listdir(SAVE_DIR)
        if f.lower().endswith(IMAGE_EXTS + VIDEO_EXTS)
    ]
    return sorted(files, key=os.path.getmtime, reverse=True)


def is_video(path):
    return path.lower().endswith(VIDEO_EXTS)


def reveal_in_file_manager(path=None):
    """Opens the saves folder (selecting `path` if given) so files can be shared."""
    os.makedirs(SAVE_DIR, exist_ok=True)
    try:
        if sys.platform.startswith("win"):
            if path:
                subprocess.Popen(["explorer", "/select,", os.path.normpath(path)])
            else:
                os.startfile(SAVE_DIR)
        elif sys.platform == "darwin":
            subprocess.Popen(["open", "-R", path] if path else ["open", SAVE_DIR])
        else:
            subprocess.Popen(["xdg-open", os.path.dirname(path) if path else SAVE_DIR])
    except OSError as e:
        print(f"Could not open file manager: {e}")
