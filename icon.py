"""Skytouch app icon: drawn procedurally, saved as .ico/.png, applied to the window.

Run `uv run icon.py` to regenerate the files in assets/.
"""

import ctypes
import os
import struct
import sys

import cv2
import numpy as np

ASSETS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
ICO_PATH = os.path.join(ASSETS, "skytouch.ico")
PNG_PATH = os.path.join(ASSETS, "skytouch.png")


def _bezier(p0, p1, p2, p3, n=200):
    t = np.linspace(0, 1, n)[:, None]
    return (1 - t) ** 3 * p0 + 3 * (1 - t) ** 2 * t * p1 + 3 * (1 - t) * t ** 2 * p2 + t ** 3 * p3


def _star(img, c, r, color, thick):
    x, y = c
    cv2.line(img, (x - r, y), (x + r, y), color, thick, cv2.LINE_AA)
    cv2.line(img, (x, y - r), (x, y + r), color, thick, cv2.LINE_AA)
    s = r // 3
    cv2.line(img, (x - s, y - s), (x + s, y + s), color, max(1, thick // 2), cv2.LINE_AA)
    cv2.line(img, (x - s, y + s), (x + s, y - s), color, max(1, thick // 2), cv2.LINE_AA)


def make_icon(size=256):
    """Returns a BGRA icon: glowing gradient brush stroke ending at a fingertip,
    on a dark rounded square."""
    S = 1024  # draw large, downsample for smooth edges
    yy, xx = np.mgrid[0:S, 0:S] / S

    # Background: diagonal gradient deep purple -> midnight blue
    top = np.array([70, 18, 46], float)     # BGR
    bot = np.array([48, 20, 12], float)
    t = ((xx + yy) / 2)[..., None]
    bg = (top * (1 - t) + bot * t)
    # soft vignette glow in the upper right
    glow = np.exp(-(((xx - 0.72) ** 2 + (yy - 0.28) ** 2) / 0.08))[..., None]
    bg = np.clip(bg + glow * np.array([90, 40, 70]), 0, 255).astype(np.uint8)

    # Rounded-square mask
    mask = np.zeros((S, S), np.uint8)
    r, m = int(S * 0.22), int(S * 0.04)
    cv2.rectangle(mask, (m + r, m), (S - m - r, S - m), 255, -1)
    cv2.rectangle(mask, (m, m + r), (S - m, S - m - r), 255, -1)
    for cx, cy in [(m + r, m + r), (S - m - r, m + r), (m + r, S - m - r), (S - m - r, S - m - r)]:
        cv2.circle(mask, (cx, cy), r, 255, -1, cv2.LINE_AA)

    # The stroke: a swoosh from lower-left up to a fingertip in the upper-right
    pts = _bezier(np.array([200, 800.0]), np.array([330, 420.0]),
                  np.array([620, 860.0]), np.array([760, 300.0]))
    cyan, magenta = np.array([255, 230, 40.0]), np.array([230, 60, 255.0])
    stroke = np.zeros((S, S, 3), np.uint8)
    for i in range(len(pts) - 1):
        f = i / (len(pts) - 1)
        col = tuple(int(c) for c in magenta * (1 - f) + cyan * f)
        w = int(40 + 50 * np.sin(np.pi * f) ** 0.8)  # thicker in the middle
        cv2.line(stroke, tuple(pts[i].astype(int)), tuple(pts[i + 1].astype(int)), col, w, cv2.LINE_AA)

    halo = cv2.GaussianBlur(stroke, (0, 0), 38)
    img = cv2.add(bg, (halo * 1.6).clip(0, 255).astype(np.uint8))
    img = cv2.add(img, stroke)

    # Fingertip: bright dot with a ring
    tip = (760, 300)
    ring = np.zeros_like(img)
    cv2.circle(ring, tip, 95, (255, 240, 120), 18, cv2.LINE_AA)
    img = cv2.add(img, cv2.GaussianBlur(ring, (0, 0), 12))
    img = cv2.add(img, ring)
    cv2.circle(img, tip, 52, (255, 255, 255), -1, cv2.LINE_AA)

    # Sparkles
    _star(img, (330, 250), 60, (255, 255, 255), 10)
    _star(img, (840, 690), 38, (255, 220, 255), 7)

    bgra = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    bgra[:, :, 3] = mask
    return cv2.resize(bgra, (size, size), interpolation=cv2.INTER_AREA)


def write_ico(path, sizes=(256, 64, 48, 32, 16)):
    """Writes a multi-size .ico (PNG-compressed entries, Windows Vista+)."""
    base = make_icon(256)
    blobs = []
    for s in sizes:
        im = base if s == 256 else cv2.resize(base, (s, s), interpolation=cv2.INTER_AREA)
        blobs.append(cv2.imencode(".png", im)[1].tobytes())
    header = struct.pack("<HHH", 0, 1, len(sizes))
    offset = 6 + 16 * len(sizes)
    entries = b""
    for s, blob in zip(sizes, blobs):
        entries += struct.pack("<BBBBHHII", s % 256, s % 256, 0, 0, 1, 32, len(blob), offset)
        offset += len(blob)
    with open(path, "wb") as f:
        f.write(header + entries + b"".join(blobs))


def ensure_icon_files():
    os.makedirs(ASSETS, exist_ok=True)
    if not os.path.exists(ICO_PATH):
        write_ico(ICO_PATH)
    if not os.path.exists(PNG_PATH):
        cv2.imwrite(PNG_PATH, make_icon(256))


def set_app_id():
    """Give the app its own taskbar identity (so it doesn't group under python.exe).
    Must run before the window is created."""
    if sys.platform.startswith("win"):
        try:
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("Skytouch.App")
        except (AttributeError, OSError):
            pass


def apply_window_icon(window_title):
    """Sets the title-bar and taskbar icon of an OpenCV window (Windows only)."""
    if not sys.platform.startswith("win"):
        return False
    try:
        ensure_icon_files()
        user32 = ctypes.windll.user32
        user32.FindWindowW.restype = ctypes.c_void_p
        user32.LoadImageW.restype = ctypes.c_void_p
        user32.LoadImageW.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p, ctypes.c_uint,
                                      ctypes.c_int, ctypes.c_int, ctypes.c_uint]
        user32.SendMessageW.argtypes = [ctypes.c_void_p, ctypes.c_uint, ctypes.c_void_p, ctypes.c_void_p]

        hwnd = user32.FindWindowW(None, window_title)
        if not hwnd:
            return False
        IMAGE_ICON, LR_LOADFROMFILE, WM_SETICON = 1, 0x10, 0x80
        for which, metric in ((0, 49), (1, 11)):  # ICON_SMALL/SM_CXSMICON, ICON_BIG/SM_CXICON
            px = user32.GetSystemMetrics(metric)
            hicon = user32.LoadImageW(None, ICO_PATH, IMAGE_ICON, px, px, LR_LOADFROMFILE)
            if hicon:
                user32.SendMessageW(hwnd, WM_SETICON, which, hicon)
        return True
    except (AttributeError, OSError) as e:
        print(f"Could not set window icon: {e}")
        return False


if __name__ == "__main__":
    os.makedirs(ASSETS, exist_ok=True)
    write_ico(ICO_PATH)
    cv2.imwrite(PNG_PATH, make_icon(256))
    print(f"Wrote {ICO_PATH} and {PNG_PATH}")
