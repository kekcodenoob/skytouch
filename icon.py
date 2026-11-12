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


# A pointing hand in MediaPipe's 21-landmark layout (index up, other fingers folded),
# drawn on a 1024px canvas.
HAND = {0: (540, 910), 1: (430, 845), 2: (360, 755), 3: (330, 665), 4: (345, 585),
        5: (460, 640), 6: (450, 500), 7: (445, 390), 8: (440, 280),
        9: (555, 650), 10: (575, 560), 11: (550, 625), 12: (535, 680),
        13: (640, 670), 14: (660, 590), 15: (635, 650), 16: (620, 700),
        17: (715, 715), 18: (735, 650), 19: (712, 700), 20: (695, 735)}
BONES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11), (11, 12),
         (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17)]


def _hue(h):
    hsv = np.uint8([[[int(h) % 180, 255, 255]]])
    return tuple(int(c) for c in cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0])


def _glow(img, layer, sigma, gain):
    halo = cv2.GaussianBlur(layer, (0, 0), sigma)
    img = cv2.add(img, (halo * gain).clip(0, 255).astype(np.uint8))
    return cv2.add(img, layer)


def make_icon(size=256):
    """Returns a BGRA icon: a hand-tracking skeleton whose fingertip leaves a
    glowing rainbow trail, on a dark rounded square."""
    S = 1024  # draw large, downsample for smooth edges
    yy, xx = np.mgrid[0:S, 0:S] / S

    # Background: diagonal gradient deep blue -> plum, soft glow upper left
    top = np.array([60, 25, 10], float)  # BGR
    bot = np.array([30, 10, 30], float)
    t = ((xx + yy) / 2)[..., None]
    bg = top * (1 - t) + bot * t
    glow = np.exp(-(((xx - 0.3) ** 2 + (yy - 0.25) ** 2) / 0.1))[..., None]
    img = np.clip(bg + glow * np.array([90, 60, 20]), 0, 255).astype(np.uint8)

    # Rounded-square mask
    mask = np.zeros((S, S), np.uint8)
    r, m = int(S * 0.22), int(S * 0.04)
    cv2.rectangle(mask, (m + r, m), (S - m - r, S - m), 255, -1)
    cv2.rectangle(mask, (m, m + r), (S - m, S - m - r), 255, -1)
    for cx, cy in [(m + r, m + r), (S - m - r, m + r), (m + r, S - m - r), (S - m - r, S - m - r)]:
        cv2.circle(mask, (cx, cy), r, 255, -1, cv2.LINE_AA)

    # Rainbow trail off the fingertip, tapering as it goes
    tip = HAND[8]
    pts = _bezier(np.array(tip, float), np.array([600, 110.0]), np.array([820, 330.0]), np.array([900, 150.0]))
    trail = np.zeros((S, S, 3), np.uint8)
    n = len(pts) - 1
    for i in range(n):
        f = i / n
        cv2.line(trail, tuple(pts[i].astype(int)), tuple(pts[i + 1].astype(int)),
                 _hue(90 + 90 * f), int(60 - 34 * f), cv2.LINE_AA)
    img = _glow(img, trail, 28, 1.6)

    # Hand skeleton: cream bones, pink joints, bright fingertip
    hand = np.zeros((S, S, 3), np.uint8)
    for a, b in BONES:
        cv2.line(hand, HAND[a], HAND[b], (255, 230, 200), 28, cv2.LINE_AA)
    for k, p in HAND.items():
        if k != 8:
            cv2.circle(hand, p, 21, (120, 80, 255), -1, cv2.LINE_AA)
    cv2.circle(hand, tip, 44, (255, 255, 255), -1, cv2.LINE_AA)
    img = _glow(img, hand, 14, 0.8)

    # Fingertip ring
    ring = np.zeros_like(img)
    cv2.circle(ring, tip, 84, (255, 230, 200), 14, cv2.LINE_AA)
    img = _glow(img, ring, 10, 1.0)

    _star(img, (830, 540), 50, (255, 255, 255), 9)

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
