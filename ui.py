"""Hover-to-click (dwell) UI: shared helpers and the on-screen toolbar."""

import cv2

DWELL_S = 0.6  # hover this long over a button to press it

FONT = cv2.FONT_HERSHEY_SIMPLEX
ACCENT = (0, 255, 255)
PANEL = (30, 30, 30)
BUTTON = (60, 60, 60)
TEXT = (235, 235, 235)


class DwellTracker:
    """Fires a target once after the cursor has rested on it for DWELL_S.

    Moving off a target resets it, so a target fires once per hover.
    """

    def __init__(self, dwell_s=DWELL_S):
        self.dwell_s = dwell_s
        self.target = None
        self.since = 0.0
        self.fired = False

    def update(self, target, now):
        """Returns (fired_target_or_None, progress 0..1 for the hovered target)."""
        if target != self.target:
            self.target, self.since, self.fired = target, now, False
        if target is None or self.fired:
            return None, 0.0
        frac = min(1.0, (now - self.since) / self.dwell_s)
        if frac >= 1.0:
            self.fired = True
            return target, 1.0
        return None, frac


def point_in(pt, rect):
    x, y = pt
    x1, y1, x2, y2 = rect
    return x1 <= x <= x2 and y1 <= y <= y2


def draw_button(frame, rect, label, progress=0.0, hovered=False, fill=None, active=False, scale=0.42):
    """Draws a rounded-ish button with a dwell progress bar along its bottom edge."""
    x1, y1, x2, y2 = rect
    cv2.rectangle(frame, (x1, y1), (x2, y2), fill if fill is not None else BUTTON, -1)
    border = ACCENT if (hovered or active) else (120, 120, 120)
    cv2.rectangle(frame, (x1, y1), (x2, y2), border, 2 if (hovered or active) else 1)
    if progress > 0:
        cv2.rectangle(frame, (x1, y2 - 5), (x1 + int((x2 - x1) * progress), y2), ACCENT, -1)
    if label:
        (tw, th), _ = cv2.getTextSize(label, FONT, scale, 1)
        tx = x1 + (x2 - x1 - tw) // 2
        ty = y1 + (y2 - y1 + th) // 2
        # dark outline so labels stay readable on bright color swatches
        cv2.putText(frame, label, (tx, ty), FONT, scale, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, label, (tx, ty), FONT, scale, TEXT, 1, cv2.LINE_AA)


def draw_cursor(frame, pt, progress=0.0):
    cv2.circle(frame, pt, 8, ACCENT, 2, cv2.LINE_AA)
    if progress > 0:
        cv2.ellipse(frame, pt, (18, 18), -90, 0, 360 * progress, ACCENT, 3, cv2.LINE_AA)


class Toolbar:
    """Button strip pressed by hovering a pointing finger.

    position="top" puts a horizontal strip along the top edge; "right" puts a
    vertical strip down the right edge.
    """

    BUTTONS = ["color", "shape", "brush", "undo", "clear", "save", "record", "gallery", "pause"]

    def __init__(self, width, height, position="top", btn_w=100, btn_h=52, gap=8, pad=8):
        self.position = position
        self.rects = {}
        n = len(self.BUTTONS)
        if position == "top":
            total = n * btn_w + (n - 1) * gap
            x = (width - total) // 2
            y = pad + 4
            for name in self.BUTTONS:
                self.rects[name] = (x, y, x + btn_w, y + btn_h)
                x += btn_w + gap
            first, last = self.rects[self.BUTTONS[0]], self.rects[self.BUTTONS[-1]]
            self.panel = (first[0] - pad, 4, last[2] + pad, y + btn_h + pad)
        else:
            x1, x2 = width - 92 - pad, width - pad
            y = 70
            for name in self.BUTTONS:
                self.rects[name] = (x1 + pad, y, x2 - pad, y + btn_h + 4)
                y += btn_h + 4 + gap
            self.panel = (x1, 70 - pad, x2, y)
        # How far the HUD/toasts should shift down so they don't sit under the bar
        self.hud_offset = self.panel[3] + 4 if position == "top" else 0
        self.dwell = DwellTracker()
        self.progress = 0.0

    def contains(self, pt, margin=0):
        if pt is None:
            return False
        x1, y1, x2, y2 = self.panel
        # Extend to the screen edge the bar sits on, so a fingertip that
        # overshoots past the edge still counts as "on the toolbar".
        if self.position == "top":
            y1 = -10**6
        else:
            x2 = 10**6
        return point_in(pt, (x1 - margin, y1 - margin, x2 + margin, y2 + margin))

    def hit(self, pt):
        if pt is None:
            return None
        for name, rect in self.rects.items():
            if point_in(pt, rect):
                return name
        return None

    def update(self, pt, now):
        """Feed the cursor (or None). Returns the pressed button name, if any."""
        pressed, self.progress = self.dwell.update(self.hit(pt), now)
        return pressed

    def draw(self, frame, state):
        """state: dict with color, shape, brush, recording, paused."""
        x1, y1, x2, y2 = self.panel
        roi = frame[y1:y2, x1:x2]
        roi[:] = (roi * 0.35 + 0.65 * PANEL[0]).astype(roi.dtype)  # translucent panel

        labels = {
            "color": "COLOR",
            "shape": {"rectangle": "RECT", "triangle": "TRI"}.get(state["shape"], state["shape"].upper()),
            "brush": {"CALLIGRAPHY": "CALLIG."}.get(state["brush"], state["brush"]),
            "undo": "UNDO",
            "clear": "CLEAR",
            "save": "SAVE",
            "record": "STOP" if state["recording"] else "REC",
            "gallery": "GALLERY",
            "pause": "RESUME" if state["paused"] else "PAUSE",
        }
        hovered = self.dwell.target
        for name, rect in self.rects.items():
            fill = None
            if name == "color":
                fill = state["color"]
            elif name == "record" and state["recording"]:
                fill = (0, 0, 170)
            elif name == "pause" and state["paused"]:
                fill = (0, 110, 160)
            draw_button(
                frame,
                rect,
                labels[name],
                progress=self.progress if name == hovered else 0.0,
                hovered=name == hovered,
                fill=fill,
                active=(name == "pause" and state["paused"]),
            )
