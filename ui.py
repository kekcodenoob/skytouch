"""Hover-to-click (dwell) UI: shared helpers, the on-screen toolbar and pop-up menus."""

import math

import cv2

DWELL_S = 0.6  # hover this long over a button to press it

FONT = cv2.FONT_HERSHEY_SIMPLEX
ACCENT = (0, 255, 255)
PANEL = (30, 30, 30)
BUTTON = (60, 60, 60)
TEXT = (235, 235, 235)

# Short toolbar / menu labels for long names
FILTER_SHORT = {"GRAYSCALE": "GRAY", "NIGHT_VISION": "NIGHT", "KALEIDOSCOPE": "KALEIDO"}


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

    BUTTONS = ["color", "shape", "brush", "filter", "eraser", "undo", "clear", "save", "record", "gallery", "pause"]

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
        """state: dict with color, shape, brush, last_brush, filter, recording, paused, menu."""
        x1, y1, x2, y2 = self.panel
        roi = frame[y1:y2, x1:x2]
        roi[:] = (roi * 0.35 + 0.65 * PANEL[0]).astype(roi.dtype)  # translucent panel

        labels = {
            "color": "COLOR",
            "shape": {"rectangle": "RECT", "triangle": "TRI"}.get(state["shape"], state["shape"].upper()),
            # While erasing, the brush button shows the brush you'll return to
            "brush": {"CALLIGRAPHY": "CALLIG."}.get(state["last_brush"], state["last_brush"]),
            "filter": FILTER_SHORT.get(state["filter"], state["filter"]),
            "eraser": "ERASER",
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
            elif name == "eraser" and state["brush"] == "ERASER":
                fill = (150, 90, 40)
            active = ((name == "pause" and state["paused"])
                      or (name == "eraser" and state["brush"] == "ERASER")
                      or name == state.get("menu"))  # its pop-up menu is open
            draw_button(
                frame,
                rect,
                labels[name],
                progress=self.progress if name == hovered else 0.0,
                hovered=name == hovered,
                fill=fill,
                active=active,
            )


class Picker:
    """Pop-up grid menu under the toolbar. Each tile shows a preview image.

    Hover a tile (dwell) or click it to choose it. CLOSE, clicking outside,
    or Esc dismisses the menu.
    """

    COLS = 4
    TILE_W, PREVIEW_H, LABEL_H = 220, 124, 28
    GAP, PAD, HEADER_H = 12, 14, 46

    def __init__(self, width, top):
        self.width = width
        self.top = top
        self.kind = None  # e.g. "brush" / "filter"; None = closed
        self.dwell = DwellTracker()
        self.progress = 0.0

    @property
    def active(self):
        return self.kind is not None

    def open(self, kind, title, items, labels=None):
        self.kind, self.title, self.items = kind, title, list(items)
        self.labels = labels or {}
        self.dwell = DwellTracker()
        self.progress = 0.0
        rows = math.ceil(len(self.items) / self.COLS)
        tile_h = self.PREVIEW_H + self.LABEL_H
        pw = self.COLS * self.TILE_W + (self.COLS - 1) * self.GAP + 2 * self.PAD
        ph = self.HEADER_H + rows * tile_h + (rows - 1) * self.GAP + self.PAD
        x0 = (self.width - pw) // 2
        self.panel = (x0, self.top, x0 + pw, self.top + ph)
        self.tiles = {}
        for i, name in enumerate(self.items):
            r, c = divmod(i, self.COLS)
            x = x0 + self.PAD + c * (self.TILE_W + self.GAP)
            y = self.top + self.HEADER_H + r * (tile_h + self.GAP)
            self.tiles[name] = (x, y, x + self.TILE_W, y + tile_h)
        self.close_rect = (x0 + pw - self.PAD - 110, self.top + 8, x0 + pw - self.PAD, self.top + self.HEADER_H - 8)

    def close(self):
        self.kind = None

    def contains(self, pt):
        return self.active and pt is not None and point_in(pt, self.panel)

    def hit(self, pt):
        if not self.contains(pt):
            return None
        if point_in(pt, self.close_rect):
            return "close"
        for name, rect in self.tiles.items():
            if point_in(pt, rect):
                return name
        return None

    def update(self, pt, now):
        """Feed the hand cursor (or None). Returns a chosen item, "close", or None."""
        chosen, self.progress = self.dwell.update(self.hit(pt), now)
        return chosen

    def click(self, pt):
        """Mouse click. Clicking outside the panel closes the menu."""
        return self.hit(pt) if self.contains(pt) else "close"

    def draw(self, frame, previews, current):
        x1, y1, x2, y2 = self.panel
        roi = frame[y1:y2, x1:x2]
        roi[:] = (roi * 0.15 + 0.85 * PANEL[0]).astype(roi.dtype)
        cv2.rectangle(frame, (x1, y1), (x2, y2), ACCENT, 1)
        cv2.putText(frame, self.title, (x1 + self.PAD, y1 + 31), FONT, 0.7, TEXT, 2, cv2.LINE_AA)
        hovered = self.dwell.target
        draw_button(frame, self.close_rect, "CLOSE",
                    progress=self.progress if hovered == "close" else 0.0, hovered=hovered == "close")

        for name, (tx1, ty1, tx2, ty2) in self.tiles.items():
            img = previews.get(name)
            py2 = ty1 + self.PREVIEW_H
            if img is not None:
                frame[ty1:py2, tx1:tx2] = cv2.resize(img, (tx2 - tx1, py2 - ty1))
            else:
                cv2.rectangle(frame, (tx1, ty1), (tx2, py2), BUTTON, -1)
            is_cur, is_hov = name == current, name == hovered
            draw_button(frame, (tx1, py2, tx2, ty2), self.labels.get(name, name), active=is_cur)
            border = ACCENT if (is_cur or is_hov) else (120, 120, 120)
            cv2.rectangle(frame, (tx1, ty1), (tx2, ty2), border, 3 if is_hov else (2 if is_cur else 1))
            if is_hov and self.progress > 0:
                cv2.rectangle(frame, (tx1, ty2 - 6), (tx1 + int((tx2 - tx1) * self.progress), ty2), ACCENT, -1)
