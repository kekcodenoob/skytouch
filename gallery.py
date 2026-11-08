"""In-app gallery of saved artwork and recordings.

Everything can be pressed by pointing and holding still (dwell) or by clicking
with the mouse.
"""

import os
from datetime import datetime

import cv2
import numpy as np

from media import is_video, list_media, reveal_in_file_manager
from ui import ACCENT, FONT, TEXT, DwellTracker, draw_button, draw_cursor, point_in

BG = (22, 22, 22)
COLS, ROWS = 4, 3


def _fit(img, box_w, box_h):
    """Resize img to fit inside box, keeping aspect ratio."""
    h, w = img.shape[:2]
    s = min(box_w / w, box_h / h)
    return cv2.resize(img, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA)


def _paste_centered(dst, img, rect):
    x1, y1, x2, y2 = rect
    fitted = _fit(img[:, :, :3], x2 - x1, y2 - y1)
    fh, fw = fitted.shape[:2]
    ox, oy = x1 + (x2 - x1 - fw) // 2, y1 + (y2 - y1 - fh) // 2
    dst[oy:oy + fh, ox:ox + fw] = fitted


def _play_icon(frame, center, r=22):
    cx, cy = center
    cv2.circle(frame, center, r, (0, 0, 0), -1, cv2.LINE_AA)
    cv2.circle(frame, center, r, TEXT, 2, cv2.LINE_AA)
    pts = np.array([[cx - r // 3, cy - r // 2], [cx - r // 3, cy + r // 2], [cx + r // 2, cy]], np.int32)
    cv2.fillPoly(frame, [pts], TEXT, cv2.LINE_AA)


def _fmt_time(seconds):
    seconds = max(0, int(seconds))
    return f"{seconds // 60}:{seconds % 60:02d}"


class Gallery:

    def __init__(self, width, height):
        self.w, self.h = width, height
        self.active = False
        self.items = []
        self.page = 0
        self.selected = None  # index into items when viewing one item
        self.dwell = DwellTracker()
        self.progress = 0.0
        self.cursor = None
        self.thumbs = {}

        # Video playback
        self.video = None
        self.video_frame = None
        self.video_playing = True
        self.video_frames = 0
        self.video_pos = 0
        self.frame_dt = 0.05
        self.next_frame_t = 0.0

        self._full_path = None
        self._full_img = None
        self._layout()

    # ---------------------------------------------------------------- layout
    def _layout(self):
        w, h = self.w, self.h
        bw, bh = 150, 48
        self.top_buttons = {
            "folder": (w - 2 * bw - 50, 24, w - bw - 50, 24 + bh),
            "close": (w - bw - 30, 24, w - 30, 24 + bh),
        }
        by = h - bh - 24
        self.grid_buttons = {
            "prev": (w // 2 - bw - 20, by, w // 2 - 20, by + bh),
            "next": (w // 2 + 20, by, w // 2 + bw + 20, by + bh),
        }
        # Viewer: BACK | < PREV | PLAY/EDIT | NEXT > | SHOW FILE
        gap = 24
        total = 5 * bw + 4 * gap
        x = (w - total) // 2
        self.view_buttons = {}
        for key in ("back", "prev", "action", "next", "show"):
            self.view_buttons[key] = (x, by, x + bw, by + bh)
            x += bw + gap
        self.seek_rect = (60, by - 44, w - 60, by - 20)

        gap_x, gap_y, caption = 24, 18, 26
        grid_top, grid_bottom = 96, by - 20
        cell_w = (w - 2 * 60 - (COLS - 1) * gap_x) // COLS
        cell_h = min(int(cell_w * 9 / 16), (grid_bottom - grid_top - (ROWS - 1) * gap_y) // ROWS - caption)
        x0 = (w - (COLS * cell_w + (COLS - 1) * gap_x)) // 2
        self.cells = []
        for r in range(ROWS):
            for c in range(COLS):
                x = x0 + c * (cell_w + gap_x)
                y = grid_top + r * (cell_h + caption + gap_y)
                self.cells.append((x, y, x + cell_w, y + cell_h))
        self.view_rect = (40, 96, w - 40, by - 60)

    @property
    def per_page(self):
        return COLS * ROWS

    @property
    def page_count(self):
        return max(1, (len(self.items) + self.per_page - 1) // self.per_page)

    @property
    def viewing_video(self):
        return self.video is not None

    # ------------------------------------------------------------ open/close
    def open(self):
        self.items = list_media()
        self.page = 0
        self.selected = None
        self.active = True
        self.dwell = DwellTracker()

    def close(self):
        self._stop_video()
        self.active = False
        self.selected = None

    def _view(self, idx):
        self._stop_video()
        self.selected = idx % len(self.items)
        path = self.items[self.selected]
        if is_video(path):
            self.video = cv2.VideoCapture(path)
            fps = self.video.get(cv2.CAP_PROP_FPS) or 20.0
            self.frame_dt = 1.0 / max(fps, 1.0)
            self.video_frames = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
            self.video_pos = 0
            self.video_playing = True
            self.next_frame_t = 0.0
            self.video_frame = None

    def _stop_video(self):
        if self.video is not None:
            self.video.release()
        self.video = None
        self.video_frame = None

    def _seek(self, x):
        x1, _, x2, _ = self.seek_rect
        frac = min(1.0, max(0.0, (x - x1) / (x2 - x1)))
        self.video_pos = int(frac * (self.video_frames - 1))
        self.video.set(cv2.CAP_PROP_POS_FRAMES, self.video_pos)
        ok, img = self.video.read()
        if ok:
            self.video_frame = img
            self.video_pos += 1

    # ----------------------------------------------------------------- input
    def _targets(self):
        t = dict(self.top_buttons)
        if self.selected is None:
            if self.items:
                t.update(self.grid_buttons)
            start = self.page * self.per_page
            for i, rect in enumerate(self.cells):
                if start + i < len(self.items):
                    t[("item", start + i)] = rect
        else:
            t.update(self.view_buttons)
            if self.viewing_video:
                t["seek"] = self.seek_rect
        return t

    def _target_at(self, pt):
        if pt is None:
            return None
        for key, rect in self._targets().items():
            if point_in(pt, rect):
                return key
        return None

    def update(self, cursor, now):
        """Feed the fingertip (or None). Returns the result of a press, if any."""
        self.cursor = cursor
        pressed, self.progress = self.dwell.update(self._target_at(cursor), now)
        return self._press(pressed, cursor) if pressed is not None else None

    def click(self, pt):
        """Mouse click at pt. Returns the result of the press, if any."""
        key = self._target_at(pt)
        return self._press(key, pt) if key is not None else None

    def _press(self, key, pt=None):
        """Returns "close", ("edit", path) or None."""
        if key == "close":
            self.close()
            return "close"
        if key == "folder":
            reveal_in_file_manager()
        elif isinstance(key, tuple):
            self._view(key[1])
        elif self.selected is None:
            if key == "prev":
                self.page = (self.page - 1) % self.page_count
            elif key == "next":
                self.page = (self.page + 1) % self.page_count
        else:
            path = self.items[self.selected]
            if key == "back":
                self._stop_video()
                self.page = self.selected // self.per_page
                self.selected = None
            elif key == "prev":
                self._view(self.selected - 1)
            elif key == "next":
                self._view(self.selected + 1)
            elif key == "show":
                reveal_in_file_manager(path)
            elif key == "seek" and pt is not None:
                self._seek(pt[0])
            elif key == "action":
                if self.viewing_video:
                    self.video_playing = not self.video_playing
                else:
                    self.close()
                    return ("edit", path)
        return None

    def key(self, k):
        """Keyboard: a/d = previous/next, space = play/pause, e = edit, Esc = back, g = close."""
        if k == 27 and self.selected is not None:
            return self._press("back")
        if k in (27, ord("g")):
            return self._press("close")
        if k == ord("a"):
            return self._press("prev")
        if k == ord("d"):
            return self._press("next")
        if self.selected is not None and (
            (k == ord(" ") and self.viewing_video) or (k == ord("e") and not self.viewing_video)
        ):
            return self._press("action")
        return None

    # ---------------------------------------------------------------- render
    def _thumb(self, path):
        if path not in self.thumbs:
            img = None
            if is_video(path):
                cap = cv2.VideoCapture(path)
                ok, img = cap.read()
                cap.release()
                img = img if ok else None
            else:
                img = cv2.imread(path)
            if img is None:
                img = np.full((90, 160, 3), 50, np.uint8)
            self.thumbs[path] = _fit(img, 320, 180)
        return self.thumbs[path]

    def render(self, now):
        frame = np.full((self.h, self.w, 3), BG, np.uint8)
        title = "GALLERY" if self.selected is None else os.path.basename(self.items[self.selected])
        cv2.putText(frame, title, (40, 60), FONT, 1.0 if self.selected is None else 0.7, ACCENT, 2, cv2.LINE_AA)

        buttons = dict(self.top_buttons)
        labels = {"folder": "OPEN FOLDER", "close": "CLOSE", "prev": "< PREV", "next": "NEXT >",
                  "back": "BACK", "show": "SHOW FILE"}

        if self.selected is None:
            self._render_grid(frame)
            if self.items:
                buttons.update(self.grid_buttons)
        else:
            self._render_item(frame, now)
            buttons.update(self.view_buttons)
            if self.viewing_video:
                labels["action"] = "PAUSE" if self.video_playing else "PLAY"
            else:
                labels["action"] = "EDIT"

        for key, rect in buttons.items():
            hov = self.dwell.target == key
            draw_button(frame, rect, labels[key], self.progress if hov else 0.0, hov, scale=0.55)

        if self.cursor is not None:
            draw_cursor(frame, self.cursor, self.progress)
        return frame

    def _render_grid(self, frame):
        if not self.items:
            msg = "No saves yet - press S or hover SAVE on the toolbar."
            (tw, _), _ = cv2.getTextSize(msg, FONT, 0.8, 2)
            cv2.putText(frame, msg, ((self.w - tw) // 2, self.h // 2), FONT, 0.8, TEXT, 2, cv2.LINE_AA)
            return

        info = f"{len(self.items)} saves  |  page {self.page + 1}/{self.page_count}  |  point & hold or click to open"
        cv2.putText(frame, info, (200, 58), FONT, 0.55, (170, 170, 170), 1, cv2.LINE_AA)

        start = self.page * self.per_page
        for i, rect in enumerate(self.cells):
            idx = start + i
            if idx >= len(self.items):
                break
            path = self.items[idx]
            x1, y1, x2, y2 = rect
            cv2.rectangle(frame, (x1, y1), (x2, y2), (45, 45, 45), -1)
            _paste_centered(frame, self._thumb(path), rect)
            if is_video(path):
                _play_icon(frame, ((x1 + x2) // 2, (y1 + y2) // 2))

            hov = self.dwell.target == ("item", idx)
            cv2.rectangle(frame, (x1, y1), (x2, y2), ACCENT if hov else (90, 90, 90), 2 if hov else 1)
            if hov and self.progress > 0:
                cv2.rectangle(frame, (x1, y2 - 6), (x1 + int((x2 - x1) * self.progress), y2), ACCENT, -1)

            stamp = datetime.fromtimestamp(os.path.getmtime(path)).strftime("%b %d, %H:%M")
            kind = "VIDEO" if is_video(path) else "IMAGE"
            cv2.putText(frame, f"{stamp}  {kind}", (x1, y2 + 19), FONT, 0.45, (190, 190, 190), 1, cv2.LINE_AA)

    def _render_item(self, frame, now):
        rect = self.view_rect
        if self.viewing_video:
            if self.video_frame is None or (self.video_playing and now >= self.next_frame_t):
                ok, img = self.video.read()
                if not ok:  # loop back to the start
                    self.video.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    self.video_pos = 0
                    ok, img = self.video.read()
                if ok:
                    self.video_frame = img
                    self.video_pos += 1
                self.next_frame_t = now + self.frame_dt
            img = self.video_frame
            self._render_seek(frame)
        else:
            img = self._full_image(self.items[self.selected])
            hint = "EDIT loads this image back onto the canvas so you can keep drawing"
            cv2.putText(frame, hint, (rect[0], self.seek_rect[3]), FONT, 0.5, (150, 150, 150), 1, cv2.LINE_AA)
        if img is not None:
            _paste_centered(frame, img, rect)
        counter = f"{self.selected + 1} / {len(self.items)}"
        cv2.putText(frame, counter, (self.w // 2 - 30, rect[1] - 14), FONT, 0.55, (170, 170, 170), 1, cv2.LINE_AA)

    def _render_seek(self, frame):
        x1, y1, x2, y2 = self.seek_rect
        frac = min(1.0, self.video_pos / max(1, self.video_frames))
        hov = self.dwell.target == "seek"
        cv2.rectangle(frame, (x1, y1 + 8), (x2, y2 - 8), (70, 70, 70), -1)
        cv2.rectangle(frame, (x1, y1 + 8), (x1 + int((x2 - x1) * frac), y2 - 8), ACCENT, -1)
        cv2.circle(frame, (x1 + int((x2 - x1) * frac), (y1 + y2) // 2), 9, TEXT, -1, cv2.LINE_AA)
        if hov and self.cursor is not None:  # show where a seek would land
            cv2.line(frame, (self.cursor[0], y1), (self.cursor[0], y2), TEXT, 2)
        t = f"{_fmt_time(self.video_pos * self.frame_dt)} / {_fmt_time(self.video_frames * self.frame_dt)}"
        cv2.putText(frame, t, (x1, y1 - 6), FONT, 0.5, (190, 190, 190), 1, cv2.LINE_AA)

    def _full_image(self, path):
        if self._full_path != path:
            self._full_path = path
            self._full_img = cv2.imread(path)
        return self._full_img
