import time
from collections import deque

import cv2
import mediapipe as mp
import numpy as np

from canvas import CanvasManager
from gallery import Gallery
from gestures import HL, HandTracker
from icon import apply_window_icon, set_app_id
from canvas import BRUSH_MODES
from ui import ACCENT, FILTER_SHORT, Picker, Toolbar, draw_cursor

# --- Timing (seconds) -------------------------------------------------------
HAND_COUNT_CONFIRM_S = 0.15  # hand count must be stable this long before mode changes
CLEAR_HOLD_S = 1.5  # hold fist to clear
UNDO_HOLD_S = 0.35  # hold shaka to undo
CYCLE_HOLD_S = 0.40  # hold rock to cycle color
FRAME_HOLD_S = 0.30  # hold two L-shapes to start a filter box
FILTER_PREVIEWS_PER_FRAME = 4  # live filter-menu previews refreshed per frame (round-robin)
FRAME_EXIT_GRACE_S = 0.40  # frame must be gone this long before it can re-trigger
ACTION_COOLDOWN_S = 0.60  # minimum gap between any two discrete actions

# --- Drawing ----------------------------------------------------------------
DRAW_LAG_FRAMES = 3  # points held back before committing; dropped when a stroke ends
MAX_JUMP_FRAC = 0.15  # break the line if the tip jumps more than this * width
DRAW_POSES = ("point", "l_shape")
WINDOW = "Skytouch"
TOOLBAR_MARGIN = 16  # px around the toolbar where drawing is also blocked
MIN_SHAPE_PX = 30  # shapes dragged shorter than this are discarded...
MIN_SHAPE_PALM = 0.4  # ...or shorter than this * palm size, whichever is larger


class SkytouchApp:

    def __init__(self, width=1280, height=720):
        self.width = width
        self.height = height

        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7,
        )

        self.tracker = HandTracker()
        self.canvas_mgr = CanvasManager(width, height)

        self.colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255)]
        self.color_names = ["Red", "Green", "Blue", "Yellow"]
        self.color_index = 0

        self.shape_modes = ["rectangle", "ellipse", "line", "circle", "triangle"]
        self.shape_index = 0

        # Hand-count debounce
        self.raw_count = 0
        self.raw_count_since = 0.0
        self.hand_count = 0
        self.primary_id = None

        # Hold-action bookkeeping
        self.fired_for = None  # (hand_id, pose_since) of the hold that already fired
        self.last_action_time = -1e9

        # Freehand stroke
        self.stroke_active = False
        self.stroke_last = None
        self.stroke_pending = deque()
        self.stroke_saved = False

        # Thumb + index pinch shape drag
        self.pinch_start = None
        self.pinch_last = None

        # Two-hand filter frame
        self.active_filter_index = 0
        self.filter_chosen = True  # next filter frame uses active_filter_index as-is (no cycling)
        self.frame_active = False
        self.frame_lost_since = None

        # Overlay manipulation
        self.prev_grab_pos = None
        self.prev_grab_dist = None

        # Per-frame UI state
        self.clear_progress = 0.0
        self.pose_label = "-"

        # Toolbar, gallery, pause
        self.toolbar = Toolbar(width, height)
        self.show_toolbar = True
        self.gallery = Gallery(width, height)
        # Pop-up brush / filter menus under the toolbar
        self.picker = Picker(width, top=self.toolbar.hud_offset + 6)
        self.brush_previews = {}  # cached per color
        self.brush_preview_color = None
        self.filter_previews = {}
        self.frame_no = 0
        self.paused = False
        self.fullscreen = False

        # Measured loop rate, so recordings play back at real speed
        self.fps = 20.0
        self.prev_t = None

    def print_terminal_instructions(self):
        print("""
===================================================================
                    SKYTOUCH CONTROL INSTRUCTIONS
===================================================================

--- HAND GESTURES ---
  [1] Freehand Draw        : Raise Index Finger only
  [2] Drag Bounding Shape  : Pinch Thumb + Index, drag, release to place
  [3] Move Overlay Image   : Index + Middle up (Peace Sign), drag (image loaded)
  [4] Resize Overlay Image : Peace Sign on BOTH hands, spread/close
  [5] Cycle Color          : Hold Index + Pinky up
  [6] Undo Action          : Hold Pinky + Thumb out (Shaka)
  [7] Clear Canvas         : Hold a closed Fist (~1.5s, bar fills)
  [8] Viewport Filter Box  : Make an "L" (thumb + index) with BOTH hands
  Open palm = idle, nothing happens. Held gestures fire once per hold.

--- TOOLBAR (top edge) ---
  Point at a button and hold still (~0.6s) to press it. Drawing is
  disabled while your fingertip is over the toolbar.

--- KEYBOARD SHORTCUTS ---
  [p] Pause / Resume Canvas (gestures off, toolbar still works)
  [g] Open / Close Gallery  (a / d = prev / next, space = play/pause,
                             e = edit image, Esc = back; mouse clicks work too)
  [t] Show / Hide Toolbar
  [f] Toggle Fullscreen     (Esc also leaves fullscreen)
  [b] Cycle Brush Style     (toolbar SHAPE / BRUSH / FILTER open preview menus)
  [e] Toggle Eraser
  [c] Clear Canvas
  [u] Upload/Prompt Background Overlay Image
  [r] Toggle Video Recording (saves/*.mp4)
  [s] Save Canvas Artwork    (saves/*.png)
  [q] Quit Application
===================================================================
""")

    # ------------------------------------------------------------------ actions
    def _do(self, action):
        """Runs a named action. Shared by the toolbar and keyboard shortcuts."""
        if action == "color":
            self.color_index = (self.color_index + 1) % len(self.colors)
            self.canvas_mgr.show_toast(f"COLOR: {self.color_names[self.color_index].upper()}")
        elif action == "shape":  # toolbar: open the shape menu
            self._toggle_menu("shape")
        elif action == "brush":  # toolbar: open the brush menu
            self._toggle_menu("brush")
        elif action == "filter":  # toolbar: open the filter menu
            self._toggle_menu("filter")
        elif action == "next_brush":  # `b` key: quick cycle
            self._end_stroke()
            self.canvas_mgr.cycle_brush_mode()
        elif action == "eraser":
            self._end_stroke()
            self.canvas_mgr.toggle_eraser()
        elif action == "undo":
            self._end_stroke()
            self.canvas_mgr.undo()
        elif action == "clear":
            self._end_stroke()
            self._cancel_pinch()
            self.canvas_mgr.clear()
            self.canvas_mgr.show_toast("CANVAS CLEARED")
        elif action == "save":
            self.canvas_mgr.save_image()
        elif action == "record":
            self.canvas_mgr.toggle_recording(fps=round(min(max(self.fps, 5.0), 60.0)))
        elif action == "gallery":
            self._end_stroke()
            self._cancel_pinch()
            self.picker.close()
            self.gallery.open()
        elif action == "pause":
            self.paused = not self.paused
            self._end_stroke()
            self._cancel_pinch()
            self.canvas_mgr.show_toast("CANVAS PAUSED" if self.paused else "CANVAS RESUMED")
        elif action == "toolbar":
            self.show_toolbar = not self.show_toolbar

    # -------------------------------------------------------------------- menus
    def _toggle_menu(self, kind):
        if self.picker.kind == kind:
            self.picker.close()
            return
        self._end_stroke()
        self._cancel_pinch()
        if kind == "brush":
            self.picker.open("brush", "CHOOSE A BRUSH", BRUSH_MODES, {"CALLIGRAPHY": "CALLIG."})
        elif kind == "shape":
            self.picker.open("shape", "CHOOSE A SHAPE  (pinch thumb + index and drag)", self.shape_modes,
                             {m: m.upper() for m in self.shape_modes})
        else:
            names = self.canvas_mgr.filter_names
            self.picker.open("filter", "CHOOSE A FILTER  (frame it with two L-shapes)", names, FILTER_SHORT)
            self.filter_previews = {}

    def _menu_choice(self, choice):
        if choice is None:
            return
        if choice == "close":
            self.picker.close()
            return
        if self.picker.kind == "brush":
            self.canvas_mgr.set_brush(choice)
        elif self.picker.kind == "shape":
            self.shape_index = self.shape_modes.index(choice)
            self.canvas_mgr.show_toast(f"SHAPE: {choice.upper()}")
        elif self.picker.kind == "filter":
            idx = self.canvas_mgr.filter_names.index(choice)
            self.active_filter_index = idx
            self.filter_chosen = True
            if self.canvas_mgr.active_filter_pts is not None:  # restyle the box already on screen
                self.canvas_mgr.save_state()
                self.canvas_mgr.active_filter_idx = idx
                self.canvas_mgr.show_toast(f"FILTER: {choice}")
            else:
                self.canvas_mgr.show_toast(f"FILTER: {choice} - FRAME WITH TWO L'S")
        self.picker.close()

    def _menu_previews(self, frame):
        """Preview images for the open menu: brushes in the current color, filters on the live camera."""
        tw, th = Picker.TILE_W, Picker.PREVIEW_H
        if self.picker.kind == "brush":
            color = self.colors[self.color_index]
            if color != self.brush_preview_color:
                self.brush_preview_color = color
                self.brush_previews = {m: self.canvas_mgr.brush_preview(m, color, tw, th) for m in BRUSH_MODES}
            return self.brush_previews, self.canvas_mgr.last_brush
        if self.picker.kind == "shape":
            return self._shape_previews(tw, th), self.shape_modes[self.shape_index]
        names = self.canvas_mgr.filter_names
        thumb = cv2.resize(frame, (tw, th), interpolation=cv2.INTER_AREA)
        if not self.filter_previews:  # first frame: render all of them once
            todo = names
        else:  # then refresh a few per frame so the live previews never stall the loop
            start = self.frame_no * FILTER_PREVIEWS_PER_FRAME
            todo = [names[(start + i) % len(names)] for i in range(FILTER_PREVIEWS_PER_FRAME)]
        for n in todo:
            self.filter_previews[n] = self.canvas_mgr.filter_image(thumb.copy(), n)
        return self.filter_previews, self.canvas_mgr.filter_names[self.active_filter_index]

    def _shape_previews(self, w, h):
        """Each shape drawn the way a pinch-drag from the white dot to the cyan dot would make it."""
        color = self.colors[self.color_index]
        out = {}
        for shape in self.shape_modes:
            img = np.zeros((h, w, 3), np.uint8)
            if shape == "circle":  # circle grows from the start point
                p1, p2 = (w // 2, h // 2), (w // 2 + int(h * 0.36), h // 2 + int(h * 0.12))
            else:
                p1, p2 = (int(w * 0.28), int(h * 0.18)), (int(w * 0.72), int(h * 0.82))
            cv2.line(img, p1, p2, (70, 70, 70), 1, cv2.LINE_AA)  # the drag
            self._draw_shape_preview(img, shape, p1, p2, color)
            cv2.circle(img, p1, 5, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(img, p2, 5, ACCENT, -1, cv2.LINE_AA)
            out[shape] = img
        return out

    # ------------------------------------------------------------------ helpers
    def _hold(self, hand, hold_s, now, preview=None, color=(255, 255, 255)):
        """True exactly once when `hand` has held its current pose for hold_s."""
        key = (hand.id, hand.pose_since)
        if self.fired_for == key:
            return False
        held = now - hand.pose_since
        if preview is not None:
            frac = min(1.0, held / hold_s)
            cv2.ellipse(preview, hand.pt(HL.INDEX_FINGER_TIP), (24, 24), -90, 0, 360 * frac, color, 3)
        if held >= hold_s and now - self.last_action_time >= ACTION_COOLDOWN_S:
            self.fired_for = key
            self.last_action_time = now
            return True
        return False

    def _extend_stroke(self, pt, color):
        if not self.stroke_active:
            self.stroke_active = True
            self.canvas_mgr.ink_width = None  # INK brush starts each stroke fresh
            self.stroke_saved = False
            self.stroke_last = None
            self.stroke_pending.clear()

        last = self.stroke_pending[-1] if self.stroke_pending else self.stroke_last
        if last is not None and np.hypot(pt[0] - last[0], pt[1] - last[1]) > MAX_JUMP_FRAC * self.width:
            self.stroke_pending.clear()  # tracking glitch: start a new segment
            self.stroke_last = None

        self.stroke_pending.append(pt)
        while len(self.stroke_pending) > DRAW_LAG_FRAMES:
            p = self.stroke_pending.popleft()
            if self.stroke_last is not None:
                if not self.stroke_saved:
                    self.canvas_mgr.save_state()
                    self.stroke_saved = True
                self.canvas_mgr.draw_line(self.stroke_last, p, color)
            self.stroke_last = p

    def _draw_pending(self, preview, color):
        pts = ([self.stroke_last] if self.stroke_last else []) + list(self.stroke_pending)
        if len(pts) >= 2:
            color = self.canvas_mgr.preview_color(color)
            cv2.polylines(preview, [np.array(pts, np.int32)], False, color, 6, cv2.LINE_AA)

    def _end_stroke(self):
        # Pending points are the tail where the finger was curling away: discard.
        self.stroke_active = False
        self.stroke_last = None
        self.stroke_pending.clear()

    def _cancel_pinch(self):
        self.pinch_start = None
        self.pinch_last = None

    def _draw_shape_preview(self, preview, shape, p1, p2, color):
        x1, y1 = p1
        x2, y2 = p2
        if shape == "rectangle":
            cv2.rectangle(preview, p1, p2, color, 3)
        elif shape == "line":
            cv2.line(preview, p1, p2, color, 3)
        elif shape == "circle":
            cv2.circle(preview, p1, int(np.hypot(x2 - x1, y2 - y1)), color, 3)
        elif shape == "triangle":
            pts = np.array([[(x1 + x2) // 2, y1], [x1, y2], [x2, y2]], np.int32)
            cv2.polylines(preview, [pts], isClosed=True, color=color, thickness=3)
        else:
            ax, ay = abs(x2 - x1) // 2, abs(y2 - y1) // 2
            if ax > 0 and ay > 0:
                cv2.ellipse(preview, ((x1 + x2) // 2, (y1 + y2) // 2), (ax, ay), 0, 0, 360, color, 3)

    # ----------------------------------------------------------------- handlers
    def _handle_one_hand(self, hand, now, preview, in_bar=False):
        """in_bar: the hand is over the toolbar, so nothing may be drawn."""
        pose, raw = hand.pose, hand.raw_pose
        tip = hand.pt(HL.INDEX_FINGER_TIP)
        color = self.colors[self.color_index]
        shape = self.shape_modes[self.shape_index]
        self.pose_label = pose

        if pose not in DRAW_POSES or in_bar:
            self._end_stroke()
        if pose != "pinch" and self.pinch_start is not None:
            s, e = self.pinch_start, self.pinch_last
            # Ignore tiny drags (e.g. an idle hand pinching and releasing),
            # and never drop a shape while the hand is over the toolbar.
            if not in_bar and np.hypot(e[0] - s[0], e[1] - s[1]) >= max(MIN_SHAPE_PX, MIN_SHAPE_PALM * hand.palm):
                self.canvas_mgr.commit_drag_shape(shape, s, e, color)
            self._cancel_pinch()
        if pose != "peace":
            self.prev_grab_pos = None

        if pose == "fist":
            self.clear_progress = min(1.0, (now - hand.pose_since) / CLEAR_HOLD_S)
            if self._hold(hand, CLEAR_HOLD_S, now):
                self._do("clear")
            elif self.fired_for == (hand.id, hand.pose_since):
                self.clear_progress = 0.0

        elif pose == "pinch":
            if in_bar:
                pass  # shape is frozen (not started, moved or previewed) over the toolbar
            elif raw == "pinch":  # only track while truly pinching (no release drift)
                # Midpoint of thumb + index tips: steadier than the index tip alone
                t, i = hand.pt(HL.THUMB_TIP), tip
                grab = ((t[0] + i[0]) // 2, (t[1] + i[1]) // 2)
                if self.pinch_start is None:
                    self.pinch_start = grab
                self.pinch_last = grab
            if self.pinch_start is not None and not in_bar:
                self._draw_shape_preview(preview, shape, self.pinch_start, self.pinch_last, color)

        elif pose in DRAW_POSES:
            if in_bar:
                self.pose_label = "toolbar"
            elif raw in DRAW_POSES:
                self._extend_stroke(tip, color)
            self._draw_pending(preview, color)

        elif pose == "shaka":
            if self._hold(hand, UNDO_HOLD_S, now, preview):
                self._do("undo")

        elif pose == "rock":
            if self._hold(hand, CYCLE_HOLD_S, now, preview, color):
                self._do("color")

        elif pose == "peace" and self.canvas_mgr.raw_overlay is not None:
            if raw == "peace":
                # Follow the middle knuckle: steadier than a fingertip
                grab = hand.pt(HL.MIDDLE_FINGER_MCP)
                if self.prev_grab_pos is not None:
                    self.canvas_mgr.move_overlay(grab[0] - self.prev_grab_pos[0], grab[1] - self.prev_grab_pos[1])
                self.prev_grab_pos = grab
            else:
                self.prev_grab_pos = None

    def _handle_two_hands(self, hands, now):
        left, right = sorted(hands, key=lambda h: h.lm[HL.WRIST][0])
        self.pose_label = f"{left.pose} + {right.pose}"
        has_overlay = self.canvas_mgr.raw_overlay is not None

        # --- Filter frame: both hands in an L-shape ---
        # Wrists must be clearly apart, so one hand detected twice can't frame a filter.
        apart = np.linalg.norm(left.lm[HL.WRIST][:2] - right.lm[HL.WRIST][:2]) > 1.5 * max(left.palm, right.palm)
        both_l = apart and left.pose == "l_shape" and right.pose == "l_shape"
        if both_l:
            self.frame_lost_since = None
            if not self.frame_active and now - max(left.pose_since, right.pose_since) >= FRAME_HOLD_S:
                self.frame_active = True
                self.canvas_mgr.save_state()
                if self.filter_chosen:  # use the filter picked from the menu
                    self.filter_chosen = False
                else:
                    self.active_filter_index = (self.active_filter_index + 1) % len(self.canvas_mgr.filter_names)
                self.canvas_mgr.show_toast(f"FILTER: {self.canvas_mgr.filter_names[self.active_filter_index]}".upper())
            if self.frame_active and left.raw_pose == "l_shape" and right.raw_pose == "l_shape":
                self.canvas_mgr.active_filter_pts = [
                    left.pt(HL.INDEX_FINGER_TIP),
                    left.pt(HL.THUMB_TIP),
                    right.pt(HL.INDEX_FINGER_TIP),
                    right.pt(HL.THUMB_TIP),
                ]
                self.canvas_mgr.active_filter_idx = self.active_filter_index
        elif self.frame_active:
            if self.frame_lost_since is None:
                self.frame_lost_since = now
            elif now - self.frame_lost_since >= FRAME_EXIT_GRACE_S:
                self.frame_active = False

        # --- Overlay: peace-sign grab (1 hand = move, 2 hands = scale) ---
        grabs = [h for h in (left, right) if h.pose == "peace" and h.raw_pose == "peace"]
        if has_overlay and len(grabs) == 2:
            a, b = grabs[0].pt(HL.MIDDLE_FINGER_MCP), grabs[1].pt(HL.MIDDLE_FINGER_MCP)
            dist = float(np.hypot(a[0] - b[0], a[1] - b[1]))
            if self.prev_grab_dist is not None:
                self.canvas_mgr.resize_overlay((dist - self.prev_grab_dist) * 0.005)
            self.prev_grab_dist = dist
            self.prev_grab_pos = None
        elif has_overlay and len(grabs) == 1:
            tip = grabs[0].pt(HL.MIDDLE_FINGER_MCP)
            if self.prev_grab_pos is not None:
                self.canvas_mgr.move_overlay(tip[0] - self.prev_grab_pos[0], tip[1] - self.prev_grab_pos[1])
            self.prev_grab_pos = tip
            self.prev_grab_dist = None
        else:
            self.prev_grab_pos = None
            self.prev_grab_dist = None

    def _update_hand_count(self, n, now):
        if n != self.raw_count:
            self.raw_count = n
            self.raw_count_since = now
        if n != self.hand_count and now - self.raw_count_since >= HAND_COUNT_CONFIRM_S:
            old = self.hand_count
            self.hand_count = n
            if old == 1:  # leaving single-hand mode
                self._end_stroke()
                self._cancel_pinch()
            if old == 2:  # leaving two-hand mode
                self.frame_active = False
                self.frame_lost_since = None
            self.prev_grab_pos = None
            self.prev_grab_dist = None

    # ------------------------------------------------------------ window/mouse
    def _on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.mouse_clicks.append((x, y))

    def _set_fullscreen(self, on):
        self.fullscreen = on
        cv2.setWindowProperty(
            WINDOW, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN if on else cv2.WINDOW_NORMAL
        )
        if not on:
            cv2.resizeWindow(WINDOW, self.width, self.height)

    def _handle_gallery_result(self, result):
        if isinstance(result, tuple) and result[0] == "edit":
            if self.canvas_mgr.load_into_canvas(result[1]):
                self.paused = False
                self.canvas_mgr.show_toast("IMAGE LOADED - KEEP DRAWING")

    def _show(self, img):
        cv2.imshow(WINDOW, img)
        if not self.icon_applied:  # the native window exists after the first imshow
            self.icon_applied = True
            apply_window_icon(WINDOW)

    # --------------------------------------------------------------------- loop
    def run(self):
        self.print_terminal_instructions()
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        set_app_id()
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(WINDOW, self._on_mouse)
        self.mouse_clicks = []
        self.icon_applied = False
        self._set_fullscreen(True)

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                continue

            now = time.monotonic()
            if frame.shape[1] != self.width or frame.shape[0] != self.height:
                frame = cv2.resize(frame, (self.width, self.height))  # camera ignored our size
            frame = cv2.flip(frame, 1)
            frame = cv2.convertScaleAbs(frame, alpha=1.0, beta=15)
            preview_layer = np.zeros_like(frame)
            self.clear_progress = 0.0
            self.pose_label = "-"

            self.frame_no += 1
            clean_frame = frame.copy() if self.picker.kind == "filter" else None  # menu previews, no landmarks
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)

            for hand_landmarks in results.multi_hand_landmarks or []:
                self.mp_drawing.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

            visible = self.tracker.update(results.multi_hand_landmarks, self.width, self.height, now)
            self._update_hand_count(len(visible), now)
            if self.prev_t is not None and now > self.prev_t:
                self.fps = 0.9 * self.fps + 0.1 * (1.0 / (now - self.prev_t))
            self.prev_t = now

            hand = None
            if self.hand_count == 1 and visible:
                hand = next((h for h in visible if h.id == self.primary_id), visible[0])
                self.primary_id = hand.id

            # Pointing fingertip doubles as the UI cursor (toolbar + gallery)
            cursor = None
            if hand is not None and hand.pose in DRAW_POSES and hand.raw_pose in DRAW_POSES:
                cursor = hand.pt(HL.INDEX_FINGER_TIP)

            # --- Gallery takes over the whole screen ---
            clicks, self.mouse_clicks = self.mouse_clicks, []
            if self.gallery.active:
                self._handle_gallery_result(self.gallery.update(cursor, now))
                for pt in clicks:
                    if self.gallery.active:
                        self._handle_gallery_result(self.gallery.click(pt))
                self._show(self.gallery.render(now))
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                    break
                if key == ord("f"):
                    self._set_fullscreen(not self.fullscreen)
                elif key != 255:
                    self._handle_gallery_result(self.gallery.key(key))
                continue

            # --- Toolbar (always shown while paused so you can resume hands-free) ---
            toolbar_on = self.show_toolbar or self.paused
            bar_cursor = cursor if toolbar_on and self.toolbar.contains(cursor) else None
            pressed = self.toolbar.update(bar_cursor, now)
            if pressed:
                self._do(pressed)
            menu_cursor = cursor if bar_cursor is None and self.picker.active else None
            if self.picker.active:
                self._menu_choice(self.picker.update(menu_cursor, now))
            for pt in clicks:  # mouse clicks: toolbar buttons first, then the open menu
                name = self.toolbar.hit(pt) if toolbar_on else None
                if name:
                    self._do(name)
                elif self.picker.active:
                    self._menu_choice(self.picker.click(pt))

            if self.picker.active:
                self.pose_label = "menu"  # no drawing or gestures while a menu is open
                self._end_stroke()
                self._cancel_pinch()
            elif self.paused:
                self.pose_label = "paused"
            elif hand is not None:
                # Nothing gets drawn while the index or thumb tip is over the toolbar
                in_bar = toolbar_on and (
                    self.toolbar.contains(hand.pt(HL.INDEX_FINGER_TIP), TOOLBAR_MARGIN)
                    or self.toolbar.contains(hand.pt(HL.THUMB_TIP), TOOLBAR_MARGIN)
                )
                self._handle_one_hand(hand, now, preview_layer, in_bar)
            elif self.hand_count == 2 and len(visible) == 2:
                self._handle_two_hands(visible, now)
            elif self.hand_count == 1 and self.stroke_active:
                self._draw_pending(preview_layer, self.colors[self.color_index])

            final_frame = self.canvas_mgr.merge_layers(frame, preview_layer)
            self.canvas_mgr.draw_hud(
                final_frame,
                self.shape_modes[self.shape_index],
                self.colors[self.color_index],
                self.clear_progress,
                top=self.toolbar.hud_offset if toolbar_on else 0,
            )
            cv2.putText(
                final_frame,
                f"Gesture: {self.pose_label.upper()}",
                (20, self.height - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (200, 200, 200),
                1,
            )
            if toolbar_on:
                self.toolbar.draw(final_frame, {
                    "color": self.colors[self.color_index],
                    "shape": self.shape_modes[self.shape_index],
                    "brush": self.canvas_mgr.brush_mode,
                    "last_brush": self.canvas_mgr.last_brush,
                    "filter": self.canvas_mgr.filter_names[self.active_filter_index],
                    "menu": self.picker.kind,
                    "recording": self.canvas_mgr.is_recording,
                    "paused": self.paused,
                })
                if bar_cursor is not None:
                    draw_cursor(final_frame, bar_cursor, self.toolbar.progress)
            if self.picker.active:
                previews, current = self._menu_previews(clean_frame if clean_frame is not None else frame)
                self.picker.draw(final_frame, previews, current)
                if menu_cursor is not None:
                    draw_cursor(final_frame, menu_cursor, self.picker.progress)
            if self.paused:
                msg = "PAUSED - hover RESUME or press P"
                (tw, _), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                x = (self.width - tw) // 2
                y = 88 + self.toolbar.hud_offset
                cv2.rectangle(final_frame, (x - 12, y), (x + tw + 12, y + 34), (0, 0, 0), -1)
                cv2.putText(final_frame, msg, (x, y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, ACCENT, 2, cv2.LINE_AA)

            self._show(final_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                break
            elif key == ord("f"):
                self._set_fullscreen(not self.fullscreen)
            elif key == 27 and self.picker.active:
                self.picker.close()
            elif key == 27 and self.fullscreen:
                self._set_fullscreen(False)
            elif key == ord("u"):
                self.canvas_mgr.prompt_and_load_background()
            else:
                action = {
                    ord("r"): "record", ord("s"): "save", ord("b"): "next_brush", ord("e"): "eraser",
                    ord("c"): "clear",
                    ord("p"): "pause", ord("g"): "gallery", ord("t"): "toolbar",
                }.get(key)
                if action:
                    self._do(action)

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    app = SkytouchApp()
    app.run()
