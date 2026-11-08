import time
from collections import deque

import cv2
import mediapipe as mp
import numpy as np

from canvas import CanvasManager
from gestures import HL, HandTracker

# --- Timing (seconds) -------------------------------------------------------
HAND_COUNT_CONFIRM_S = 0.15  # hand count must be stable this long before mode changes
CLEAR_HOLD_S = 1.5  # hold fist to clear
UNDO_HOLD_S = 0.35  # hold shaka to undo
CYCLE_HOLD_S = 0.40  # hold rock / peace to cycle color / shape
FRAME_HOLD_S = 0.30  # hold two L-shapes to start a filter box
FRAME_EXIT_GRACE_S = 0.40  # frame must be gone this long before it can re-trigger
ACTION_COOLDOWN_S = 0.60  # minimum gap between any two discrete actions

# --- Drawing ----------------------------------------------------------------
DRAW_LAG_FRAMES = 3  # points held back before committing; dropped when a stroke ends
MAX_JUMP_FRAC = 0.15  # break the line if the tip jumps more than this * width
DRAW_POSES = ("point", "l_shape")


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

        # Pinky-pinch shape drag
        self.pinch_start = None
        self.pinch_last = None

        # Two-hand filter frame
        self.active_filter_index = 0
        self.frame_active = False
        self.frame_lost_since = None

        # Overlay manipulation
        self.prev_ok_pos = None
        self.prev_ok_dist = None

        # Per-frame UI state
        self.clear_progress = 0.0
        self.pose_label = "-"

    def print_terminal_instructions(self):
        print("""
===================================================================
                    SKYTOUCH CONTROL INSTRUCTIONS
===================================================================

--- HAND GESTURES ---
  [1] Freehand Draw        : Raise Index Finger only
  [2] Drag Bounding Shape  : Touch Thumb to Pinky, drag, release to place
  [3] Move Overlay Image   : 1 Hand OK Gesture and Drag (image loaded)
  [4] Resize Overlay Image : 2 Hands OK Gestures Spread/Pinch
  [5] Cycle Color          : Hold Index + Pinky up
  [6] Cycle Shape Mode     : Hold Index + Middle up (Peace Sign)
  [7] Undo Action          : Hold Pinky + Thumb out (Shaka)
  [8] Clear Canvas         : Hold a closed Fist (~1.5s, bar fills)
  [9] Viewport Filter Box  : Make an "L" (thumb + index) with BOTH hands
  Open palm = idle, nothing happens. Held gestures fire once per hold.

--- KEYBOARD SHORTCUTS ---
  [b] Cycle Brush Style
  [c] Clear Canvas
  [u] Upload/Prompt Background Overlay Image
  [r] Toggle Video Recording
  [s] Save Canvas Artwork
  [q] Quit Application
===================================================================
""")

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
            if self.canvas_mgr.brush_mode == "ERASER":
                color = (128, 128, 128)
            cv2.polylines(preview, [np.array(pts, np.int32)], False, color, 6)

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
    def _handle_one_hand(self, hand, now, preview):
        pose, raw = hand.pose, hand.raw_pose
        tip = hand.pt(HL.INDEX_FINGER_TIP)
        color = self.colors[self.color_index]
        shape = self.shape_modes[self.shape_index]
        self.pose_label = pose

        if pose not in DRAW_POSES:
            self._end_stroke()
        if pose != "pinky_pinch" and self.pinch_start is not None:
            self.canvas_mgr.commit_drag_shape(shape, self.pinch_start, self.pinch_last, color)
            self._cancel_pinch()
        if pose != "ok":
            self.prev_ok_pos = None

        if pose == "fist":
            self.clear_progress = min(1.0, (now - hand.pose_since) / CLEAR_HOLD_S)
            if self._hold(hand, CLEAR_HOLD_S, now):
                self.canvas_mgr.clear()
                self.canvas_mgr.show_toast("CANVAS CLEARED")
            elif self.fired_for == (hand.id, hand.pose_since):
                self.clear_progress = 0.0

        elif pose == "pinky_pinch":
            if raw == "pinky_pinch":  # only track while truly pinching (no release drift)
                if self.pinch_start is None:
                    self.pinch_start = tip
                self.pinch_last = tip
            if self.pinch_start is not None:
                self._draw_shape_preview(preview, shape, self.pinch_start, self.pinch_last, color)

        elif pose in DRAW_POSES:
            if raw in DRAW_POSES:
                self._extend_stroke(tip, color)
            self._draw_pending(preview, color)

        elif pose == "shaka":
            if self._hold(hand, UNDO_HOLD_S, now, preview):
                self.canvas_mgr.undo()

        elif pose == "rock":
            if self._hold(hand, CYCLE_HOLD_S, now, preview, color):
                self.color_index = (self.color_index + 1) % len(self.colors)
                self.canvas_mgr.show_toast(f"COLOR: {self.color_names[self.color_index].upper()}")

        elif pose == "peace":
            if self._hold(hand, CYCLE_HOLD_S, now, preview):
                self.shape_index = (self.shape_index + 1) % len(self.shape_modes)
                self.canvas_mgr.show_toast(f"SHAPE: {self.shape_modes[self.shape_index].upper()}")

        elif pose == "ok" and self.canvas_mgr.raw_overlay is not None:
            if raw == "ok":
                if self.prev_ok_pos is not None:
                    self.canvas_mgr.move_overlay(tip[0] - self.prev_ok_pos[0], tip[1] - self.prev_ok_pos[1])
                self.prev_ok_pos = tip
            else:
                self.prev_ok_pos = None

    def _handle_two_hands(self, hands, now):
        left, right = sorted(hands, key=lambda h: h.lm[HL.WRIST][0])
        self.pose_label = f"{left.pose} + {right.pose}"
        has_overlay = self.canvas_mgr.raw_overlay is not None

        # --- Filter frame: both hands in an L-shape ---
        both_l = left.pose == "l_shape" and right.pose == "l_shape"
        if both_l:
            self.frame_lost_since = None
            if not self.frame_active and now - max(left.pose_since, right.pose_since) >= FRAME_HOLD_S:
                self.frame_active = True
                self.canvas_mgr.save_state()
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

        # --- Overlay: OK gestures ---
        oks = [h for h in (left, right) if h.pose == "ok" and h.raw_pose == "ok"]
        if has_overlay and len(oks) == 2:
            a, b = oks[0].pt(HL.INDEX_FINGER_TIP), oks[1].pt(HL.INDEX_FINGER_TIP)
            dist = float(np.hypot(a[0] - b[0], a[1] - b[1]))
            if self.prev_ok_dist is not None:
                self.canvas_mgr.resize_overlay((dist - self.prev_ok_dist) * 0.005)
            self.prev_ok_dist = dist
            self.prev_ok_pos = None
        elif has_overlay and len(oks) == 1:
            tip = oks[0].pt(HL.INDEX_FINGER_TIP)
            if self.prev_ok_pos is not None:
                self.canvas_mgr.move_overlay(tip[0] - self.prev_ok_pos[0], tip[1] - self.prev_ok_pos[1])
            self.prev_ok_pos = tip
            self.prev_ok_dist = None
        else:
            self.prev_ok_pos = None
            self.prev_ok_dist = None

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
            self.prev_ok_pos = None
            self.prev_ok_dist = None

    # --------------------------------------------------------------------- loop
    def run(self):
        self.print_terminal_instructions()
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                continue

            now = time.monotonic()
            frame = cv2.flip(frame, 1)
            frame = cv2.convertScaleAbs(frame, alpha=1.0, beta=15)
            preview_layer = np.zeros_like(frame)
            self.clear_progress = 0.0
            self.pose_label = "-"

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)

            for hand_landmarks in results.multi_hand_landmarks or []:
                self.mp_drawing.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

            visible = self.tracker.update(results.multi_hand_landmarks, self.width, self.height, now)
            self._update_hand_count(len(visible), now)

            if self.hand_count == 1 and visible:
                hand = next((h for h in visible if h.id == self.primary_id), visible[0])
                self.primary_id = hand.id
                self._handle_one_hand(hand, now, preview_layer)
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

            cv2.imshow("Skytouch", final_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("r"):
                self.canvas_mgr.toggle_recording()
                self.canvas_mgr.show_toast(
                    "RECORDING STARTED" if self.canvas_mgr.is_recording else "RECORDING STOPPED"
                )
            elif key == ord("s"):
                self.canvas_mgr.save_image()
                self.canvas_mgr.show_toast("CANVAS SAVED")
            elif key == ord("u"):
                self.canvas_mgr.prompt_and_load_background()
            elif key == ord("b"):
                self.canvas_mgr.cycle_brush_mode()
            elif key == ord("c"):
                self._end_stroke()
                self._cancel_pinch()
                self.canvas_mgr.clear()
                self.canvas_mgr.show_toast("CANVAS CLEARED")

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    app = SkytouchApp()
    app.run()
