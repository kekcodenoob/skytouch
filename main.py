import cv2
import numpy as np
import mediapipe as mp
from gestures import GestureRecognizer
from canvas import CanvasManager


class SkytouchApp:

    def __init__(self, width=1280, height=720):
        self.width = width
        self.height = height

        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils

        # Balanced confidence thresholds to prevent tracking resets
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7,
        )

        # Module initializations
        self.recognizer = GestureRecognizer()
        self.canvas_mgr = CanvasManager(width, height)

        self.colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255)]
        self.color_names = ["Red", "Green", "Blue", "Yellow"]
        self.color_index = 0

        self.shape_modes = [
            "rectangle",
            "ellipse",
            "line",
            "circle",
            "triangle",
        ]
        self.shape_index = 0

        self.px, self.py = 0, 0
        self.is_pinching = False
        self.pinch_start_pt = None
        self.clear_counter = 0

        # Gesture Hold Counters & Threshold
        self.HOLD_THRESHOLD = 8  # Require ~8 consecutive frames before confirming
        self.color_hold_counter = 0
        self.shape_hold_counter = 0
        self.brush_hold_counter = 0

        self.gesture_cooldown = 0
        self.pointing_frame_count = 0
        self.is_drawing_stroke = False

        # Viewport Filter State
        self.active_filter_index = 0
        self.was_two_hands = False

        # Landmark Smoothing (EMA)
        self.smoothed_pt = None
        self.alpha = 0.35  # Smoothing factor

        # Overlay Manipulation State
        self.prev_ok_pos = None
        self.prev_ok_dist = None

    def smooth_point(self, raw_point):
        """Applies Exponential Moving Average (EMA) to smooth keypoint coordinates."""
        if self.smoothed_pt is None:
            self.smoothed_pt = (float(raw_point[0]), float(raw_point[1]))
            return raw_point

        sx = (
            self.alpha * raw_point[0]
            + (1 - self.alpha) * self.smoothed_pt[0]
        )
        sy = (
            self.alpha * raw_point[1]
            + (1 - self.alpha) * self.smoothed_pt[1]
        )
        self.smoothed_pt = (sx, sy)

        return int(sx), int(sy)

    def print_terminal_instructions(self):
        """Prints control instructions to the terminal."""
        instructions = """
===================================================================
                    SKYTOUCH CONTROL INSTRUCTIONS                  
===================================================================

--- HAND GESTURES ---
  [1] Freehand Draw        : Raise Index Finger ONLY
  [2] Drag Bounding Shape  : Pinky + Thumb Pinch and Drag
  [3] Move Overlay Image   : 1 Hand OK Gesture and Drag
  [4] Resize Overlay Image : 2 Hands OK Gestures Spread/Pinch
  [5] Cycle Color          : Raise Index + Pinky Fingers (Hold)
  [6] Cycle Shape Mode     : Raise Index + Middle Fingers (Peace Sign) (Hold)
  [7] Cycle Brush Style    : Raise Index + Middle + Ring Fingers (Hold)
  [8] Undo Action          : Pinky + Thumb Out (Surfer Handsign)
  [9] Clear Canvas         : Hold a closed Fist (progress bar fills)
  [10] Viewport Filter Box : Use TWO hands to form a boundary region

--- KEYBOARD SHORTCUTS ---
  [u] Upload/Prompt Background Overlay Image
  [r] Toggle Video Recording
  [s] Save Canvas Artwork
  [q] Quit Application

===================================================================
        """
        print(instructions)

    def run(self):
        self.print_terminal_instructions()
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                continue

            frame = cv2.flip(frame, 1)
            frame = cv2.convertScaleAbs(frame, alpha=1.0, beta=15)
            preview_layer = np.zeros_like(frame)

            if self.gesture_cooldown > 0:
                self.gesture_cooldown -= 1

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)
            clear_progress = 0.0

            if results.multi_hand_landmarks:
                num_hands = len(results.multi_hand_landmarks)

                for hand_landmarks in results.multi_hand_landmarks:
                    self.mp_drawing.draw_landmarks(
                        frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS
                    )

                # --- CHECK FOR OK GESTURE OVERLAY CONTROLS ---
                active_ok_gestures = []
                hand_datas = []

                for hand_landmarks in results.multi_hand_landmarks:
                    hdata = self.recognizer.parse_hand_data(
                        hand_landmarks, self.width, self.height
                    )
                    hand_datas.append(hdata)
                    if hdata.get("ok_gesture", False):
                        active_ok_gestures.append(hdata["index_tip"])

                # 1 Hand OK Gesture -> Move Overlay
                if len(active_ok_gestures) == 1:
                    curr_pos = active_ok_gestures[0]
                    if self.prev_ok_pos is not None:
                        dx = curr_pos[0] - self.prev_ok_pos[0]
                        dy = curr_pos[1] - self.prev_ok_pos[1]
                        self.canvas_mgr.move_overlay(dx, dy)
                    self.prev_ok_pos = curr_pos
                    self.prev_ok_dist = None

                # 2 Hands OK Gestures -> Scale Overlay
                elif len(active_ok_gestures) == 2:
                    dist = self.recognizer.calculate_distance(
                        active_ok_gestures[0], active_ok_gestures[1]
                    )
                    if self.prev_ok_dist is not None:
                        scale_delta = (dist - self.prev_ok_dist) * 0.005
                        self.canvas_mgr.resize_overlay(scale_delta)
                    self.prev_ok_dist = dist
                    self.prev_ok_pos = None

                else:
                    self.prev_ok_pos = None
                    self.prev_ok_dist = None

                # --- TWO HAND MODE (VIEWPORT FILTER) ---
                if num_hands == 2 and len(active_ok_gestures) == 0:
                    self.smoothed_pt = None
                    self.is_drawing_stroke = False
                    if not self.was_two_hands:
                        self.active_filter_index = (
                            self.active_filter_index + 1
                        ) % len(self.canvas_mgr.filter_names)
                        self.was_two_hands = True

                    left_hand, right_hand = self.recognizer.sort_two_hands(
                        results.multi_hand_landmarks
                    )
                    if left_hand and right_hand:
                        lh = left_hand.landmark
                        rh = right_hand.landmark

                        p1 = (
                            int(
                                lh[
                                    self.mp_hands.HandLandmark.INDEX_FINGER_TIP
                                ].x
                                * self.width
                            ),
                            int(
                                lh[
                                    self.mp_hands.HandLandmark.INDEX_FINGER_TIP
                                ].y
                                * self.height
                            ),
                        )
                        p2 = (
                            int(
                                lh[self.mp_hands.HandLandmark.THUMB_TIP].x
                                * self.width
                            ),
                            int(
                                lh[self.mp_hands.HandLandmark.THUMB_TIP].y
                                * self.height
                            ),
                        )
                        p3 = (
                            int(
                                rh[
                                    self.mp_hands.HandLandmark.INDEX_FINGER_TIP
                                ].x
                                * self.width
                            ),
                            int(
                                rh[
                                    self.mp_hands.HandLandmark.INDEX_FINGER_TIP
                                ].y
                                * self.height
                            ),
                        )
                        p4 = (
                            int(
                                rh[self.mp_hands.HandLandmark.THUMB_TIP].x
                                * self.width
                            ),
                            int(
                                rh[self.mp_hands.HandLandmark.THUMB_TIP].y
                                * self.height
                            ),
                        )

                        frame = self.canvas_mgr.apply_viewport_filter(
                            frame, p1, p2, p3, p4, self.active_filter_index
                        )

                # --- SINGLE HAND MODE ---
                elif num_hands == 1 and len(active_ok_gestures) == 0:
                    self.was_two_hands = False
                    data = hand_datas[0]

                    smoothed_index_tip = self.smooth_point(data["index_tip"])
                    current_color = self.colors[self.color_index]
                    current_shape = self.shape_modes[self.shape_index]

                    # Drag shape trigger: Pinky + Thumb Pinch
                    pinky_pinch = data.get("pinky_pinch_ratio", 1.0)
                    is_pinching_now = not data["is_fist"] and (
                        (pinky_pinch < 0.20)
                        or (self.is_pinching and pinky_pinch < 0.28)
                    )

                    # Identify explicit gesture flags
                    is_color_gesture = (
                        data["index_up"]
                        and data["pinky_up"]
                        and not data["middle_up"]
                        and not data["ring_up"]
                    )

                    # Cycle Shape Gesture: Index + Middle Finger raised
                    is_shape_gesture = (
                        data["index_up"]
                        and data["middle_up"]
                        and not data["ring_up"]
                        and not data["pinky_up"]
                    )

                    is_brush_gesture = (
                        data["index_up"]
                        and data["middle_up"]
                        and data["ring_up"]
                        and not data["pinky_up"]
                    )

                    # Reset idle hold counters when gestures are released
                    if not is_color_gesture:
                        self.color_hold_counter = 0
                    if not is_shape_gesture:
                        self.shape_hold_counter = 0
                    if not is_brush_gesture:
                        self.brush_hold_counter = 0

                    # 1. Fist: Clear Canvas
                    if data["is_fist"]:
                        self.pointing_frame_count = 0
                        self.is_pinching = False
                        self.pinch_start_pt = None
                        self.smoothed_pt = None
                        self.is_drawing_stroke = False
                        self.clear_counter += 1
                        clear_progress = min(1.0, self.clear_counter / 30.0)

                        # Reset all hold counters while clearing
                        self.color_hold_counter = 0
                        self.shape_hold_counter = 0
                        self.brush_hold_counter = 0

                        if self.clear_counter >= 30:
                            self.canvas_mgr.clear()
                            self.px, self.py = 0, 0
                            self.clear_counter = 0

                    # 2. Pinky Pinching: Live Shape Drag Preview
                    elif is_pinching_now:
                        self.pointing_frame_count = 0
                        self.clear_counter = 0
                        self.px, self.py = 0, 0
                        self.is_drawing_stroke = False

                        if not self.is_pinching:
                            self.is_pinching = True
                            self.pinch_start_pt = smoothed_index_tip

                        x1, y1 = self.pinch_start_pt
                        x2, y2 = smoothed_index_tip

                        if current_shape == "rectangle":
                            cv2.rectangle(
                                preview_layer,
                                (x1, y1),
                                (x2, y2),
                                current_color,
                                3,
                            )
                        elif current_shape == "line":
                            cv2.line(
                                preview_layer,
                                (x1, y1),
                                (x2, y2),
                                current_color,
                                3,
                            )
                        elif current_shape == "circle":
                            radius = int(np.hypot(x2 - x1, y2 - y1))
                            cv2.circle(
                                preview_layer,
                                (x1, y1),
                                radius,
                                current_color,
                                3,
                            )
                        elif current_shape == "triangle":
                            pts = np.array(
                                [
                                    [(x1 + x2) // 2, y1],
                                    [x1, y2],
                                    [x2, y2],
                                ],
                                np.int32,
                            )
                            cv2.polylines(
                                preview_layer,
                                [pts],
                                isClosed=True,
                                color=current_color,
                                thickness=3,
                            )
                        else:  # Ellipse / Default
                            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                            ax, ay = abs(x2 - x1) // 2, abs(y2 - y1) // 2
                            if ax > 0 and ay > 0:
                                cv2.ellipse(
                                    preview_layer,
                                    (cx, cy),
                                    (ax, ay),
                                    0,
                                    0,
                                    360,
                                    current_color,
                                    3,
                                )

                    # 3. Pinch Release: Commit Shape
                    elif self.is_pinching:
                        self.pointing_frame_count = 0
                        self.clear_counter = 0
                        self.is_drawing_stroke = False
                        self.canvas_mgr.commit_drag_shape(
                            current_shape,
                            self.pinch_start_pt,
                            smoothed_index_tip,
                            current_color,
                        )
                        self.is_pinching = False
                        self.pinch_start_pt = None

                    # 4. Gesture: Pinky + Thumb Out (Undo)
                    elif data["pinky_thumb_out"] and self.gesture_cooldown == 0:
                        self.pointing_frame_count = 0
                        self.clear_counter = 0
                        self.px, self.py = 0, 0
                        self.is_drawing_stroke = False
                        self.canvas_mgr.undo()
                        self.gesture_cooldown = 25

                    # 5. Gesture: Cycle Color
                    elif is_color_gesture and self.gesture_cooldown == 0:
                        self.pointing_frame_count = 0
                        self.clear_counter = 0
                        self.is_drawing_stroke = False
                        self.color_hold_counter += 1

                        if self.color_hold_counter >= self.HOLD_THRESHOLD:
                            self.color_index = (self.color_index + 1) % len(
                                self.colors
                            )
                            self.gesture_cooldown = 25
                            self.color_hold_counter = 0
                            self.canvas_mgr.show_toast(
                                f"COLOR: {self.color_names[self.color_index].upper()}"
                            )

                    # 6. Gesture: Cycle Shape (Index + Middle Up)
                    elif is_shape_gesture and self.gesture_cooldown == 0:
                        self.pointing_frame_count = 0
                        self.clear_counter = 0
                        self.is_drawing_stroke = False
                        self.shape_hold_counter += 1

                        if self.shape_hold_counter >= self.HOLD_THRESHOLD:
                            self.shape_index = (self.shape_index + 1) % len(
                                self.shape_modes
                            )
                            self.gesture_cooldown = 25
                            self.shape_hold_counter = 0
                            self.canvas_mgr.show_toast(
                                f"SHAPE: {self.shape_modes[self.shape_index].upper()}"
                            )

                    # 7. Gesture: Cycle Brush
                    elif is_brush_gesture and self.gesture_cooldown == 0:
                        self.pointing_frame_count = 0
                        self.clear_counter = 0
                        self.is_drawing_stroke = False
                        self.brush_hold_counter += 1

                        if self.brush_hold_counter >= self.HOLD_THRESHOLD:
                            self.canvas_mgr.cycle_brush_mode()
                            self.gesture_cooldown = 25
                            self.brush_hold_counter = 0

                    # 8. Single Index Finger Up: Freehand Drawing
                    elif (
                        data["index_up"]
                        and not data["middle_up"]
                        and not data["ring_up"]
                        and not data["pinky_up"]
                        and not data["thumbs_up"]
                    ):
                        self.clear_counter = 0
                        self.pointing_frame_count += 1

                        if self.pointing_frame_count >= 2:
                            if not self.is_drawing_stroke:
                                self.canvas_mgr.save_state()
                                self.is_drawing_stroke = True

                            if self.px == 0 and self.py == 0:
                                self.px, self.py = smoothed_index_tip
                            self.canvas_mgr.draw_line(
                                (self.px, self.py),
                                smoothed_index_tip,
                                current_color,
                            )
                            self.px, self.py = smoothed_index_tip

                    else:
                        self.pointing_frame_count = 0
                        self.clear_counter = 0
                        self.px, self.py = 0, 0
                        self.smoothed_pt = None
                        self.is_drawing_stroke = False

            else:
                self.was_two_hands = False
                self.smoothed_pt = None
                self.px, self.py = 0, 0
                self.is_drawing_stroke = False
                self.prev_ok_pos = None
                self.prev_ok_dist = None

            # Render output layers
            final_frame = self.canvas_mgr.merge_layers(frame, preview_layer)
            self.canvas_mgr.draw_hud(
                final_frame,
                self.shape_modes[self.shape_index],
                self.colors[self.color_index],
                clear_progress,
            )

            cv2.imshow("Skytouch", final_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("r"):
                self.canvas_mgr.toggle_recording()
                status = (
                    "RECORDING STARTED"
                    if self.canvas_mgr.is_recording
                    else "RECORDING STOPPED"
                )
                self.canvas_mgr.show_toast(status)
            elif key == ord("s"):
                self.canvas_mgr.save_image()
                self.canvas_mgr.show_toast("CANVAS SAVED")
            elif key == ord("u"):
                self.canvas_mgr.prompt_and_load_background()

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    app = SkytouchApp()
    app.run()