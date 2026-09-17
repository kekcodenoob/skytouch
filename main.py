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
            min_tracking_confidence=0.7
        )

        # Setting up gestures.py and drawing tools from canvas.py
        self.recognizer = GestureRecognizer()
        self.canvas_mgr = CanvasManager(width, height)

        self.colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255)]
        self.color_names = ["Red", "Green", "Blue", "Yellow"]
        self.color_index = 0

        # Extended shape modes list
        self.shape_modes = ["rectangle", "ellipse", "line", "circle", "triangle"]
        self.shape_index = 0

        self.px, self.py = 0, 0
        self.is_pinching = False
        self.pinch_start_pt = None
        self.clear_counter = 0

        # Gesture Hold Counters & Threshold
        self.HOLD_THRESHOLD = 8  # Require ~8 consecutive frames before confirming selection
        self.color_hold_counter = 0
        self.shape_hold_counter = 0
        self.brush_hold_counter = 0

        self.gesture_cooldown = 0
        self.pointing_frame_count = 0
        self.is_drawing_stroke = False  # Track stroke state for undo

        # Viewport Filter State (8 total filters)
        self.active_filter_index = 0
        self.was_two_hands = False

        # LANDMARK SMOOTHING (EMA) STATE
        self.smoothed_pt = None
        self.alpha = 0.35  # Smoothing factor (0.1 = heavy smoothing, 1.0 = raw input)

    def smooth_point(self, raw_point):
        """Applies Exponential Moving Average (EMA) to smooth keypoint coordinates."""
        if self.smoothed_pt is None:
            self.smoothed_pt = (float(raw_point[0]), float(raw_point[1]))
            return raw_point

        sx = self.alpha * raw_point[0] + (1 - self.alpha) * self.smoothed_pt[0]
        sy = self.alpha * raw_point[1] + (1 - self.alpha) * self.smoothed_pt[1]
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
  [2] Drag Bounding Shape  : Pinch (Thumb + Index Finger) and Drag
  [3] Cycle Color          : Raise Index + Pinky Fingers (Hold)
  [4] Cycle Shape Mode     : Raise Index + Middle Fingers (Hold)
  [5] Cycle Brush Style    : Raise Index + Middle + Ring Fingers (Hold)
  [6] Undo Action          : Pinky + Thumb Out (Surfer Handsign)
  [7] Clear Canvas         : Hold a closed Fist (progress bar fills)
  [8] Viewport Filter Box  : Use TWO hands to form a boundary region

--- KEYBOARD SHORTCUTS ---
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

            # Natural camera brightness with slight contrast adjustment
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
                    self.mp_drawing.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

                # --- TWO HAND MODE ---
                if num_hands == 2:
                    self.smoothed_pt = None  # Reset single-finger filter
                    self.is_drawing_stroke = False
                    if not self.was_two_hands:
                        self.active_filter_index = (self.active_filter_index + 1) % len(self.canvas_mgr.filter_names)
                        self.was_two_hands = True

                    left_hand, right_hand = self.recognizer.sort_two_hands(results.multi_hand_landmarks)
                    lh = left_hand.landmark
                    rh = right_hand.landmark

                    p1 = (int(lh[self.mp_hands.HandLandmark.INDEX_FINGER_TIP].x * self.width),
                          int(lh[self.mp_hands.HandLandmark.INDEX_FINGER_TIP].y * self.height))
                    p2 = (int(lh[self.mp_hands.HandLandmark.THUMB_TIP].x * self.width),
                          int(lh[self.mp_hands.HandLandmark.THUMB_TIP].y * self.height))
                    p3 = (int(rh[self.mp_hands.HandLandmark.INDEX_FINGER_TIP].x * self.width),
                          int(rh[self.mp_hands.HandLandmark.INDEX_FINGER_TIP].y * self.height))
                    p4 = (int(rh[self.mp_hands.HandLandmark.THUMB_TIP].x * self.width),
                          int(rh[self.mp_hands.HandLandmark.THUMB_TIP].y * self.height))

                    frame = self.canvas_mgr.apply_viewport_filter(frame, p1, p2, p3, p4, self.active_filter_index)

                # --- SINGLE HAND MODE ---
                elif num_hands == 1:
                    self.was_two_hands = False
                    data = self.recognizer.parse_hand_data(results.multi_hand_landmarks[0], self.width, self.height)
                    
                    # Apply EMA smoothing filter to active index finger tip
                    smoothed_index_tip = self.smooth_point(data['index_tip'])
                    
                    current_color = self.colors[self.color_index]
                    current_shape = self.shape_modes[self.shape_index]

                    is_pinching_now = not data['is_fist'] and ((data['pinch_ratio'] < 0.18) or (self.is_pinching and data['pinch_ratio'] < 0.28))

                    # Track active gesture targets to clear hold counters when released
                    is_color_gesture = data['index_up'] and data['pinky_up'] and not data['middle_up'] and not data['ring_up']
                    is_shape_gesture = data['index_up'] and data['middle_up'] and not data['ring_up'] and not data['pinky_up']
                    is_brush_gesture = data['index_up'] and data['middle_up'] and data['ring_up'] and not data['pinky_up']

                    if not is_color_gesture:
                        self.color_hold_counter = 0
                    if not is_shape_gesture:
                        self.shape_hold_counter = 0
                    if not is_brush_gesture:
                        self.brush_hold_counter = 0

                    # 1. Fist: Clear Canvas
                    if data['is_fist']:
                        self.pointing_frame_count = 0
                        self.is_pinching = False
                        self.pinch_start_pt = None
                        self.smoothed_pt = None
                        self.is_drawing_stroke = False
                        self.clear_counter += 1
                        clear_progress = min(1.0, self.clear_counter / 30.0)

                        if self.clear_counter >= 30:
                            self.canvas_mgr.clear()
                            self.px, self.py = 0, 0
                            self.clear_counter = 0

                    # 2. Pinching: Live Shape Drag Preview (Updated for all shapes)
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
                            cv2.rectangle(preview_layer, (x1, y1), (x2, y2), current_color, 3)

                        elif current_shape == "line":
                            cv2.line(preview_layer, (x1, y1), (x2, y2), current_color, 3)

                        elif current_shape == "circle":
                            radius = int(np.hypot(x2 - x1, y2 - y1))
                            cv2.circle(preview_layer, (x1, y1), radius, current_color, 3)

                        elif current_shape == "triangle":
                            pts = np.array([
                                [(x1 + x2) // 2, y1],
                                [x1, y2],
                                [x2, y2]
                            ], np.int32)
                            cv2.polylines(preview_layer, [pts], isClosed=True, color=current_color, thickness=3)

                        else:  # Ellipse / Default
                            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                            ax, ay = abs(x2 - x1) // 2, abs(y2 - y1) // 2
                            if ax > 0 and ay > 0:
                                cv2.ellipse(preview_layer, (cx, cy), (ax, ay), 0, 0, 360, current_color, 3)

                    # 3. Pinch Release: Commit Shape to Canvas
                    elif self.is_pinching:
                        self.pointing_frame_count = 0
                        self.clear_counter = 0
                        self.is_drawing_stroke = False
                        self.canvas_mgr.commit_drag_shape(current_shape, self.pinch_start_pt, smoothed_index_tip, current_color)
                        self.is_pinching = False
                        self.pinch_start_pt = None

                    # 4. Gesture: Pinky + Thumb Out (Undo Action)
                    elif data['pinky_thumb_out'] and self.gesture_cooldown == 0:
                        self.pointing_frame_count = 0
                        self.clear_counter = 0
                        self.px, self.py = 0, 0
                        self.is_drawing_stroke = False
                        self.canvas_mgr.undo()
                        self.gesture_cooldown = 25

                    # 5. Gesture: Cycle Color (Index + Pinky Up with Hold)
                    elif is_color_gesture and self.gesture_cooldown == 0:
                        self.pointing_frame_count = 0
                        self.clear_counter = 0
                        self.is_drawing_stroke = False
                        self.color_hold_counter += 1

                        if self.color_hold_counter >= self.HOLD_THRESHOLD:
                            self.color_index = (self.color_index + 1) % len(self.colors)
                            self.gesture_cooldown = 25
                            self.color_hold_counter = 0
                            self.canvas_mgr.show_toast(f"COLOR: {self.color_names[self.color_index].upper()}")

                    # 6. Gesture: Cycle Shape Mode (Index + Middle Up with Hold)
                    elif is_shape_gesture and self.gesture_cooldown == 0:
                        self.pointing_frame_count = 0
                        self.clear_counter = 0
                        self.is_drawing_stroke = False
                        self.shape_hold_counter += 1

                        if self.shape_hold_counter >= self.HOLD_THRESHOLD:
                            self.shape_index = (self.shape_index + 1) % len(self.shape_modes)
                            self.gesture_cooldown = 25
                            self.shape_hold_counter = 0
                            self.canvas_mgr.show_toast(f"SHAPE: {self.shape_modes[self.shape_index].upper()}")

                    # 7. Gesture: Cycle Brush Style (Index + Middle + Ring Up with Hold)
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
                    elif data['index_up'] and not data['middle_up'] and not data['ring_up'] and not data['pinky_up']:
                        self.clear_counter = 0
                        self.pointing_frame_count += 1
                        
                        if self.pointing_frame_count >= 2:
                            # Save state ONCE at the start of a freehand stroke
                            if not self.is_drawing_stroke:
                                self.canvas_mgr.save_state()
                                self.is_drawing_stroke = True

                            if self.px == 0 and self.py == 0:
                                self.px, self.py = smoothed_index_tip
                            self.canvas_mgr.draw_line((self.px, self.py), smoothed_index_tip, current_color)
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

            # Render output layers
            final_frame = self.canvas_mgr.merge_layers(frame, preview_layer)
            self.canvas_mgr.draw_hud(final_frame, self.shape_modes[self.shape_index], self.colors[self.color_index], clear_progress)

            cv2.imshow("Skytouch", final_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                self.canvas_mgr.toggle_recording()
                status = "RECORDING STARTED" if self.canvas_mgr.is_recording else "RECORDING STOPPED"
                self.canvas_mgr.show_toast(status)
            elif key == ord('s'):
                self.canvas_mgr.save_image()
                self.canvas_mgr.show_toast("CANVAS SAVED")

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    app = SkytouchApp()
    app.run()