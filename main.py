import cv2
import mediapipe as mp
from gestures import GestureRecognizer
from canvas import CanvasManager

class SkytouchApp:
    def __init__(self, width=1280, height=720):
        self.width = width
        self.height = height

        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.8,
            min_tracking_confidence=0.8
        )

        self.recognizer = GestureRecognizer()
        self.canvas_mgr = CanvasManager(width, height)

        self.colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255)]
        self.color_index = 0
        self.modes = ["line", "circle", "rectangle"]
        self.mode_index = 0

        self.px, self.py = 0, 0
        self.is_pinching = False
        self.pinch_start_pt = None
        self.clear_counter = 0

        self.color_cooldown = 0
        self.mode_cooldown = 0
        self.pointing_frame_count = 0

        # Viewport Filter State
        self.active_filter_index = 0
        self.was_two_hands = False

    def run(self):
        cap = cv2.VideoCapture(0)
        cap.set(3, self.width)
        cap.set(4, self.height)

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                continue

            frame = cv2.flip(frame, 1)
            preview_layer = frame.copy()

            if self.color_cooldown > 0: self.color_cooldown -= 1
            if self.mode_cooldown > 0: self.mode_cooldown -= 1

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)

            clear_progress = 0.0

            if results.multi_hand_landmarks:
                num_hands = len(results.multi_hand_landmarks)

                for hand_landmarks in results.multi_hand_landmarks:
                    self.mp_drawing.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

                # --- TWO HAND MODE ---
                if num_hands == 2:
                    # Cycle to the next filter when viewport is freshly spawned
                    if not self.was_two_hands:
                        self.active_filter_index += 1
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
                    self.was_two_hands = False  # Reset spawn flag
                    data = self.recognizer.parse_hand_data(results.multi_hand_landmarks[0], self.width, self.height)
                    current_color = self.colors[self.color_index]
                    current_mode = self.modes[self.mode_index]

                    is_pinching_now = not data['is_fist'] and ((data['pinch_ratio'] < 0.18) or (self.is_pinching and data['pinch_ratio'] < 0.28))

                    # 1. Fist: Clear Canvas with status percentage tip
                    if data['is_fist']:
                        self.pointing_frame_count = 0
                        self.is_pinching = False
                        self.pinch_start_pt = None
                        self.clear_counter += 1
                        
                        clear_progress = min(1.0, self.clear_counter / 30.0)

                        if self.clear_counter >= 30:
                            self.canvas_mgr.clear()
                            self.px, self.py = 0, 0
                            self.clear_counter = 0

                    # 2. Pinch: Drag preview shape
                    elif is_pinching_now:
                        self.pointing_frame_count = 0
                        self.clear_counter = 0
                        self.px, self.py = 0, 0

                        if not self.is_pinching:
                            self.is_pinching = True
                            self.pinch_start_pt = data['index_tip']

                        x1, y1 = self.pinch_start_pt
                        x2, y2 = data['index_tip']
                        if current_mode == "rectangle":
                            cv2.rectangle(preview_layer, (x1, y1), (x2, y2), current_color, 3)
                        else:
                            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                            ax, ay = abs(x2 - x1) // 2, abs(y2 - y1) // 2
                            if ax > 0 and ay > 0:
                                cv2.ellipse(preview_layer, (cx, cy), (ax, ay), 0, 0, 360, current_color, 3)

                    # 3. Pinch release: Commit shape
                    elif self.is_pinching:
                        self.pointing_frame_count = 0
                        self.clear_counter = 0
                        self.canvas_mgr.commit_drag_shape(current_mode, self.pinch_start_pt, data['index_tip'], current_color)
                        self.is_pinching = False
                        self.pinch_start_pt = None

                    # 4. Color Swap
                    elif data['index_up'] and data['pinky_up'] and not data['middle_up'] and not data['ring_up'] and self.color_cooldown == 0:
                        self.pointing_frame_count = 0
                        self.clear_counter = 0
                        self.color_index = (self.color_index + 1) % len(self.colors)
                        self.color_cooldown = 20

                    # 5. Mode Swap
                    elif data['index_up'] and data['middle_up'] and not data['ring_up'] and not data['pinky_up'] and self.mode_cooldown == 0:
                        self.pointing_frame_count = 0
                        self.clear_counter = 0
                        self.mode_index = (self.mode_index + 1) % len(self.modes)
                        self.mode_cooldown = 20
                        self.px, self.py = 0, 0

                    # 6. Freehand Drawing
                    elif data['index_up'] and not data['middle_up'] and not data['ring_up'] and not data['pinky_up']:
                        self.clear_counter = 0
                        self.pointing_frame_count += 1
                        if self.pointing_frame_count >= 2:
                            if current_mode == "line":
                                if self.px == 0 and self.py == 0:
                                    self.px, self.py = data['index_tip']
                                self.canvas_mgr.draw_line((self.px, self.py), data['index_tip'], current_color)
                                self.px, self.py = data['index_tip']
                            else:
                                self.canvas_mgr.draw_shape_stamp(current_mode, data['index_tip'], current_color)
                                self.px, self.py = 0, 0
                    else:
                        self.pointing_frame_count = 0
                        self.clear_counter = 0
                        self.px, self.py = 0, 0

            else:
                self.was_two_hands = False  # Reset when no hands are in view

            # Render output
            final_frame = self.canvas_mgr.merge_layers(frame, preview_layer)
            self.canvas_mgr.draw_hud(final_frame, self.modes[self.mode_index], self.colors[self.color_index], clear_progress)

            cv2.imshow("Air Canvas & Filter Viewport", final_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    app = SkytouchApp()
    app.run()