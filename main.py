import cv2
import math
import numpy as np
import mediapipe as mp

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# Webcam Setup
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

# Canvas & State Variables
canvas = None
colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255)] # BGR: Red, Green, Blue, Yellow
color_index = 0
drawing_mode = "line"  # Options: "line", "circle", "rectangle"

px, py = 0, 0          # Previous finger coordinates
start_x, start_y = 0, 0 # Shape start coordinates
color_cooldown = 0
mode_cooldown = 0

def calculate_distance(p1, p2):
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])

print("Gesture Controls:")
print(" - Point Index Finger: Draw / Position Shapes")
print(" - Pinch Thumb + Index: Change Color")
print(" - Pinch Thumb + Middle: Toggle Mode (Line -> Circle -> Rectangle)")
print(" - Open Hand (All Fingers): Clear Canvas")
print(" - Press 'q': Exit")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        continue

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    if canvas is None:
        canvas = np.zeros((h, w, 3), dtype=np.uint8)

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if color_cooldown > 0: color_cooldown -= 1
    if mode_cooldown > 0: mode_cooldown -= 1

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Extract Key Landmark Positions (x, y pixels)
            landmarks = hand_landmarks.landmark
            
            thumb_tip = (int(landmarks[4].x * w), int(landmarks[4].y * h))
            index_tip = (int(landmarks[8].x * w), int(landmarks[8].y * h))
            middle_tip = (int(landmarks[12].x * w), int(landmarks[12].y * h))
            
            # Check raised status of fingers
            index_up = landmarks[8].y < landmarks[6].y
            middle_up = landmarks[12].y < landmarks[10].y
            ring_up = landmarks[16].y < landmarks[14].y
            pinky_up = landmarks[20].y < landmarks[18].y

            # 1. Clear Canvas: Open Hand (All main fingers up)
            if index_up and middle_up and ring_up and pinky_up:
                canvas = np.zeros((h, w, 3), dtype=np.uint8)
                px, py = 0, 0

            # 2. Pinch Thumb + Index -> Cycle Colors
            elif calculate_distance(thumb_tip, index_tip) < 30 and color_cooldown == 0:
                color_index = (color_index + 1) % len(colors)
                color_cooldown = 15  # Prevent rapid switching
                px, py = 0, 0

            # 3. Pinch Thumb + Middle -> Switch Mode (Line / Circle / Rectangle)
            elif calculate_distance(thumb_tip, middle_tip) < 30 and mode_cooldown == 0:
                if drawing_mode == "line":
                    drawing_mode = "circle"
                elif drawing_mode == "circle":
                    drawing_mode = "rectangle"
                else:
                    drawing_mode = "line"
                mode_cooldown = 15
                px, py = 0, 0

            # 4. Drawing Logic (Only Index Finger Up)
            elif index_up and not middle_up:
                current_color = colors[color_index]
                
                if drawing_mode == "line":
                    if px == 0 and py == 0:
                        px, py = index_tip
                    cv2.line(canvas, (px, py), index_tip, current_color, 8)
                    px, py = index_tip

                elif drawing_mode == "circle":
                    cv2.circle(canvas, index_tip, 25, current_color, -1)
                    px, py = 0, 0

                elif drawing_mode == "rectangle":
                    cv2.rectangle(canvas, (index_tip[0] - 25, index_tip[1] - 25),
                                  (index_tip[0] + 25, index_tip[1] + 25), current_color, -1)
                    px, py = 0, 0
            else:
                px, py = 0, 0

            # Draw Hand Skeleton on Frame
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Merge Canvas with Video Stream
    gray_canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, inv_canvas = cv2.threshold(gray_canvas, 20, 255, cv2.THRESH_BINARY_INV)
    inv_canvas = cv2.cvtColor(inv_canvas, cv2.COLOR_GRAY2BGR)
    
    frame = cv2.bitwise_and(frame, inv_canvas)
    frame = cv2.bitwise_or(frame, canvas)

    # HUD / Overlay Information
    current_color_bgr = colors[color_index]
    cv2.putText(frame, f"Mode: {drawing_mode.upper()}", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.rectangle(frame, (10, 70), (110, 110), current_color_bgr, -1)
    cv2.rectangle(frame, (10, 70), (110, 110), (255, 255, 255), 2)

    cv2.imshow("Air Canvas & Gestures", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()