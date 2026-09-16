import math
import mediapipe as mp

class GestureRecognizer:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        
    def calculate_distance(self, p1, p2):
        return math.hypot(p2[0] - p1[0], p2[1] - p1[1])

    def parse_hand_data(self, hand_landmarks, width, height):
        """Extracts key landmark coordinates and finger states for a single hand."""
        landmarks = hand_landmarks.landmark

        # Coordinates
        thumb_tip = (int(landmarks[self.mp_hands.HandLandmark.THUMB_TIP].x * width), 
                     int(landmarks[self.mp_hands.HandLandmark.THUMB_TIP].y * height))
        index_tip = (int(landmarks[self.mp_hands.HandLandmark.INDEX_FINGER_TIP].x * width), 
                     int(landmarks[self.mp_hands.HandLandmark.INDEX_FINGER_TIP].y * height))
        middle_tip = (int(landmarks[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP].x * width), 
                      int(landmarks[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y * height))
        ring_tip = (int(landmarks[self.mp_hands.HandLandmark.RING_FINGER_TIP].x * width), 
                    int(landmarks[self.mp_hands.HandLandmark.RING_FINGER_TIP].y * height))
        pinky_tip = (int(landmarks[self.mp_hands.HandLandmark.PINKY_TIP].x * width), 
                     int(landmarks[self.mp_hands.HandLandmark.PINKY_TIP].y * height))

        wrist = (int(landmarks[self.mp_hands.HandLandmark.WRIST].x * width), 
                 int(landmarks[self.mp_hands.HandLandmark.WRIST].y * height))
        index_mcp = (int(landmarks[self.mp_hands.HandLandmark.INDEX_FINGER_MCP].x * width), 
                     int(landmarks[self.mp_hands.HandLandmark.INDEX_FINGER_MCP].y * height))

        # Finger Extension Flags
        index_up = landmarks[self.mp_hands.HandLandmark.INDEX_FINGER_TIP].y < landmarks[self.mp_hands.HandLandmark.INDEX_FINGER_PIP].y
        middle_up = landmarks[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y < landmarks[self.mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y
        ring_up = landmarks[self.mp_hands.HandLandmark.RING_FINGER_TIP].y < landmarks[self.mp_hands.HandLandmark.RING_FINGER_PIP].y
        pinky_up = landmarks[self.mp_hands.HandLandmark.PINKY_TIP].y < landmarks[self.mp_hands.HandLandmark.PINKY_PIP].y

        # Fist Detection
        mcp_dist = self.calculate_distance(index_mcp, wrist)
        is_fist = (self.calculate_distance(index_tip, wrist) < mcp_dist * 1.25 and 
                   self.calculate_distance(middle_tip, wrist) < mcp_dist * 1.25 and 
                   self.calculate_distance(ring_tip, wrist) < mcp_dist * 1.25 and 
                   self.calculate_distance(pinky_tip, wrist) < mcp_dist * 1.25)

        # Pinch Ratio Check (Scaled to hand size)
        pinch_dist = self.calculate_distance(thumb_tip, index_tip)
        hand_scale = self.calculate_distance(wrist, index_mcp)
        pinch_ratio = pinch_dist / max(hand_scale, 1.0)

        return {
            'thumb_tip': thumb_tip,
            'index_tip': index_tip,
            'index_up': index_up,
            'middle_up': middle_up,
            'ring_up': ring_up,
            'pinky_up': pinky_up,
            'is_fist': is_fist,
            'pinch_ratio': pinch_ratio
        }

    def sort_two_hands(self, multi_hand_landmarks):
        """Ensures the left-most hand is assigned to left viewport corners."""
        sorted_hands = sorted(
            multi_hand_landmarks,
            key=lambda hand: hand.landmark[self.mp_hands.HandLandmark.WRIST].x
        )
        return sorted_hands[0], sorted_hands[1]