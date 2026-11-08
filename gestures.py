import math

import mediapipe as mp
import numpy as np

HL = mp.solutions.hands.HandLandmark

# --- Tuning -----------------------------------------------------------------
# Finger extension: bend angle (degrees) at the PIP/DIP joints.
FINGER_UP_BEND = 40.0  # straighter than this -> finger becomes "up"
FINGER_DOWN_BEND = 65.0  # more bent than this -> finger becomes "down"

# Thumb out: thumb tip distance from index knuckle, relative to palm size.
THUMB_OUT_ENTER = 0.85
THUMB_OUT_EXIT = 0.65

# Thumb + index pinch ratio (fingertip distance / palm size), with hysteresis.
PINCH_ENTER = 0.25
PINCH_EXIT = 0.35
L_SHAPE_MIN_GAP = 0.40  # thumb must be clearly apart from index for an "L"

FIST_TIP_RADIUS = 1.3  # all fingertips within this * palm of the wrist

POSE_CONFIRM_S = 0.10  # raw pose must hold this long to become the stable pose

# Hand tracking across frames
MATCH_DIST_FRAC = 0.25  # max wrist jump (fraction of width) to count as same hand
STATE_TIMEOUT_S = 0.30  # forget a hand after this long unseen
FILTER_RESET_GAP_S = 0.20  # reset smoothing if the hand was gone this long
DUPLICATE_HAND_FRAC = 0.5  # detections closer than this * palm (mean landmark dist) are one hand

FINGERS = {
    "index": (HL.INDEX_FINGER_MCP, HL.INDEX_FINGER_PIP, HL.INDEX_FINGER_TIP),
    "middle": (HL.MIDDLE_FINGER_MCP, HL.MIDDLE_FINGER_PIP, HL.MIDDLE_FINGER_TIP),
    "ring": (HL.RING_FINGER_MCP, HL.RING_FINGER_PIP, HL.RING_FINGER_TIP),
    "pinky": (HL.PINKY_MCP, HL.PINKY_PIP, HL.PINKY_TIP),
}


class OneEuroFilter:
    """One Euro filter over an array of points (vectorized).

    Smooths heavily when the hand is still (kills jitter) and lightly when it
    moves fast (kills lag). See Casiez et al., CHI 2012.
    """

    def __init__(self, min_cutoff=1.5, beta=0.008, d_cutoff=1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.reset()

    def reset(self):
        self.x_prev = None
        self.dx_prev = None
        self.t_prev = None

    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x, t):
        if self.x_prev is None:
            self.x_prev = x.copy()
            self.dx_prev = np.zeros_like(x)
            self.t_prev = t
            return x.copy()

        dt = max(t - self.t_prev, 1e-3)
        dx = (x - self.x_prev) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        dx_hat = a_d * dx + (1.0 - a_d) * self.dx_prev

        speed = np.linalg.norm(dx_hat, axis=-1, keepdims=True)
        a = self._alpha(self.min_cutoff + self.beta * speed, dt)
        x_hat = a * x + (1.0 - a) * self.x_prev

        self.x_prev, self.dx_prev, self.t_prev = x_hat, dx_hat, t
        return x_hat


def _dist(a, b):
    return float(np.linalg.norm(a - b))


def _bend_deg(mcp, pip, tip):
    v1, v2 = pip - mcp, tip - pip
    n = np.linalg.norm(v1) * np.linalg.norm(v2)
    if n < 1e-6:
        return 0.0
    return math.degrees(math.acos(max(-1.0, min(1.0, float(np.dot(v1, v2) / n)))))


class HandState:
    """Smoothed landmarks, finger states and debounced pose for one tracked hand."""

    def __init__(self, hand_id):
        self.id = hand_id
        self.filter = OneEuroFilter()
        self.lm = None  # (21, 3) smoothed landmarks in pixel space
        self.last_seen = 0.0

        self.fingers = {name: False for name in FINGERS}
        self.thumb_out = False
        self.pinch_ratio = 1.0

        self.raw_pose = "other"
        self.raw_since = 0.0
        self.pose = "other"  # debounced / stable pose
        self.pose_since = 0.0

    def pt(self, idx):
        return int(self.lm[idx][0]), int(self.lm[idx][1])

    def update(self, pts, now):
        if now - self.last_seen > FILTER_RESET_GAP_S:
            self.filter.reset()
        self.lm = self.filter(pts, now)
        self.last_seen = now
        self._update_features()
        self._debounce(self._classify(), now)

    def _update_features(self):
        lm = self.lm
        wrist = lm[HL.WRIST]
        palm = max(_dist(wrist, lm[HL.MIDDLE_FINGER_MCP]), 1.0)
        self.palm = palm

        for name, (mcp, pip, tip) in FINGERS.items():
            bend = _bend_deg(lm[mcp], lm[pip], lm[tip])
            reaches_out = _dist(lm[tip], wrist) > _dist(lm[pip], wrist)
            if self.fingers[name]:
                if bend > FINGER_DOWN_BEND or not reaches_out:
                    self.fingers[name] = False
            elif bend < FINGER_UP_BEND and reaches_out:
                self.fingers[name] = True

        thumb_ratio = _dist(lm[HL.THUMB_TIP], lm[HL.INDEX_FINGER_MCP]) / palm
        if self.thumb_out:
            self.thumb_out = thumb_ratio > THUMB_OUT_EXIT
        else:
            self.thumb_out = thumb_ratio > THUMB_OUT_ENTER

        self.pinch_ratio = _dist(lm[HL.THUMB_TIP], lm[HL.INDEX_FINGER_TIP]) / palm
        self.fist_shape = all(
            _dist(lm[tip], wrist) < palm * FIST_TIP_RADIUS
            for (_, _, tip) in FINGERS.values()
        )

    def _classify(self):
        f = self.fingers
        i, m, r, p = f["index"], f["middle"], f["ring"], f["pinky"]
        prev = self.raw_pose

        pinch_limit = PINCH_EXIT if prev == "pinch" else PINCH_ENTER
        if self.pinch_ratio < pinch_limit and not self.fist_shape:
            return "pinch"

        if self.fist_shape and not (i or m or r or p):
            return "fist"
        if i and not (m or r or p):
            if self.thumb_out and self.pinch_ratio > L_SHAPE_MIN_GAP:
                return "l_shape"
            return "point"
        if i and m and not (r or p):
            return "peace"
        if i and p and not (m or r):
            return "rock"
        if p and self.thumb_out and not (i or m or r):
            return "shaka"
        if i and m and r and p:
            return "open"
        return "other"

    def _debounce(self, pose, now):
        if pose != self.raw_pose:
            self.raw_pose = pose
            self.raw_since = now
        if pose != self.pose and now - self.raw_since >= POSE_CONFIRM_S:
            self.pose = pose
            self.pose_since = now


class HandTracker:
    """Keeps HandState objects matched to MediaPipe detections across frames."""

    def __init__(self):
        self.hands = []
        self._next_id = 0

    def update(self, multi_hand_landmarks, width, height, now):
        """Returns the HandStates visible this frame."""
        alive = [h for h in self.hands if now - h.last_seen < STATE_TIMEOUT_S]
        visible = []

        detections = []
        for hand_landmarks in multi_hand_landmarks or []:
            pts = np.array(
                [(l.x * width, l.y * height, l.z * width) for l in hand_landmarks.landmark],
                dtype=np.float64,
            )
            # MediaPipe sometimes reports one physical hand twice; drop near-copies.
            palm = max(np.linalg.norm(pts[HL.WRIST][:2] - pts[HL.MIDDLE_FINGER_MCP][:2]), 1.0)
            if any(
                np.mean(np.linalg.norm(pts[:, :2] - d[:, :2], axis=1)) < DUPLICATE_HAND_FRAC * palm
                for d in detections
            ):
                continue
            detections.append(pts)

        for pts in detections:
            wrist = pts[HL.WRIST][:2]
            candidates = [h for h in alive if h not in visible]
            best = min(
                candidates,
                key=lambda h: np.linalg.norm(h.lm[HL.WRIST][:2] - wrist),
                default=None,
            )
            if (
                best is None
                or np.linalg.norm(best.lm[HL.WRIST][:2] - wrist) > MATCH_DIST_FRAC * width
            ):
                best = HandState(self._next_id)
                self._next_id += 1
            best.update(pts, now)
            visible.append(best)

        self.hands = visible + [h for h in alive if h not in visible]
        return visible
