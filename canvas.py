import cv2
import numpy as np
import time
import tkinter as tk
from tkinter import filedialog

from media import new_media_path

# Eraser is not in the cycle; it's a separate toggle (toolbar button / `e` key).
BRUSH_MODES = ["SOLID", "NEON", "RAINBOW", "INK", "CALLIGRAPHY", "SPRAY", "SPARKLE", "DASHED"]


class CanvasManager:

    def __init__(self, width=1280, height=720):
        self.width = width
        self.height = height

        # Overlay Image Properties
        self.raw_overlay = None
        self.overlay_pos = [100, 100]
        self.overlay_scale = 1.0
        self.overlay_alpha = 0.7

        # Primary drawing canvas
        self.canvas = np.zeros((height, width, 3), dtype=np.uint8)

        # Persistent Viewport Filter State
        self.active_filter_pts = None  # Stores saved quad points [p1, p2, p3, p4]
        self.active_filter_idx = None  # Stores applied filter index

        # Undo history stack (stores tuple of canvas and active filter state)
        self.undo_stack = []
        self.max_undo = 50

        # Optional full-size background image
        self.background_image = None

        # Brush state configuration
        self.brush_mode = "SOLID"  # one of BRUSH_MODES, or "ERASER"
        self.last_brush = "SOLID"  # brush to return to when the eraser is toggled off
        self.dash_counter = 0
        self.rainbow_hue = 0
        self.ink_width = None
        self.rng = np.random.default_rng()

        # Filter list for viewport rendering
        self.filter_names = [
            "GRAYSCALE",
            "INVERT",
            "SEPIA",
            "SOBEL",
            "BLUR",
            "PIXELATE",
            "THERMAL",
            "GLASS",
            "ASCII",
            "NIGHT_VISION",
            "KALEIDOSCOPE",
        ]

        # On-screen toast notifications
        self.toast_msg = ""
        self.toast_end_time = 0.0

        # Recording stream state
        self.video_writer = None
        self.is_recording = False

    def save_state(self):
        """Saves current canvas state and active filter state to undo stack."""
        if len(self.undo_stack) >= self.max_undo:
            self.undo_stack.pop(0)

        # Store canvas copy along with current persistent filter state
        filter_state = (
            self.active_filter_pts.copy()
            if self.active_filter_pts is not None
            else None,
            self.active_filter_idx,
        )
        self.undo_stack.append((self.canvas.copy(), filter_state))

    def undo(self):
        """Restores canvas and active filter to the previous saved state."""
        if self.undo_stack:
            saved_canvas, saved_filter_state = self.undo_stack.pop()
            self.canvas = saved_canvas
            if saved_filter_state is not None:
                self.active_filter_pts, self.active_filter_idx = (
                    saved_filter_state
                )
            else:
                self.active_filter_pts = None
                self.active_filter_idx = None
            self.show_toast("UNDO SUCCESSFUL")
            return True
        else:
            self.show_toast("NOTHING TO UNDO")
            return False

    def show_toast(self, text, duration=1.5):
        """Triggers an on-screen toast announcement message."""
        self.toast_msg = text
        self.toast_end_time = time.time() + duration

    def cycle_brush_mode(self):
        """Cycles through available brush styles."""
        if self.brush_mode == "ERASER":
            # Leaving the eraser via the brush button: go back to the last brush
            self.brush_mode = self.last_brush
        else:
            idx = BRUSH_MODES.index(self.brush_mode)
            self.brush_mode = BRUSH_MODES[(idx + 1) % len(BRUSH_MODES)]
        self.last_brush = self.brush_mode
        self.ink_width = None
        self.show_toast(f"BRUSH: {self.brush_mode}")
        return self.brush_mode

    def set_brush(self, mode):
        """Picks a brush directly (from the brush menu)."""
        self.brush_mode = self.last_brush = mode
        self.ink_width = None
        self.show_toast(f"BRUSH: {mode}")

    def brush_preview(self, mode, color, w, h):
        """Renders a sample wavy stroke in `mode` on a small dark tile."""
        saved = (self.canvas, self.brush_mode, self.width, self.height,
                 self.dash_counter, self.rainbow_hue, self.ink_width)
        try:
            self.canvas = np.zeros((h, w, 3), np.uint8)
            self.width, self.height = w, h
            self.brush_mode, self.dash_counter, self.ink_width = mode, 0, None
            # Points get further apart along the stroke (speeding up), so INK visibly thins
            t = np.linspace(0, 1, 40) ** 1.6
            xs = w * (0.08 + 0.84 * t)
            ys = h / 2 + np.sin(t * 2 * np.pi) * h * 0.28
            pts = [(int(x), int(y)) for x, y in zip(xs, ys)]
            for a, b in zip(pts, pts[1:]):
                self.draw_line(a, b, color)
            return self.canvas
        finally:
            (self.canvas, self.brush_mode, self.width, self.height,
             self.dash_counter, self.rainbow_hue, self.ink_width) = saved

    def toggle_eraser(self):
        """Switches to the eraser, or back to the brush that was active before."""
        if self.brush_mode == "ERASER":
            self.brush_mode = self.last_brush
            self.show_toast(f"BRUSH: {self.brush_mode}")
        else:
            self.last_brush = self.brush_mode
            self.brush_mode = "ERASER"
            self.show_toast("ERASER ON")
        self.ink_width = None
        return self.brush_mode

    def preview_color(self, color):
        """Color to use for the live (not yet committed) part of a stroke."""
        if self.brush_mode == "ERASER":
            return (128, 128, 128)
        if self.brush_mode == "RAINBOW":
            return self._hue_bgr(self.rainbow_hue)
        return color

    @staticmethod
    def _hue_bgr(hue):
        hsv = np.uint8([[[hue % 180, 255, 255]]])
        return tuple(int(c) for c in cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0])

    def _spray(self, p1, p2, color, radius, density):
        """Scatter dots around the segment p1 -> p2 (spray paint / sparkle dust)."""
        seg = max(1.0, float(np.hypot(p2[0] - p1[0], p2[1] - p1[1])))
        n = int(density * (seg / 4 + 3))
        t = self.rng.random(n)
        pts = np.stack([p1[0] + (p2[0] - p1[0]) * t, p1[1] + (p2[1] - p1[1]) * t], axis=1)
        pts += self.rng.normal(0, radius / 2, (n, 2))
        for x, y in pts.astype(int):
            if 0 <= x < self.width and 0 <= y < self.height:
                self.canvas[y, x] = color
        return pts

    def load_into_canvas(self, image_path):
        """Puts a saved artwork back on the drawing layer so it can be edited (undoable)."""
        img = cv2.imread(image_path)
        if img is None:
            self.show_toast("FAILED TO LOAD IMAGE")
            return False
        self.save_state()
        self.canvas = cv2.resize(img, (self.width, self.height), interpolation=cv2.INTER_AREA)
        return True

    def load_background_image(self, image_path):
        """Loads a static background image stretched across canvas dimensions."""
        img = cv2.imread(image_path)
        if img is not None:
            self.background_image = cv2.resize(img, (self.width, self.height))
            return True
        return False

    def clear(self):
        """Resets drawing layer and active persistent filter."""
        self.save_state()
        self.canvas[:] = 0
        self.active_filter_pts = None
        self.active_filter_idx = None

    def draw_line(self, p1, p2, color, thickness=6):
        """Draws lines onto the canvas layer using the active brush style."""
        if self.brush_mode == "ERASER":
            cv2.line(self.canvas, p1, p2, (0, 0, 0), thickness * 4)

        elif self.brush_mode == "NEON":
            cv2.line(self.canvas, p1, p2, color, thickness * 2)
            cv2.line(
                self.canvas,
                p1,
                p2,
                (255, 255, 255),
                max(2, thickness // 3),
            )

        elif self.brush_mode == "DASHED":
            self.dash_counter += 1
            if (self.dash_counter // 4) % 2 == 0:
                cv2.line(self.canvas, p1, p2, color, thickness)

        elif self.brush_mode == "RAINBOW":
            self.rainbow_hue = (self.rainbow_hue + 3) % 180
            cv2.line(self.canvas, p1, p2, self._hue_bgr(self.rainbow_hue), thickness, cv2.LINE_AA)

        elif self.brush_mode == "INK":
            # Fast strokes get thinner, slow strokes get thicker, like a brush pen
            seg = float(np.hypot(p2[0] - p1[0], p2[1] - p1[1]))
            target = float(np.clip(thickness * 2.4 - seg * 0.22, 2, thickness * 2.4))
            self.ink_width = target if self.ink_width is None else 0.7 * self.ink_width + 0.3 * target
            cv2.line(self.canvas, p1, p2, color, max(1, int(self.ink_width)), cv2.LINE_AA)

        elif self.brush_mode == "CALLIGRAPHY":
            # Flat nib held at 45 degrees: thick or thin depending on stroke direction
            n = np.array([1.0, -1.0]) / np.sqrt(2) * thickness * 1.5
            a, b = np.array(p1, float), np.array(p2, float)
            quad = np.array([a + n, b + n, b - n, a - n], np.int32)
            cv2.fillConvexPoly(self.canvas, quad, color, cv2.LINE_AA)

        elif self.brush_mode == "SPRAY":
            self._spray(p1, p2, color, radius=thickness * 4, density=6)

        elif self.brush_mode == "SPARKLE":
            cv2.line(self.canvas, p1, p2, color, max(2, thickness // 2), cv2.LINE_AA)
            dust = self._spray(p1, p2, (255, 255, 255), radius=thickness * 5, density=0.6)
            if self.rng.random() < 0.35 and len(dust):  # occasional 4-point star
                x, y = dust[0].astype(int)
                r = int(self.rng.integers(5, 11))
                cv2.line(self.canvas, (x - r, y), (x + r, y), (255, 255, 255), 1, cv2.LINE_AA)
                cv2.line(self.canvas, (x, y - r), (x, y + r), (255, 255, 255), 1, cv2.LINE_AA)
                cv2.circle(self.canvas, (x, y), 2, color, -1, cv2.LINE_AA)

        else:  # SOLID
            cv2.line(self.canvas, p1, p2, color, thickness)

    def commit_drag_shape(self, mode, p1, p2, color):
        """Commits shape geometry directly onto the canvas."""
        if not p1 or not p2:
            return

        self.save_state()

        x1, y1 = p1
        x2, y2 = p2

        if mode == "rectangle":
            cv2.rectangle(self.canvas, (x1, y1), (x2, y2), color, 3)

        elif mode == "line":
            cv2.line(self.canvas, (x1, y1), (x2, y2), color, 3)

        elif mode == "circle":
            radius = int(np.hypot(x2 - x1, y2 - y1))
            cv2.circle(self.canvas, (x1, y1), radius, color, 3)

        elif mode == "triangle":
            pts = np.array(
                [
                    [(x1 + x2) // 2, y1],
                    [x1, y2],
                    [x2, y2],
                ],
                np.int32,
            )
            cv2.polylines(
                self.canvas, [pts], isClosed=True, color=color, thickness=3
            )

        else:  # Ellipse / Default
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            ax, ay = abs(x2 - x1) // 2, abs(y2 - y1) // 2
            if ax > 0 and ay > 0:
                cv2.ellipse(
                    self.canvas, (cx, cy), (ax, ay), 0, 0, 360, color, 3
                )

    def filter_image(self, roi, filter_name):
        """Returns a filtered copy of `roi` (any size). Used for the filter box and menu previews."""
        h, w = roi.shape[:2]
        if filter_name == "GRAYSCALE":
            filtered_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            filtered_roi = cv2.cvtColor(filtered_roi, cv2.COLOR_GRAY2BGR)

        elif filter_name == "INVERT":
            filtered_roi = cv2.bitwise_not(roi)

        elif filter_name == "SEPIA":
            sepia_kernel = np.array(
                [
                    [0.272, 0.534, 0.131],
                    [0.349, 0.686, 0.168],
                    [0.393, 0.769, 0.189],
                ]
            )
            filtered_roi = cv2.transform(roi, sepia_kernel)
            filtered_roi = np.clip(filtered_roi, 0, 255).astype(np.uint8)

        elif filter_name == "SOBEL":
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            sobel = cv2.magnitude(sobelx, sobely)
            sobel = np.uint8(np.clip(sobel, 0, 255))
            filtered_roi = cv2.cvtColor(sobel, cv2.COLOR_GRAY2BGR)

        elif filter_name == "BLUR":
            filtered_roi = cv2.GaussianBlur(roi, (21, 21), 0)

        elif filter_name == "PIXELATE":
            pixel_size = 16
            temp_w = max(1, w // pixel_size)
            temp_h = max(1, h // pixel_size)
            small = cv2.resize(
                roi, (temp_w, temp_h), interpolation=cv2.INTER_LINEAR
            )
            filtered_roi = cv2.resize(
                small, (w, h), interpolation=cv2.INTER_NEAREST
            )

        elif filter_name == "THERMAL":
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            filtered_roi = cv2.applyColorMap(gray, cv2.COLORMAP_JET)

        elif filter_name == "GLASS":
            stripe_width = 12.0
            distortion_amplitude = 6.0

            x_coords = np.arange(w, dtype=np.float32)
            displacement = (
                np.sin(2 * np.pi * x_coords / stripe_width)
                * distortion_amplitude
            )

            map_x = np.tile(x_coords + displacement, (h, 1)).astype(np.float32)
            map_y = np.tile(np.arange(h, dtype=np.float32)[:, None], (1, w)).astype(
                np.float32
            )

            map_x = np.clip(map_x, 0, w - 1)
            map_y = np.clip(map_y, 0, h - 1)

            distorted = cv2.remap(roi, map_x, map_y, cv2.INTER_LINEAR)
            blurred = cv2.GaussianBlur(distorted, (1, 15), 0)

            highlights = (
                np.sin(2 * np.pi * x_coords / stripe_width) + 1.0
            ) / 2.0
            highlights = (highlights * 25).astype(np.uint8)

            highlight_strip = highlights[None, :, None]
            highlight_mask = np.tile(highlight_strip, (h, 1, 1))
            highlight_mask = np.repeat(highlight_mask, 3, axis=2)

            filtered_roi = cv2.add(blurred, highlight_mask)

        elif filter_name == "ASCII":
            chars = np.array(list(" .:-=+*#%@"))
            scale_factor = 10
            small_w = max(1, w // scale_factor)
            small_h = max(1, h // scale_factor)

            small_roi = cv2.resize(
                roi, (small_w, small_h), interpolation=cv2.INTER_AREA
            )
            small_gray = cv2.cvtColor(small_roi, cv2.COLOR_BGR2GRAY)

            char_indices = (
                small_gray.astype(np.float32) / 255.0 * (len(chars) - 1)
            ).astype(np.int32)
            ascii_canvas = np.zeros((h, w, 3), dtype=np.uint8)

            cell_w = w / small_w
            cell_h = h / small_h
            font_scale = max(0.3, min(cell_w, cell_h) / 18.0)

            for row in range(small_h):
                for col in range(small_w):
                    ch = chars[char_indices[row, col]]
                    posX = int(col * cell_w)
                    posY = int((row + 1) * cell_h - 2)

                    cell_color = tuple(int(c) for c in small_roi[row, col])

                    cv2.putText(
                        ascii_canvas,
                        ch,
                        (posX, posY),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale,
                        cell_color,
                        1,
                        cv2.LINE_AA,
                    )

            filtered_roi = ascii_canvas

        elif filter_name == "NIGHT_VISION":
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            bright_gray = cv2.convertScaleAbs(gray, alpha=1.4, beta=20)

            green_tinted = np.zeros_like(roi)
            green_tinted[:, :, 1] = bright_gray
            green_tinted[:, :, 0] = (bright_gray * 0.15).astype(np.uint8)
            green_tinted[:, :, 2] = (bright_gray * 0.15).astype(np.uint8)

            scanlines = np.ones((h, w), dtype=np.float32)
            scanlines[::4, :] = 0.6

            for c in range(3):
                green_tinted[:, :, c] = np.clip(
                    green_tinted[:, :, c] * scanlines, 0, 255
                )

            noise = np.random.randint(-15, 15, (h, w, 3), dtype=np.int16)
            noisy_nv = np.clip(
                green_tinted.astype(np.int16) + noise, 0, 255
            ).astype(np.uint8)

            filtered_roi = noisy_nv

        elif filter_name == "KALEIDOSCOPE":
            num_slices = 8
            cx, cy = w // 2, h // 2

            yy, xx = np.mgrid[0:h, 0:w]
            dx = xx - cx
            dy = yy - cy

            r = np.hypot(dx, dy)
            theta = np.arctan2(dy, dx)

            slice_angle = (2 * np.pi) / num_slices
            theta_folded = np.abs(
                (theta % slice_angle) - (slice_angle / 2.0)
            )

            map_x = np.clip(cx + r * np.cos(theta_folded), 0, w - 1).astype(
                np.float32
            )
            map_y = np.clip(cy + r * np.sin(theta_folded), 0, h - 1).astype(
                np.float32
            )

            filtered_roi = cv2.remap(roi, map_x, map_y, cv2.INTER_LINEAR)

        else:
            filtered_roi = roi
        return filtered_roi

    def apply_viewport_filter(self, frame, p1, p2, p3, p4, filter_idx):
        """Applies real-time image filters inside a bounded quad region."""
        pts = np.array([p1, p2, p4, p3], dtype=np.int32)
        x, y, w, h = cv2.boundingRect(pts)

        x = max(0, x)
        y = max(0, y)
        w = min(self.width - x, w)
        h = min(self.height - y, h)

        if w <= 0 or h <= 0:
            return frame

        roi = frame[y : y + h, x : x + w]
        if roi.size == 0:
            return frame

        filter_name = self.filter_names[filter_idx % len(self.filter_names)]
        filtered_roi = self.filter_image(roi, filter_name)

        mask = np.zeros((h, w), dtype=np.uint8)
        shifted_pts = pts - np.array([x, y])
        cv2.fillConvexPoly(mask, shifted_pts, 255)

        mask_inv = cv2.bitwise_not(mask)
        img1_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
        img2_fg = cv2.bitwise_and(filtered_roi, filtered_roi, mask=mask)

        frame[y : y + h, x : x + w] = cv2.add(img1_bg, img2_fg)
        cv2.polylines(frame, [pts], True, (0, 255, 255), 2)

        top_y = max(25, min(p1[1], p2[1], p3[1], p4[1]) - 10)
        min_x = max(10, min(p1[0], p2[0], p3[0], p4[0]))

        cv2.putText(
            frame,
            f"FILTER: {filter_name}",
            (min_x, top_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
        )

        return frame

    def draw_hud(self, frame, shape_mode, color, clear_progress=0.0, top=0):
        """Renders HUD status indicators and toast announcements.

        `top` shifts everything down (e.g. to sit below a top toolbar)."""
        cv2.putText(
            frame,
            f"Brush Style: {self.brush_mode}",
            (20, 35 + top),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            frame,
            f"Pinch Shape: {shape_mode.upper()}",
            (20, 65 + top),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (200, 200, 200),
            1,
        )

        cv2.rectangle(frame, (20, 80 + top), (60, 110 + top), color, -1)
        cv2.rectangle(frame, (20, 80 + top), (60, 110 + top), (255, 255, 255), 2)

        if time.time() < self.toast_end_time:
            text_size = cv2.getTextSize(
                self.toast_msg, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2
            )[0]
            cx = (self.width - text_size[0]) // 2

            cv2.rectangle(
                frame,
                (cx - 15, 30 + top),
                (cx + text_size[0] + 15, 75 + top),
                (0, 0, 0),
                -1,
            )
            cv2.rectangle(
                frame,
                (cx - 15, 30 + top),
                (cx + text_size[0] + 15, 75 + top),
                (0, 255, 255),
                2,
            )
            cv2.putText(
                frame,
                self.toast_msg,
                (cx, 62 + top),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 255),
                2,
            )

        if clear_progress > 0:
            bar_w = int(200 * clear_progress)
            cv2.rectangle(frame, (20, 130 + top), (220, 145 + top), (50, 50, 50), -1)
            cv2.rectangle(frame, (20, 130 + top), (20 + bar_w, 145 + top), (0, 0, 255), -1)
            cv2.putText(
                frame,
                "HOLD FIST TO CLEAR",
                (20, 125 + top),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 0, 255),
                1,
            )

        if self.is_recording:
            cv2.circle(frame, (self.width - 30, 30 + top), 10, (0, 0, 255), -1)
            cv2.putText(
                frame,
                "REC",
                (self.width - 80, 35 + top),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                2,
            )

    def toggle_recording(self, fps=20.0):
        """Toggles MP4 recording on and off. `fps` should match the real loop rate
        so playback runs at normal speed."""
        if not self.is_recording:
            self.recording_path = new_media_path(".mp4")
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self.video_writer = cv2.VideoWriter(
                self.recording_path, fourcc, float(fps), (self.width, self.height)
            )
            if not self.video_writer.isOpened():
                self.video_writer = None
                self.show_toast("RECORDING FAILED")
                return
            self.is_recording = True
            self.show_toast("RECORDING STARTED")
        else:
            self.is_recording = False
            if self.video_writer:
                self.video_writer.release()
                self.video_writer = None
            self.show_toast("RECORDING SAVED")
            print(f"Recording saved as '{self.recording_path}'.")

    def prompt_and_load_background(self):
        """Opens a file dialog to load an image overlay."""
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)

        file_path = filedialog.askopenfilename(
            title="Select Image to Upload",
            filetypes=[("Image Files", "*.png *.jpg *.jpeg *.bmp *.webp")],
        )
        root.destroy()

        if file_path:
            img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
            if img is not None:
                self.raw_overlay = img
                self.overlay_pos = [100, 100]
                self.overlay_scale = 1.0
                self.show_toast("IMAGE LOADED")
                return True
            self.show_toast("FAILED TO LOAD IMAGE")
        return False

    def move_overlay(self, dx, dy):
        """Shifts overlay image position based on delta movement."""
        if self.raw_overlay is not None:
            self.overlay_pos[0] += int(dx)
            self.overlay_pos[1] += int(dy)

    def resize_overlay(self, scale_change):
        """Adjusts image scale factor (clamped between 0.1x and 3.0x)."""
        if self.raw_overlay is not None:
            self.overlay_scale = float(
                np.clip(self.overlay_scale + scale_change, 0.1, 3.0)
            )

    def merge_layers(self, camera_frame, preview_layer):
        """Composites webcam feed, image overlay, persistent filter, canvas, and preview layer."""
        base = camera_frame.copy()

        # 1. Static background image (if loaded)
        if self.background_image is not None:
            base = cv2.addWeighted(base, 0.4, self.background_image, 0.6, 0)

        # 2. Render repositionable/scaled overlay image
        if self.raw_overlay is not None:
            oh, ow = self.raw_overlay.shape[:2]
            new_w = int(ow * self.overlay_scale)
            new_h = int(oh * self.overlay_scale)

            if new_w > 0 and new_h > 0:
                resized_img = cv2.resize(
                    self.raw_overlay,
                    (new_w, new_h),
                    interpolation=cv2.INTER_LINEAR,
                )

                x1, y1 = self.overlay_pos[0], self.overlay_pos[1]
                x2, y2 = x1 + new_w, y1 + new_h

                ix1, iy1 = max(0, x1), max(0, y1)
                ix2, iy2 = min(self.width, x2), min(self.height, y2)

                if ix1 < ix2 and iy1 < iy2:
                    ox1, oy1 = ix1 - x1, iy1 - y1
                    ox2, oy2 = ox1 + (ix2 - ix1), oy1 + (iy2 - iy1)

                    overlay_crop = resized_img[oy1:oy2, ox1:ox2]
                    frame_roi = base[iy1:iy2, ix1:ix2]

                    if (
                        overlay_crop.ndim == 3
                        and overlay_crop.shape[2] == 4
                    ):
                        alpha_channel = overlay_crop[:, :, 3] / 255.0
                        for c in range(3):
                            frame_roi[:, :, c] = (
                                alpha_channel * overlay_crop[:, :, c]
                                + (1.0 - alpha_channel) * frame_roi[:, :, c]
                            )
                        base[iy1:iy2, ix1:ix2] = frame_roi
                    else:
                        blended = cv2.addWeighted(
                            frame_roi,
                            1.0 - self.overlay_alpha,
                            overlay_crop[:, :, :3],
                            self.overlay_alpha,
                            0,
                        )
                        base[iy1:iy2, ix1:ix2] = blended

        # 3. Apply persistent viewport filter over camera + background + uploaded image composite
        if (
            self.active_filter_pts is not None
            and self.active_filter_idx is not None
        ):
            p1, p2, p3, p4 = self.active_filter_pts
            base = self.apply_viewport_filter(
                base, p1, p2, p3, p4, self.active_filter_idx
            )

        # 4. Blend primary drawing canvas
        gray_canvas = cv2.cvtColor(self.canvas, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray_canvas, 1, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)

        bg = cv2.bitwise_and(base, base, mask=mask_inv)
        fg = cv2.bitwise_and(self.canvas, self.canvas, mask=mask)
        combined = cv2.add(bg, fg)

        # 5. Blend dynamic preview layer
        final_output = cv2.addWeighted(combined, 1.0, preview_layer, 1.0, 0)

        if self.is_recording and self.video_writer is not None:
            self.video_writer.write(final_output)

        return final_output

    def save_image(self, filename=None):
        """Saves the static background, uploaded overlay photo, persistent filter, and drawings to disk without camera feed."""
        # 1. Base background frame
        if self.background_image is not None:
            final_composite = self.background_image.copy()
        else:
            final_composite = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        # 2. Composite the raw_overlay photo at its current scale and position
        if self.raw_overlay is not None:
            oh, ow = self.raw_overlay.shape[:2]
            new_w = int(ow * self.overlay_scale)
            new_h = int(oh * self.overlay_scale)

            if new_w > 0 and new_h > 0:
                resized_img = cv2.resize(
                    self.raw_overlay,
                    (new_w, new_h),
                    interpolation=cv2.INTER_LINEAR,
                )

                x1, y1 = self.overlay_pos[0], self.overlay_pos[1]
                x2, y2 = x1 + new_w, y1 + new_h

                ix1, iy1 = max(0, x1), max(0, y1)
                ix2, iy2 = min(self.width, x2), min(self.height, y2)

                if ix1 < ix2 and iy1 < iy2:
                    ox1, oy1 = ix1 - x1, iy1 - y1
                    ox2, oy2 = ox1 + (ix2 - ix1), oy1 + (iy2 - iy1)

                    overlay_crop = resized_img[oy1:oy2, ox1:ox2]
                    frame_roi = final_composite[iy1:iy2, ix1:ix2]

                    # Blend alpha PNG or standard BGR photo
                    if overlay_crop.ndim == 3 and overlay_crop.shape[2] == 4:
                        alpha = overlay_crop[:, :, 3] / 255.0
                        for c in range(3):
                            frame_roi[:, :, c] = (
                                alpha * overlay_crop[:, :, c]
                                + (1.0 - alpha) * frame_roi[:, :, c]
                            )
                        final_composite[iy1:iy2, ix1:ix2] = frame_roi
                    else:
                        blended = cv2.addWeighted(
                            frame_roi,
                            1.0 - self.overlay_alpha,
                            overlay_crop[:, :, :3],
                            self.overlay_alpha,
                            0,
                        )
                        final_composite[iy1:iy2, ix1:ix2] = blended

        # 3. Apply active viewport filter over background + uploaded overlay composite
        if (
            self.active_filter_pts is not None
            and self.active_filter_idx is not None
        ):
            p1, p2, p3, p4 = self.active_filter_pts
            final_composite = self.apply_viewport_filter(
                final_composite, p1, p2, p3, p4, self.active_filter_idx
            )

        # 4. Layer the drawing canvas on top
        gray_canvas = cv2.cvtColor(self.canvas, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray_canvas, 1, 255, cv2.THRESH_BINARY)
        final_composite[mask > 0] = self.canvas[mask > 0]

        # 5. Export to disk
        filename = filename or new_media_path(".png")
        cv2.imwrite(filename, final_composite)
        self.show_toast("ARTWORK SAVED")
        print(f"Artwork saved successfully as '{filename}'.")