import cv2
import numpy as np
import time
import tkinter as tk
from tkinter import filedialog


class CanvasManager:

    def __init__(self, width=1280, height=720):
        self.width = width
        self.height = height

        # Overlay Image Properties
        self.raw_overlay = None  # Original unscaled image
        self.overlay_pos = [100, 100]  # Top-left corner [x, y]
        self.overlay_scale = 1.0  # Scale multiplier (0.1x to 3.0x)
        self.overlay_alpha = 0.7  # Transparency layer blend ratio

        # Primary drawing canvas
        self.canvas = np.zeros((height, width, 3), dtype=np.uint8)

        # Undo history stack
        self.undo_stack = []
        self.max_undo = 50

        # Optional full-size background image
        self.background_image = None

        # Brush state configuration
        self.brush_mode = (
            "SOLID"  # Supported modes: "SOLID", "NEON", "DASHED", "ERASER"
        )
        self.dash_counter = 0

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
        """Saves current canvas state to the undo stack."""
        if len(self.undo_stack) >= self.max_undo:
            self.undo_stack.pop(0)
        self.undo_stack.append(self.canvas.copy())

    def undo(self):
        """Restores the canvas to the previous saved state."""
        if self.undo_stack:
            self.canvas = self.undo_stack.pop()
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
        modes = ["SOLID", "NEON", "DASHED", "ERASER"]
        current_idx = modes.index(self.brush_mode)
        self.brush_mode = modes[(current_idx + 1) % len(modes)]
        self.show_toast(f"BRUSH: {self.brush_mode}")
        return self.brush_mode

    def load_background_image(self, image_path):
        """Loads a static background image stretched across canvas dimensions."""
        img = cv2.imread(image_path)
        if img is not None:
            self.background_image = cv2.resize(img, (self.width, self.height))
            return True
        return False

    def clear(self):
        """Resets the canvas drawing layer."""
        self.save_state()
        self.canvas[:] = 0

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

        filter_mode = filter_idx % len(self.filter_names)
        filter_name = self.filter_names[filter_mode]

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

    def draw_hud(self, frame, shape_mode, color, clear_progress=0.0):
        """Renders HUD status indicators and toast announcements."""
        cv2.putText(
            frame,
            f"Brush Style: {self.brush_mode}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            frame,
            f"Pinch Shape: {shape_mode.upper()}",
            (20, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (200, 200, 200),
            1,
        )

        cv2.rectangle(frame, (20, 80), (60, 110), color, -1)
        cv2.rectangle(frame, (20, 80), (60, 110), (255, 255, 255), 2)

        if time.time() < self.toast_end_time:
            text_size = cv2.getTextSize(
                self.toast_msg, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2
            )[0]
            cx = (self.width - text_size[0]) // 2

            cv2.rectangle(
                frame,
                (cx - 15, 30),
                (cx + text_size[0] + 15, 75),
                (0, 0, 0),
                -1,
            )
            cv2.rectangle(
                frame,
                (cx - 15, 30),
                (cx + text_size[0] + 15, 75),
                (0, 255, 255),
                2,
            )
            cv2.putText(
                frame,
                self.toast_msg,
                (cx, 62),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 255),
                2,
            )

        if clear_progress > 0:
            bar_w = int(200 * clear_progress)
            cv2.rectangle(frame, (20, 130), (220, 145), (50, 50, 50), -1)
            cv2.rectangle(frame, (20, 130), (20 + bar_w, 145), (0, 0, 255), -1)
            cv2.putText(
                frame,
                "HOLD FIST TO CLEAR",
                (20, 125),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 0, 255),
                1,
            )

        if self.is_recording:
            cv2.circle(frame, (self.width - 30, 30), 10, (0, 0, 255), -1)
            cv2.putText(
                frame,
                "REC",
                (self.width - 80, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                2,
            )

    def toggle_recording(self, filename="session.avi"):
        """Toggles video recording on and off."""
        if not self.is_recording:
            fourcc = cv2.VideoWriter_fourcc(*"XVID")
            self.video_writer = cv2.VideoWriter(
                filename, fourcc, 20.0, (self.width, self.height)
            )
            self.is_recording = True
            self.show_toast("RECORDING STARTED")
        else:
            self.is_recording = False
            if self.video_writer:
                self.video_writer.release()
                self.video_writer = None
            self.show_toast("RECORDING SAVED")

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
        """Composites webcam feed, image overlay, drawing canvas, and shape previews."""
        base = camera_frame.copy()

        # 1. Static full-screen background image (if loaded)
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

                    # Handle 4-channel PNG transparency or 3-channel standard image
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

        # 3. Blend primary drawing canvas
        gray_canvas = cv2.cvtColor(self.canvas, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray_canvas, 1, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)

        bg = cv2.bitwise_and(base, base, mask=mask_inv)
        fg = cv2.bitwise_and(self.canvas, self.canvas, mask=mask)
        combined = cv2.add(bg, fg)

        # 4. Blend dynamic preview layer (shape dragging previews)
        final_output = cv2.addWeighted(combined, 1.0, preview_layer, 1.0, 0)

        # 5. Output frame to video recorder if active
        if self.is_recording and self.video_writer is not None:
            self.video_writer.write(final_output)

        return final_output

    def save_image(self, filename="artwork.png"):
        """Saves the static background, uploaded overlay photo, and drawings to disk without camera feed."""
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

        # 3. Layer the drawing canvas on top
        gray_canvas = cv2.cvtColor(self.canvas, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray_canvas, 1, 255, cv2.THRESH_BINARY)
        final_composite[mask > 0] = self.canvas[mask > 0]

        # 4. Export to disk
        cv2.imwrite(filename, final_composite)
        self.show_toast("ARTWORK SAVED")
        print(f"Artwork saved successfully as '{filename}'.")