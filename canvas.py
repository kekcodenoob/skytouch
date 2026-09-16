import cv2
import numpy as np
import time

class CanvasManager:
    def __init__(self, width=1280, height=720):
        self.width = width
        self.height = height
        
        # Primary drawing canvas
        self.canvas = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Undo history stack
        self.undo_stack = []
        self.max_undo = 50
        
        # Optional image editing layer
        self.background_image = None
        
        # Brush state configuration
        self.brush_mode = "SOLID"  # Supported: "SOLID", "NEON", "DASHED", "ERASER"
        self.dash_counter = 0
        
        # Extended list including filters: PIXELATE, THERMAL, and COLOR_INVERT
        self.filter_names = [
            "GRAYSCALE", 
            "INVERT", 
            "SEPIA", 
            "SOBEL", 
            "BLUR", 
            "PIXELATE", 
            "THERMAL", 
            "COLOR_INVERT"
        ]

        # On-screen toast notifications
        self.toast_msg = ""
        self.toast_end_time = 0.0

        # Recording video stream setup
        self.video_writer = None
        self.is_recording = False

    def save_state(self):
        """Saves current canvas state to undo stack."""
        if len(self.undo_stack) >= self.max_undo:
            self.undo_stack.pop(0)  # Remove oldest state
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
        """Triggers an on-screen announcement message."""
        self.toast_msg = text
        self.toast_end_time = time.time() + duration

    def cycle_brush_mode(self):
        """Cycles through available brush rendering styles."""
        modes = ["SOLID", "NEON", "DASHED", "ERASER"]
        current_idx = modes.index(self.brush_mode)
        self.brush_mode = modes[(current_idx + 1) % len(modes)]
        self.show_toast(f"BRUSH: {self.brush_mode}")
        return self.brush_mode

    def load_background_image(self, image_path):
        """Loads an image to edit/draw on top of."""
        img = cv2.imread(image_path)
        if img is not None:
            self.background_image = cv2.resize(img, (self.width, self.height))
            return True
        return False

    def clear(self):
        """Resets the canvas after saving state for undo."""
        self.save_state()
        self.canvas[:] = 0

    def draw_line(self, p1, p2, color, thickness=6):
        """Draws freehand lines on the canvas layer using the active brush mode."""
        if self.brush_mode == "ERASER":
            cv2.line(self.canvas, p1, p2, (0, 0, 0), thickness * 4)

        elif self.brush_mode == "NEON":
            # Outer glow
            cv2.line(self.canvas, p1, p2, color, thickness * 2)
            # Bright inner core
            cv2.line(self.canvas, p1, p2, (255, 255, 255), max(2, thickness // 3))

        elif self.brush_mode == "DASHED":
            self.dash_counter += 1
            if (self.dash_counter // 4) % 2 == 0:
                cv2.line(self.canvas, p1, p2, color, thickness)

        else:  # SOLID
            cv2.line(self.canvas, p1, p2, color, thickness)

    def commit_drag_shape(self, mode, p1, p2, color):
        """Commits dragged bounding shapes directly onto the canvas."""
        if not p1 or not p2:
            return
        
        # Save state before applying shape permanently
        self.save_state()
        
        x1, y1 = p1
        x2, y2 = p2

        if mode == "rectangle":
            cv2.rectangle(self.canvas, (x1, y1), (x2, y2), color, 3)
        else:
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            ax, ay = abs(x2 - x1) // 2, abs(y2 - y1) // 2
            if ax > 0 and ay > 0:
                cv2.ellipse(self.canvas, (cx, cy), (ax, ay), 0, 0, 360, color, 3)

    def apply_viewport_filter(self, frame, p1, p2, p3, p4, filter_idx):
        """Applies image processing filters within a bounded viewport polygon."""
        pts = np.array([p1, p2, p4, p3], dtype=np.int32)
        x, y, w, h = cv2.boundingRect(pts)
        
        # Clamp ROI to image bounds to avoid slice indexing faults
        x = max(0, x)
        y = max(0, y)
        w = min(self.width - x, w)
        h = min(self.height - y, h)

        if w <= 0 or h <= 0:
            return frame

        roi = frame[y:y+h, x:x+w]
        if roi.size == 0:
            return frame

        filter_mode = filter_idx % len(self.filter_names)
        filter_name = self.filter_names[filter_mode]

        if filter_name == "GRAYSCALE":
            filtered_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            filtered_roi = cv2.cvtColor(filtered_roi, cv2.COLOR_GRAY2BGR)

        elif filter_name == "INVERT" or filter_name == "COLOR_INVERT":
            filtered_roi = cv2.bitwise_not(roi)

        elif filter_name == "SEPIA":
            sepia_kernel = np.array([
                [0.272, 0.534, 0.131],
                [0.349, 0.686, 0.168],
                [0.393, 0.769, 0.189]
            ])
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
            small = cv2.resize(roi, (temp_w, temp_h), interpolation=cv2.INTER_LINEAR)
            filtered_roi = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)

        elif filter_name == "THERMAL":
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            filtered_roi = cv2.applyColorMap(gray, cv2.COLORMAP_JET)

        else:
            filtered_roi = roi

        mask = np.zeros((h, w), dtype=np.uint8)
        shifted_pts = pts - np.array([x, y])
        cv2.fillConvexPoly(mask, shifted_pts, 255)

        mask_inv = cv2.bitwise_not(mask)
        img1_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
        img2_fg = cv2.bitwise_and(filtered_roi, filtered_roi, mask=mask)

        frame[y:y+h, x:x+w] = cv2.add(img1_bg, img2_fg)
        cv2.polylines(frame, [pts], True, (0, 255, 255), 2)

        # Viewport Caption directly above the bounded region
        top_y = max(25, min(p1[1], p2[1], p3[1], p4[1]) - 10)
        min_x = max(10, min(p1[0], p2[0], p3[0], p4[0]))
        
        cv2.putText(frame, f"FILTER: {filter_name}", (min_x, top_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        return frame

    def merge_layers(self, camera_frame, preview_layer):
        """Blends camera feed, optional background image, canvas, and UI previews."""
        base = camera_frame.copy()

        if self.background_image is not None:
            base = cv2.addWeighted(base, 0.4, self.background_image, 0.6, 0)

        gray_canvas = cv2.cvtColor(self.canvas, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray_canvas, 1, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)

        bg = cv2.bitwise_and(base, base, mask=mask_inv)
        fg = cv2.bitwise_and(self.canvas, self.canvas, mask=mask)
        combined = cv2.add(bg, fg)

        final_output = cv2.addWeighted(combined, 1.0, preview_layer, 1.0, 0)

        if self.is_recording and self.video_writer is not None:
            self.video_writer.write(final_output)

        return final_output

    def draw_hud(self, frame, shape_mode, color, clear_progress=0.0):
        """Renders HUD status indicators and toast announcements."""
        # Top Left Status HUD
        cv2.putText(frame, f"Brush Style: {self.brush_mode}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Pinch Shape: {shape_mode.upper()}", (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        cv2.rectangle(frame, (20, 80), (60, 110), color, -1)
        cv2.rectangle(frame, (20, 80), (60, 110), (255, 255, 255), 2)

        # On-Screen Toast Announcement Banner
        if time.time() < self.toast_end_time:
            text_size = cv2.getTextSize(self.toast_msg, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0]
            cx = (self.width - text_size[0]) // 2
            
            # Dark backing pill
            cv2.rectangle(frame, (cx - 15, 30), (cx + text_size[0] + 15, 75), (0, 0, 0), -1)
            cv2.rectangle(frame, (cx - 15, 30), (cx + text_size[0] + 15, 75), (0, 255, 255), 2)
            cv2.putText(frame, self.toast_msg, (cx, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

        # Clear Canvas Meter
        if clear_progress > 0:
            bar_w = int(200 * clear_progress)
            cv2.rectangle(frame, (20, 130), (220, 145), (50, 50, 50), -1)
            cv2.rectangle(frame, (20, 130), (20 + bar_w, 145), (0, 0, 255), -1)
            cv2.putText(frame, "HOLD FIST TO CLEAR", (20, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

        # Recording Badge
        if self.is_recording:
            cv2.circle(frame, (self.width - 30, 30), 10, (0, 0, 255), -1)
            cv2.putText(frame, "REC", (self.width - 80, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    def toggle_recording(self, filename="session.avi"):
        if not self.is_recording:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            self.video_writer = cv2.VideoWriter(filename, fourcc, 20.0, (self.width, self.height))
            self.is_recording = True
        else:
            self.is_recording = False
            if self.video_writer:
                self.video_writer.release()
                self.video_writer = None

    def save_image(self, filename="artwork.png"):
        cv2.imwrite(filename, self.canvas)