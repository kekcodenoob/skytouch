import cv2
import numpy as np

class CanvasManager:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
        self.filter_names = [
            "Invert Gray", 
            "Edge Detect", 
            "Sepia Tone", 
            "Thermal Heat",
            "Pixelate",
            "Cartoon",
            "BGR Channel Swap"
        ]

    def clear(self):
        self.canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)

    def draw_line(self, pt1, pt2, color, thickness=8):
        cv2.line(self.canvas, pt1, pt2, color, thickness)

    def draw_shape_stamp(self, mode, pt, color):
        if mode == "circle":
            cv2.circle(self.canvas, pt, 25, color, -1)
        elif mode == "rectangle":
            cv2.rectangle(self.canvas, (pt[0] - 25, pt[1] - 25), (pt[0] + 25, pt[1] + 25), color, -1)

    def commit_drag_shape(self, mode, start_pt, end_pt, color):
        x1, y1 = start_pt
        x2, y2 = end_pt
        if mode == "rectangle":
            cv2.rectangle(self.canvas, (x1, y1), (x2, y2), color, -1)
        else:
            center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
            axes_x, axes_y = abs(x2 - x1) // 2, abs(y2 - y1) // 2
            if axes_x > 0 and axes_y > 0:
                cv2.ellipse(self.canvas, (center_x, center_y), (axes_x, axes_y), 0, 0, 360, color, -1)

    def _generate_filter_effect(self, frame, filter_index):
        """Generates different full-frame video filter effects."""
        mode = filter_index % len(self.filter_names)

        if mode == 0:  # Inverted Grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            filtered = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            return cv2.bitwise_not(filtered)

        elif mode == 1:  # Edge Detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

        elif mode == 2:  # Sepia Tone
            sepia_kernel = np.array([
                [0.272, 0.534, 0.131],
                [0.349, 0.686, 0.168],
                [0.393, 0.769, 0.189]
            ])
            return cv2.transform(frame, sepia_kernel)

        elif mode == 3:  # Thermal Look
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            return cv2.applyColorMap(gray, cv2.COLORMAP_JET)

        elif mode == 4:  # Pixelate
            pixel_size = 16
            h, w, _ = frame.shape
            temp = cv2.resize(frame, (w // pixel_size, h // pixel_size), interpolation=cv2.INTER_LINEAR)
            return cv2.resize(temp, (w, h), interpolation=cv2.INTER_NEAREST)

        elif mode == 5:  # Cartoon
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_blur = cv2.medianBlur(gray, 7)
            edges = cv2.adaptiveThreshold(gray_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                          cv2.THRESH_BINARY, 9, 9)
            color = cv2.bilateralFilter(frame, d=9, sigmaColor=300, sigmaSpace=300)
            return cv2.bitwise_and(color, color, mask=edges)

        elif mode == 6:  # BGR Channel Swap
            b, g, r = cv2.split(frame)
            return cv2.merge([r, b, g])

        return frame

    def apply_viewport_filter(self, frame, p1, p2, p3, p4, filter_index):
        """Applies the selected filter inside the region framed by 2 hands."""
        dst_pts = np.array([p1, p2, p4, p3], dtype=np.int32)

        # Mask of hand region
        mask = np.zeros((self.height, self.width), dtype=np.uint8)
        cv2.fillConvexPoly(mask, dst_pts, 255)

        # Generate selected filter
        filtered_frame = self._generate_filter_effect(frame, filter_index)

        # Apply filter only inside quad
        mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        frame_with_filter = np.where(mask_3ch == 255, filtered_frame, frame)

        # Draw frame outline
        cv2.polylines(frame_with_filter, [dst_pts], isClosed=True, color=(0, 255, 255), thickness=2)

        # Label active filter name over viewport
        filter_name = self.filter_names[filter_index % len(self.filter_names)]
        cv2.putText(frame_with_filter, f"VIEWPORT: {filter_name}", (p1[0], max(30, p1[1] - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        return frame_with_filter

    def merge_layers(self, frame, preview_layer):
        gray_canvas = cv2.cvtColor(self.canvas, cv2.COLOR_BGR2GRAY)
        _, inv_canvas = cv2.threshold(gray_canvas, 20, 255, cv2.THRESH_BINARY_INV)
        inv_canvas = cv2.cvtColor(inv_canvas, cv2.COLOR_GRAY2BGR)

        frame = cv2.bitwise_and(frame, inv_canvas)
        frame = cv2.bitwise_or(frame, self.canvas)
        return cv2.addWeighted(frame, 0.7, preview_layer, 0.3, 0)

    def draw_hud(self, frame, mode_str, color, clear_progress=0.0):
        cv2.putText(frame, f"Pointer Mode: {mode_str.upper()}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, "1 Hand: Draw/Shape | 2 Hands: Perspective Filter Viewport", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.rectangle(frame, (10, 85), (110, 125), color, -1)
        cv2.rectangle(frame, (10, 85), (110, 125), (255, 255, 255), 2)

        # Restored status tip for canvas clearing
        if clear_progress > 0:
            pct = int(clear_progress * 100)
            cv2.putText(frame, f"CLEARING CANVAS... {pct}%", (10, self.height - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)