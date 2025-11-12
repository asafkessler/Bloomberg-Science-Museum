import tkinter as tk
import cv2
import numpy as np
import random
from PIL import Image, ImageTk
import threading


class CameraApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Camera Filters - Pixel Inspector")
        self.root.geometry("900x700")
        self.root.configure(bg='#f0f0f0')

        self.current_filter = 'bw'
        self.cap = None
        self.running = False
        self.frozen = False
        self.frozen_frame = None

        # Threading for better performance
        self.current_frame = None
        self.frame_lock = threading.Lock()

        # Pixel info
        self.pixel_info_var = tk.StringVar()
        self.pixel_info_var.set("Click on image to see pixel values")

        # Title
        title_label = tk.Label(self.root, text="🎥 Camera Filters & Pixel Inspector",
                               font=("Arial", 18, "bold"), bg='#f0f0f0', fg='#333')
        title_label.pack(pady=10)

        # Filter buttons
        self.button_frame = tk.Frame(self.root, bg='#f0f0f0')
        self.button_frame.pack(pady=10)

        filters = [
            ("⚫ Black & White", 'bw', '#555555'),
            ("🔴 Red Channel", 'red', '#ff4444'),
            ("🟢 Green Channel", 'green', '#44ff44'),
            ("🔵 Blue Channel", 'blue', '#4444ff')
        ]

        for label, mode, color in filters:
            btn = tk.Button(self.button_frame, text=label, width=15, height=2,
                            font=("Arial", 11, "bold"), bg=color, fg='white',
                            command=lambda m=mode: self.set_filter(m),
                            relief=tk.RAISED, bd=3, cursor='hand2')
            btn.pack(side=tk.LEFT, padx=5)

        # Control buttons
        control_frame = tk.Frame(self.root, bg='#f0f0f0')
        control_frame.pack(pady=10)

        self.freeze_button = tk.Button(control_frame, text="❄️ Freeze Frame", width=15, height=2,
                                       font=("Arial", 11, "bold"), bg='#00aaff', fg='white',
                                       command=self.toggle_freeze, relief=tk.RAISED, bd=3, cursor='hand2')
        self.freeze_button.pack(side=tk.LEFT, padx=5)

        self.stop_button = tk.Button(control_frame, text="⏹️ Stop Camera", width=15, height=2,
                                     font=("Arial", 11, "bold"), bg='#ff3333', fg='white',
                                     command=self.stop_camera, relief=tk.RAISED, bd=3, cursor='hand2')
        self.stop_button.pack(side=tk.LEFT, padx=5)

        # Video display
        video_frame = tk.Frame(self.root, bg='#f0f0f0')
        video_frame.pack(pady=10)

        self.video_label = tk.Label(video_frame, bg='black')
        self.video_label.pack()
        self.video_label.bind('<Button-1>', self.on_image_click)

        # Pixel info display
        info_frame = tk.Frame(self.root, bg='white', relief=tk.SUNKEN, bd=2)
        info_frame.pack(pady=10, padx=20, fill=tk.X)

        pixel_label = tk.Label(info_frame, textvariable=self.pixel_info_var,
                               font=("Courier", 12, "bold"), bg='white', fg='#333', pady=10)
        pixel_label.pack()

        self.set_filter('bw')

    def set_filter(self, mode):
        self.current_filter = mode
        if not self.running:
            self.running = True

            # Initialize webcam with optimizations
            self.cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

            if not self.cap.isOpened():
                self.cap = cv2.VideoCapture(1)

            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

            # Start capture thread
            self.capture_thread = threading.Thread(target=self.capture_frames, daemon=True)
            self.capture_thread.start()

            self.update_frame()

    def capture_frames(self):
        """Separate thread for capturing frames"""
        while self.running:
            if self.cap and self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret:
                    with self.frame_lock:
                        self.current_frame = frame

    def toggle_freeze(self):
        self.frozen = not self.frozen
        if self.frozen:
            self.freeze_button.config(text="▶️ Unfreeze", bg='#ffaa00')
            with self.frame_lock:
                if self.current_frame is not None:
                    self.frozen_frame = self.current_frame.copy()
        else:
            self.freeze_button.config(text="❄️ Freeze Frame", bg='#00aaff')
            self.frozen_frame = None

    def stop_camera(self):
        self.running = False
        self.frozen = False
        self.frozen_frame = None
        if self.cap:
            self.cap.release()
        self.video_label.config(image='')
        self.pixel_info_var.set("Camera stopped")

    def update_frame(self):
        if self.running:
            if self.frozen and self.frozen_frame is not None:
                frame = self.frozen_frame
            else:
                with self.frame_lock:
                    if self.current_frame is not None:
                        frame = self.current_frame.copy()
                    else:
                        frame = None

            if frame is not None:
                filtered = self.apply_filter(frame, self.current_filter)

                # Store for pixel inspection
                self.display_frame = filtered.copy()

                # Convert to RGB for display
                img = cv2.cvtColor(filtered, cv2.COLOR_BGR2RGB)

                img = Image.fromarray(img)
                imgtk = ImageTk.PhotoImage(image=img)
                self.video_label.imgtk = imgtk
                self.video_label.configure(image=imgtk)

            self.root.after(16, self.update_frame)  # ~60fps

    def apply_filter(self, frame, mode):
        if mode == 'bw':
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Convert back to BGR so pixel values show as RGB
            return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        elif mode == 'red':
            filtered = frame.copy()
            filtered[:, :, 0] = 0  # B
            filtered[:, :, 1] = 0  # G
            return filtered
        elif mode == 'green':
            filtered = frame.copy()
            filtered[:, :, 0] = 0  # B
            filtered[:, :, 2] = 0  # R
            return filtered
        elif mode == 'blue':
            filtered = frame.copy()
            filtered[:, :, 1] = 0  # G
            filtered[:, :, 2] = 0  # R
            return filtered
        return frame

    def on_image_click(self, event):
        """Handle click on image to show pixel values"""
        if not hasattr(self, 'display_frame') or self.display_frame is None:
            return

        x, y = event.x, event.y

        # Check bounds
        if y >= self.display_frame.shape[0] or x >= self.display_frame.shape[1]:
            return

        # Get pixel value (always BGR/RGB now)
        b, g, r = self.display_frame[y, x]
        info = f"Position: ({x}, {y}) | RGB: (R={r}, G={g}, B={b}) | Hex: #{r:02X}{g:02X}{b:02X}"

        self.pixel_info_var.set(info)


# Run the app
if __name__ == "__main__":
    root = tk.Tk()
    app = CameraApp(root)
    root.mainloop()