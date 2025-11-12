import tkinter as tk
import cv2
import numpy as np
from PIL import Image, ImageTk
import threading


class CameraApp:
    def __init__(self, root):
        self.root = root
        self.root.title("🎓 Face Detection: See What Computer Sees")
        self.root.geometry("900x700")
        self.root.configure(bg='#2c3e50')

        self.running = False
        self.cap = None
        self.mode = 'face'
        self.show_features = False  # Toggle to show basic features
        self.frame_skip = 3
        self.frame_count = 0

        # Threading
        self.current_frame = None
        self.frame_lock = threading.Lock()

        # Title
        title = tk.Label(self.root, text="👁️ See How Computer Detects Faces",
                         font=("Arial", 20, "bold"), bg='#2c3e50', fg='#00ff88')
        title.pack(pady=15)

        # Explanation
        self.explanation = tk.Label(self.root,
                                    text="Computer looks for simple patterns like eyes and mouth!",
                                    font=("Arial", 12, "bold"), bg='#2c3e50', fg='#ffffff')
        self.explanation.pack(pady=5)

        # Button frame
        self.button_frame = tk.Frame(self.root, bg='#2c3e50')
        self.button_frame.pack(pady=10)

        # Start button
        self.start_btn = tk.Button(self.button_frame, text="▶️ Start Camera",
                                   command=self.face_love_mode,
                                   font=("Arial", 13, "bold"), bg='#27ae60', fg='white',
                                   width=18, height=2, relief=tk.RAISED, bd=4, cursor='hand2')
        self.start_btn.grid(row=0, column=0, padx=10, pady=5)

        # Toggle features button
        self.feature_btn = tk.Button(self.button_frame, text="🔍 Show Basic Features",
                                     command=self.toggle_features,
                                     font=("Arial", 13, "bold"), bg='#3498db', fg='white',
                                     width=20, height=2, relief=tk.RAISED, bd=4, cursor='hand2',
                                     state=tk.DISABLED)
        self.feature_btn.grid(row=0, column=1, padx=10, pady=5)

        # Stop button
        self.stop_button = tk.Button(self.button_frame, text="⏹️ Stop Camera",
                                     command=self.stop_camera,
                                     font=("Arial", 13, "bold"), bg='#e74c3c', fg='white',
                                     width=18, height=2, relief=tk.RAISED, bd=4, cursor='hand2')
        self.stop_button.grid(row=0, column=2, padx=10, pady=5)

        # Video label
        self.video_label = tk.Label(self.root, bg='black')
        self.video_label.pack(pady=10)

        # Info label
        self.info_label = tk.Label(self.root, text="",
                                   font=("Arial", 12, "bold"), bg='#2c3e50', fg='#f39c12',
                                   wraplength=850, justify=tk.LEFT)
        self.info_label.pack(pady=10)

        # Load cascades
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
        self.mouth_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_smile.xml")

        self.last_faces = []
        self.frame_width = 640
        self.frame_height = 480

    def toggle_features(self):
        """Toggle between showing features and normal mode"""
        self.show_features = not self.show_features

        if self.show_features:
            self.feature_btn.config(text="😊 Show Normal Mode", bg='#e67e22')
            self.explanation.config(text="👁️ BASIC FEATURES MODE: See the simple patterns computer looks for!")
            self.info_label.config(fg='#00ff88')
        else:
            self.feature_btn.config(text="🔍 Show Basic Features", bg='#3498db')
            self.explanation.config(text="😊 NORMAL MODE: See complete face detection!")
            self.info_label.config(fg='#f39c12')

    def face_love_mode(self):
        if not self.running:
            self.running = True
            self.start_btn.config(state=tk.DISABLED)
            self.feature_btn.config(state=tk.NORMAL)

            # Initialize webcam
            self.cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

            if not self.cap.isOpened():
                self.cap = cv2.VideoCapture(1)

            # Optimize settings
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
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

    def stop_camera(self):
        self.running = False
        if self.cap:
            self.cap.release()
        self.video_label.config(image='')
        self.last_faces = []
        self.start_btn.config(state=tk.NORMAL)
        self.feature_btn.config(state=tk.DISABLED)
        self.info_label.config(text="")

    def update_frame(self):
        if self.running:
            with self.frame_lock:
                if self.current_frame is not None:
                    frame = self.current_frame.copy()
                else:
                    frame = None

            if frame is not None:
                if self.show_features:
                    # Show basic features mode
                    frame, info = self.detect_basic_features(frame)
                else:
                    # Normal face detection mode
                    frame, info = self.detect_faces_and_draw_love(frame)

                self.info_label.config(text=info)

                # Display
                img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(img)
                imgtk = ImageTk.PhotoImage(image=img)
                self.video_label.imgtk = imgtk
                self.video_label.configure(image=imgtk)

            self.root.after(16, self.update_frame)

    def detect_basic_features(self, frame):
        """Show basic features that computer looks for"""
        self.frame_count += 1
        info = "🔍 BASIC FEATURES MODE:\n"

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect eyes (every frame for educational purposes)
        eyes = self.eye_cascade.detectMultiScale(gray, 1.1, 5, minSize=(20, 20))
        info += f"👁️ Found {len(eyes)} EYE-LIKE rectangles (dark areas)\n"

        for (x, y, w, h) in eyes:
            # Draw yellow rectangle
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 3)
            # Label
            cv2.putText(frame, "EYE", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 255, 255), 2, cv2.LINE_AA)
            # Draw center dot
            center_x = x + w // 2
            center_y = y + h // 2
            cv2.circle(frame, (center_x, center_y), 4, (0, 255, 255), -1)

        # Detect mouth areas
        mouths = self.mouth_cascade.detectMultiScale(gray, 1.3, 5, minSize=(30, 20))
        info += f"👄 Found {len(mouths)} MOUTH-LIKE rectangles\n"

        for (x, y, w, h) in mouths:
            # Draw cyan rectangle
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 3)
            cv2.putText(frame, "MOUTH", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255, 255, 0), 2, cv2.LINE_AA)

        # Also show potential face regions lightly
        if self.frame_count % self.frame_skip == 0:
            small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
            small_gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)

            faces = self.face_cascade.detectMultiScale(
                small_gray, scaleFactor=1.1, minNeighbors=3, minSize=(20, 20)
            )

            if len(faces) > 0:
                self.last_faces = [(x * 2, y * 2, w * 2, h * 2) for (x, y, w, h) in faces]

        # Draw face regions as dashed lines
        for (x, y, w, h) in self.last_faces:
            # Dashed rectangle (draw corners)
            corner_len = 20
            # Top-left
            cv2.line(frame, (x, y), (x + corner_len, y), (100, 100, 100), 2)
            cv2.line(frame, (x, y), (x, y + corner_len), (100, 100, 100), 2)
            # Top-right
            cv2.line(frame, (x + w, y), (x + w - corner_len, y), (100, 100, 100), 2)
            cv2.line(frame, (x + w, y), (x + w, y + corner_len), (100, 100, 100), 2)
            # Bottom-left
            cv2.line(frame, (x, y + h), (x + corner_len, y + h), (100, 100, 100), 2)
            cv2.line(frame, (x, y + h), (x, y + h - corner_len), (100, 100, 100), 2)
            # Bottom-right
            cv2.line(frame, (x + w, y + h), (x + w - corner_len, y + h), (100, 100, 100), 2)
            cv2.line(frame, (x + w, y + h), (x + w, y + h - corner_len), (100, 100, 100), 2)

            cv2.putText(frame, "FACE AREA", (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (100, 100, 100), 2, cv2.LINE_AA)

        info += f"\n📐 The computer combines these simple rectangles to find faces!"

        return frame, info

    def detect_faces_and_draw_love(self, frame):
        """Normal face detection with hearts"""
        self.frame_count += 1
        info = "😊 NORMAL MODE: Complete face detection\n"

        # Only detect faces every N frames
        if self.frame_count % self.frame_skip == 0:
            small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
            gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)

            faces = self.face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=3, minSize=(20, 20),
                flags=cv2.CASCADE_SCALE_IMAGE
            )

            if len(faces) > 0:
                self.last_faces = [(x * 2, y * 2, w * 2, h * 2) for (x, y, w, h) in faces]

        info += f"✅ Found {len(self.last_faces)} face(s)\n"

        # Draw hearts and text
        for (x, y, w, h) in self.last_faces:
            # Draw heart (ellipse)
            center = (x + w // 2, y + h // 2)
            axes = (w // 2, h // 2)
            cv2.ellipse(frame, center, axes, 0, 0, 360, (0, 0, 255), 3)

            # Add text
            cv2.putText(frame, "I love Science", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, (0, 0, 255), 2, cv2.LINE_AA)

        info += "\n💡 Click 'Show Basic Features' to see what computer looks for!"

        return frame, info


# Run the app
if __name__ == "__main__":
    root = tk.Tk()
    app = CameraApp(root)
    root.mainloop()