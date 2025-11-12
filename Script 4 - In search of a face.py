import tkinter as tk
import cv2
from PIL import Image, ImageTk
import numpy as np
import threading


class FaceDetectionEducational:
    def __init__(self, root):
        self.root = root
        self.root.title("👁️ Face Detection: From Features to Faces")

        # Get screen size
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        window_width = int(screen_width * 0.9)
        window_height = int(screen_height * 0.9)

        self.root.geometry(f"{window_width}x{window_height}")
        self.root.configure(bg='#1a1a2e')

        # Variables
        self.cap = None
        self.running = False
        self.current_frame = None
        self.frame_lock = threading.Lock()

        # Detection mode: 'features', 'components', 'full'
        self.detection_mode = tk.StringVar(value='features')

        # Load OpenCV cascades
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        self.mouth_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

        # Calculate canvas size
        self.canvas_width = window_width - 150
        self.canvas_height = int(window_height * 0.75)

        # Create UI
        self.create_ui()

    def create_ui(self):
        """Create user interface"""
        # Title
        title = tk.Label(self.root,
                         text="🎓 Face Detection Education: From Basic Features to Complete Faces",
                         font=("Arial", 22, "bold"), bg='#1a1a2e', fg='#00ff88')
        title.pack(pady=15)

        # Explanation
        explanation = tk.Label(self.root,
                               text="Learn how computers detect faces by finding basic features first!",
                               font=("Arial", 13), bg='#1a1a2e', fg='#ffffff')
        explanation.pack(pady=5)

        # Main container
        main_container = tk.Frame(self.root, bg='#1a1a2e')
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # Left side - Video display
        self.create_video_panel(main_container)

        # Right side - Controls
        self.create_control_panel(main_container)

    def create_video_panel(self, parent):
        """Create video display panel"""
        video_frame = tk.Frame(parent, bg='#16213e', relief=tk.RAISED, bd=3)
        video_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)

        # Status
        self.status_label = tk.Label(video_frame,
                                     text="🎥 Click 'Start Camera' to begin",
                                     font=("Arial", 13, "bold"),
                                     bg='#16213e', fg='#00ff88')
        self.status_label.pack(pady=10)

        # Canvas
        self.canvas = tk.Canvas(video_frame, width=self.canvas_width, height=self.canvas_height,
                                bg='black', highlightthickness=2, highlightbackground='#00ff88')
        self.canvas.pack(pady=10)

        # Info display
        self.info_label = tk.Label(video_frame,
                                   text="",
                                   font=("Courier", 11, "bold"),
                                   bg='#16213e', fg='#ffaa00',
                                   justify=tk.LEFT)
        self.info_label.pack(pady=10)

    def create_control_panel(self, parent):
        """Create control panel"""
        control_frame = tk.Frame(parent, bg='#16213e', relief=tk.RAISED, bd=3, width=400)
        control_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=5)
        control_frame.pack_propagate(False)

        # Title
        title = tk.Label(control_frame, text="🎛️ Detection Controls",
                         font=("Arial", 16, "bold"), bg='#16213e', fg='#00ff88')
        title.pack(pady=15)

        # Camera controls
        cam_frame = tk.LabelFrame(control_frame, text="📹 Camera",
                                  font=("Arial", 13, "bold"),
                                  bg='#16213e', fg='#ffffff', relief=tk.RAISED, bd=2)
        cam_frame.pack(pady=10, padx=15, fill=tk.X)

        self.start_btn = tk.Button(cam_frame, text="▶️ Start Camera",
                                   command=self.toggle_camera,
                                   font=("Arial", 12, "bold"),
                                   bg='#27ae60', fg='white',
                                   height=2, relief=tk.RAISED, bd=3, cursor='hand2')
        self.start_btn.pack(pady=10, padx=10, fill=tk.X)

        # Detection modes
        mode_frame = tk.LabelFrame(control_frame, text="🔍 Detection Stages",
                                   font=("Arial", 13, "bold"),
                                   bg='#16213e', fg='#ffffff', relief=tk.RAISED, bd=2)
        mode_frame.pack(pady=10, padx=15, fill=tk.X)

        modes = [
            ('features', '📐 Stage 1: Basic Features\n(Eyes, Nose, Mouth areas)'),
            ('components', '🎯 Stage 2: Face Components\n(Detailed features)'),
            ('full', '😊 Stage 3: Complete Face\n(Full detection)')
        ]

        for value, text in modes:
            rb = tk.Radiobutton(mode_frame, text=text,
                                variable=self.detection_mode, value=value,
                                font=("Arial", 11, "bold"),
                                bg='#16213e', fg='#ffffff',
                                selectcolor='#0f3460',
                                activebackground='#16213e',
                                activeforeground='#00ff88',
                                cursor='hand2',
                                justify=tk.LEFT)
            rb.pack(anchor='w', pady=8, padx=15)

        # Explanation panel
        explain_frame = tk.LabelFrame(control_frame, text="📚 How It Works",
                                      font=("Arial", 13, "bold"),
                                      bg='#16213e', fg='#ffffff', relief=tk.RAISED, bd=2)
        explain_frame.pack(pady=10, padx=15, fill=tk.BOTH, expand=True)

        explanation_text = """
🔹 STAGE 1: Basic Features
The algorithm looks for simple 
patterns:
• Dark rectangles (eyes)
• Light rectangles (nose bridge)
• Dark areas (mouth)

These are called "Haar Features"

🔹 STAGE 2: Components
Combines features to find:
• Eye pairs
• Nose position
• Mouth location

🔹 STAGE 3: Full Face
Uses all components to:
• Confirm it's a face
• Draw bounding box
• Calculate confidence

This is the Viola-Jones 
algorithm from 2001!
        """

        explain_label = tk.Label(explain_frame, text=explanation_text,
                                 font=("Arial", 10),
                                 bg='#16213e', fg='#ecf0f1',
                                 justify=tk.LEFT)
        explain_label.pack(pady=10, padx=10)

    def toggle_camera(self):
        """Start/stop camera"""
        if not self.running:
            self.running = True
            self.start_btn.config(text="⏹️ Stop Camera", bg='#c0392b')
            self.status_label.config(text="🎥 Camera active - Select detection stage")
            threading.Thread(target=self.run_camera, daemon=True).start()
        else:
            self.stop_camera()

    def run_camera(self):
        """Run camera with threading"""
        self.cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(1)

        if not self.cap.isOpened():
            self.status_label.config(text="❌ Cannot open webcam!")
            self.running = False
            return

        # Optimize settings
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # Capture thread
        def capture_thread():
            while self.running:
                if self.cap and self.cap.isOpened():
                    ret, frame = self.cap.read()
                    if ret:
                        with self.frame_lock:
                            self.current_frame = frame

        threading.Thread(target=capture_thread, daemon=True).start()
        self.update_display()

    def update_display(self):
        """Update display with detection"""
        if self.running:
            with self.frame_lock:
                if self.current_frame is not None:
                    frame = self.current_frame.copy()
                else:
                    frame = None

            if frame is not None:
                # Process based on mode
                mode = self.detection_mode.get()
                processed_frame, info_text = self.process_frame(frame, mode)

                # Display
                self.display_frame(processed_frame)
                self.info_label.config(text=info_text)

            self.root.after(33, self.update_display)

    def process_frame(self, frame, mode):
        """Process frame based on detection mode"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        info_text = ""

        if mode == 'features':
            # Stage 1: Basic Haar features
            info_text = "🔍 STAGE 1: Looking for basic features...\n"

            # Detect eyes
            eyes = self.eye_cascade.detectMultiScale(gray, 1.1, 5, minSize=(20, 20))
            info_text += f"👁️ Found {len(eyes)} eye-like features\n"

            for (x, y, w, h) in eyes:
                # Draw as simple rectangle
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)
                cv2.putText(frame, "EYE?", (x, y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

            # Try to detect mouth
            mouths = self.mouth_cascade.detectMultiScale(gray, 1.5, 5, minSize=(30, 20))
            info_text += f"👄 Found {len(mouths)} mouth-like features"

            for (x, y, w, h) in mouths:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
                cv2.putText(frame, "MOUTH?", (x, y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        elif mode == 'components':
            # Stage 2: Face components
            info_text = "🎯 STAGE 2: Finding face components...\n"

            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
            info_text += f"🔲 Found {len(faces)} face region(s)\n"

            for (fx, fy, fw, fh) in faces:
                # Draw face region lightly
                cv2.rectangle(frame, (fx, fy), (fx + fw, fy + fh), (100, 100, 100), 1)

                # Look for components within face
                roi_gray = gray[fy:fy + fh, fx:fx + fw]
                roi_color = frame[fy:fy + fh, fx:fx + fw]

                # Eyes
                eyes = self.eye_cascade.detectMultiScale(roi_gray, 1.1, 3)
                for (ex, ey, ew, eh) in eyes:
                    cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
                    cv2.putText(roi_color, "EYE", (ex, ey - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

                info_text += f"  ➜ {len(eyes)} eyes detected\n"

                # Mouth in lower half
                mouth_roi_gray = roi_gray[fh // 2:, :]
                mouth_roi_color = roi_color[fh // 2:, :]
                mouths = self.mouth_cascade.detectMultiScale(mouth_roi_gray, 1.3, 3)

                for (mx, my, mw, mh) in mouths:
                    cv2.rectangle(mouth_roi_color, (mx, my), (mx + mw, my + mh), (255, 0, 255), 2)
                    cv2.putText(mouth_roi_color, "MOUTH", (mx, my - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)

                info_text += f"  ➜ {len(mouths)} mouth detected"

        elif mode == 'full':
            # Stage 3: Full face detection
            info_text = "😊 STAGE 3: Complete face detection!\n"

            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
            info_text += f"✅ Detected {len(faces)} face(s)\n"

            for i, (x, y, w, h) in enumerate(faces):
                # Draw full face box
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
                cv2.putText(frame, f"FACE #{i + 1}", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                # Add confidence indicator
                center_x = x + w // 2
                center_y = y + h // 2
                cv2.circle(frame, (center_x, center_y), 5, (0, 255, 0), -1)

                info_text += f"\nFace #{i + 1}:\n"
                info_text += f"  Position: ({x}, {y})\n"
                info_text += f"  Size: {w}×{h} pixels"

        return frame, info_text

    def display_frame(self, frame):
        """Display frame on canvas"""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Resize to fit canvas
        h, w = frame_rgb.shape[:2]
        ratio = min(self.canvas_width / w, self.canvas_height / h)
        new_w = int(w * ratio)
        new_h = int(h * ratio)

        resized = cv2.resize(frame_rgb, (new_w, new_h))

        pil_img = Image.fromarray(resized)
        self.tk_image = ImageTk.PhotoImage(pil_img)

        self.canvas.delete("all")
        x_offset = (self.canvas_width - new_w) // 2
        y_offset = (self.canvas_height - new_h) // 2
        self.canvas.create_image(x_offset, y_offset, anchor=tk.NW, image=self.tk_image)

    def stop_camera(self):
        """Stop camera"""
        self.running = False
        if self.cap:
            self.cap.release()
        self.start_btn.config(text="▶️ Start Camera", bg='#27ae60')
        self.status_label.config(text="Camera stopped")
        self.info_label.config(text="")


if __name__ == "__main__":
    root = tk.Tk()
    app = FaceDetectionEducational(root)
    root.mainloop()