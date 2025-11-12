import tkinter as tk
import cv2
import numpy as np
from PIL import Image, ImageTk
import threading


class CameraApp:
    def __init__(self, root):
        self.root = root
        self.root.title("🐯 Tiger Face Overlay - Educational Mode")
        self.root.geometry("900x700")
        self.root.configure(bg='#2c3e50')

        self.running = False
        self.cap = None
        self.mode = 'tiger'
        self.show_detection = False  # Toggle to show detection process
        self.frame_skip = 3
        self.frame_count = 0

        # Threading
        self.current_frame = None
        self.frame_lock = threading.Lock()

        # Title
        title = tk.Label(self.root, text="🐯 Tiger Face Overlay - See How It Works!",
                        font=("Arial", 20, "bold"), bg='#2c3e50', fg='#ff6b35')
        title.pack(pady=15)

        # Explanation
        self.explanation = tk.Label(self.root,
                                    text="Computer finds your face and places tiger perfectly!",
                                    font=("Arial", 12, "bold"), bg='#2c3e50', fg='#ffffff')
        self.explanation.pack(pady=5)

        # Button frame
        self.button_frame = tk.Frame(self.root, bg='#2c3e50')
        self.button_frame.pack(pady=10)

        # Start Tiger Mode button
        self.tiger_btn = tk.Button(self.button_frame, text="🐯 Start Tiger Mode",
                                   command=self.tiger_mode,
                                   font=("Arial", 13, "bold"), bg='#e67e22', fg='white',
                                   width=18, height=2, relief=tk.RAISED, bd=4, cursor='hand2')
        self.tiger_btn.grid(row=0, column=0, padx=10, pady=5)

        # Toggle detection view button
        self.detect_btn = tk.Button(self.button_frame, text="🎯 Show Detection Process",
                                    command=self.toggle_detection,
                                    font=("Arial", 13, "bold"), bg='#3498db', fg='white',
                                    width=22, height=2, relief=tk.RAISED, bd=4, cursor='hand2',
                                    state=tk.DISABLED)
        self.detect_btn.grid(row=0, column=1, padx=10, pady=5)

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
                                   font=("Arial", 11, "bold"), bg='#2c3e50', fg='#f39c12',
                                   wraplength=850, justify=tk.LEFT)
        self.info_label.pack(pady=10)

        # Load face cascade
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

        # Load tiger image
        self.tiger_img = cv2.imread("tiger.jpg")
        if self.tiger_img is None:
            print("Warning: tiger.jpg not found!")
        self.cached_tigers = {}
        self.last_faces = []

        # Settings
        self.frame_width = 640
        self.frame_height = 480

    def toggle_detection(self):
        """Toggle between showing detection process and normal tiger overlay"""
        self.show_detection = not self.show_detection

        if self.show_detection:
            self.detect_btn.config(text="🐯 Show Tiger Overlay", bg='#e67e22')
            self.explanation.config(text="🎯 DETECTION MODE: See how computer finds face boundaries!")
        else:
            self.detect_btn.config(text="🎯 Show Detection Process", bg='#3498db')
            self.explanation.config(text="🐯 TIGER MODE: Tiger image placed perfectly on face!")

    def tiger_mode(self):
        if not self.running:
            self.running = True
            self.tiger_btn.config(state=tk.DISABLED)
            self.detect_btn.config(state=tk.NORMAL)

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
        self.cached_tigers.clear()
        self.last_faces = []
        self.tiger_btn.config(state=tk.NORMAL)
        self.detect_btn.config(state=tk.DISABLED)
        self.info_label.config(text="")

    def update_frame(self):
        if self.running:
            with self.frame_lock:
                if self.current_frame is not None:
                    frame = self.current_frame.copy()
                else:
                    frame = None

            if frame is not None:
                if self.show_detection:
                    # Show detection process
                    frame, info = self.show_detection_process(frame)
                else:
                    # Normal tiger overlay
                    frame, info = self.replace_faces_with_tiger(frame)

                self.info_label.config(text=info)

                # Display
                img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(img)
                imgtk = ImageTk.PhotoImage(image=img)
                self.video_label.imgtk = imgtk
                self.video_label.configure(image=imgtk)

            self.root.after(16, self.update_frame)

    def show_detection_process(self, frame):
        """Show the detection process - how computer finds where to place tiger"""
        self.frame_count += 1
        info = "🎯 DETECTION PROCESS:\n"

        # Detect faces
        if self.frame_count % self.frame_skip == 0:
            small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
            gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)

            faces = self.face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=3, minSize=(20, 20),
                flags=cv2.CASCADE_SCALE_IMAGE
            )

            if len(faces) > 0:
                self.last_faces = [(x*2, y*2, w*2, h*2) for (x, y, w, h) in faces]

        info += f"Step 1: Found {len(self.last_faces)} face region(s)\n"

        # Show detection details
        for i, (x, y, w, h) in enumerate(self.last_faces):
            # Draw face bounding box
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)
            cv2.putText(frame, f"Face #{i+1}", (x, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

            # Draw center point
            center_x = x + w // 2
            center_y = y + h // 2
            cv2.circle(frame, (center_x, center_y), 5, (0, 255, 0), -1)
            cv2.putText(frame, "CENTER", (center_x + 10, center_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Draw measurements
            # Width
            cv2.line(frame, (x, y+h+20), (x+w, y+h+20), (255, 255, 0), 2)
            cv2.putText(frame, f"Width: {w}px", (x, y+h+40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

            # Height
            cv2.line(frame, (x-20, y), (x-20, y+h), (255, 255, 0), 2)
            cv2.putText(frame, f"H:{h}", (x-80, y+h//2),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

            # Show corners
            corner_size = 15
            # Top-left
            cv2.circle(frame, (x, y), corner_size, (255, 0, 0), 3)
            cv2.putText(frame, f"({x},{y})", (x+5, y+20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

            # Top-right
            cv2.circle(frame, (x+w, y), corner_size, (255, 0, 0), 3)

            # Bottom-left
            cv2.circle(frame, (x, y+h), corner_size, (255, 0, 0), 3)

            # Bottom-right
            cv2.circle(frame, (x+w, y+h), corner_size, (255, 0, 0), 3)
            cv2.putText(frame, f"({x+w},{y+h})", (x+w-80, y+h-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

            # Try to detect eyes for better placement
            roi_gray = cv2.cvtColor(frame[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)
            eyes = self.eye_cascade.detectMultiScale(roi_gray, 1.1, 3)

            if len(eyes) >= 2:
                info += f"\nStep 2: Found {len(eyes)} eyes in face #{i+1}\n"
                for (ex, ey, ew, eh) in eyes[:2]:  # Show first 2 eyes
                    cv2.rectangle(frame, (x+ex, y+ey), (x+ex+ew, y+ey+eh), (255, 0, 255), 2)
                    cv2.circle(frame, (x+ex+ew//2, y+ey+eh//2), 3, (255, 0, 255), -1)

            # Show tiger placement guide
            if self.tiger_img is not None:
                # Draw dashed outline where tiger will be placed
                dash_length = 10
                for dx in range(x, x+w, dash_length*2):
                    cv2.line(frame, (dx, y), (min(dx+dash_length, x+w), y), (0, 255, 255), 2)
                    cv2.line(frame, (dx, y+h), (min(dx+dash_length, x+w), y+h), (0, 255, 255), 2)
                for dy in range(y, y+h, dash_length*2):
                    cv2.line(frame, (x, dy), (x, min(dy+dash_length, y+h)), (0, 255, 255), 2)
                    cv2.line(frame, (x+w, dy), (x+w, min(dy+dash_length, y+h)), (0, 255, 255), 2)

                info += f"\nStep 3: Tiger will be resized to {w}×{h} and placed here!"

        if len(self.last_faces) == 0:
            info += "\n❌ No face detected - move to better lighting or face camera!"

        info += "\n\n💡 Click 'Show Tiger Overlay' to see the final result!"

        return frame, info

    def replace_faces_with_tiger(self, frame):
        """Normal tiger overlay mode"""
        self.frame_count += 1
        info = "🐯 TIGER OVERLAY MODE:\n"

        # Detect faces
        if self.frame_count % self.frame_skip == 0:
            small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
            gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)

            faces = self.face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=3, minSize=(20, 20),
                flags=cv2.CASCADE_SCALE_IMAGE
            )

            if len(faces) > 0:
                self.last_faces = [(x*2, y*2, w*2, h*2) for (x, y, w, h) in faces]

        info += f"✅ Applying tiger overlay to {len(self.last_faces)} face(s)\n"

        # Apply tiger overlay
        for i, (x, y, w, h) in enumerate(self.last_faces):
            size_key = (w, h)

            # Cache resized tigers
            if size_key not in self.cached_tigers:
                if self.tiger_img is not None:
                    self.cached_tigers[size_key] = cv2.resize(self.tiger_img, (w, h),
                                                              interpolation=cv2.INTER_LINEAR)

            if size_key in self.cached_tigers:
                resized_tiger = self.cached_tigers[size_key]

                # Bounds check and overlay
                if 0 <= y and 0 <= x and y + h <= frame.shape[0] and x + w <= frame.shape[1]:
                    frame[y:y+h, x:x+w] = resized_tiger
                    info += f"  Face #{i+1}: Tiger placed at ({x},{y}) size {w}×{h}\n"

        if len(self.last_faces) > 0:
            info += "\n💡 Click 'Show Detection Process' to see how it works!"
        else:
            info += "\n❌ No face detected - trying to find you..."

        return frame, info


# Run the app
if __name__ == "__main__":
    root = tk.Tk()
    app = CameraApp(root)
    root.mainloop()