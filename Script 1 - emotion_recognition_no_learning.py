import tkinter as tk
from PIL import Image, ImageTk
import cv2
import os
from deepface import DeepFace
from datetime import datetime
import threading

# בדיקת תיקיית data קיימת
DATA_FOLDER = "data"
emotion_dirs = ['angry', 'disgusted', 'fearful', 'happy', 'sad', 'surprised', 'neutral']

# מיפוי בין הרגשות של DeepFace לתיקיות שלך
emotion_mapping = {
    'angry': 'angry',
    'disgust': 'disgusted',
    'fear': 'fearful',
    'happy': 'happy',
    'sad': 'sad',
    'surprise': 'surprised',
    'neutral': 'neutral'
}

# בדיקה אם התיקייה קיימת
if not os.path.exists(DATA_FOLDER):
    print(f"⚠️ Warning: '{DATA_FOLDER}' folder not found. Creating it...")
    os.makedirs(DATA_FOLDER)
    for e in emotion_dirs:
        os.makedirs(f"{DATA_FOLDER}/{e}", exist_ok=True)
else:
    print(f"✅ Using existing '{DATA_FOLDER}' folder")
    # בדיקה שכל תתי-התיקיות קיימות
    for e in emotion_dirs:
        if not os.path.exists(f"{DATA_FOLDER}/{e}"):
            print(f"  Creating missing subfolder: {e}")
            os.makedirs(f"{DATA_FOLDER}/{e}", exist_ok=True)

# מיפוי אימוג'ים וצבעים
emoji_map = {
    'angry': '😡',
    'disgust': '🤢',
    'fear': '😱',
    'happy': '😊',
    'sad': '😢',
    'surprise': '😲',
    'neutral': '😐'
}

color_map = {
    'angry': (0, 0, 255),  # Red
    'disgust': (0, 128, 0),  # Green
    'fear': (128, 0, 128),  # Purple
    'happy': (0, 255, 0),  # Bright Green
    'sad': (255, 0, 0),  # Blue
    'surprise': (0, 255, 255),  # Yellow
    'neutral': (128, 128, 128)  # Gray
}


class EmotionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("🎭 Multi-Face Emotion Detector")

        # Get screen size
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        window_width = int(screen_width * 0.85)
        window_height = int(screen_height * 0.85)

        self.root.geometry(f"{window_width}x{window_height}")
        self.root.configure(bg="#1a1a2e")

        # Variables
        self.running = False
        self.cap = None
        self.freeze = False
        self.current_frame = None
        self.detected_faces = []  # רשימה של כל הפרצופים שזוהו
        self.selected_face = None  # הפרצוף שנבחר לשמירה

        # Threading
        self.frame_lock = threading.Lock()
        self.latest_frame = None
        self.face_counter = 0  # מונה פרצופים

        # Create UI
        self.create_ui()

    def create_ui(self):
        """Create user interface"""
        # Title
        title = tk.Label(self.root, text="🎭 Multi-Face Emotion Detection Lab",
                         font=("Arial", 24, "bold"), bg="#1a1a2e", fg="#00ff88")
        title.pack(pady=15)

        # Explanation
        explanation = tk.Label(self.root,
                               text="AI analyzes multiple faces to detect emotions in real-time! Click on a face to select it for saving.",
                               font=("Arial", 13), bg="#1a1a2e", fg="#ffffff")
        explanation.pack(pady=5)

        # Control buttons
        button_frame = tk.Frame(self.root, bg="#1a1a2e")
        button_frame.pack(pady=10)

        self.start_btn = tk.Button(button_frame, text="▶️ Start Camera",
                                   command=self.start_camera,
                                   font=("Arial", 14, "bold"), bg="#27ae60", fg="white",
                                   width=18, height=2, relief=tk.RAISED, bd=4, cursor="hand2")
        self.start_btn.grid(row=0, column=0, padx=10)

        self.freeze_btn = tk.Button(button_frame, text="❄️ Freeze",
                                    command=self.toggle_freeze,
                                    font=("Arial", 14, "bold"), bg="#3498db", fg="white",
                                    width=18, height=2, relief=tk.RAISED, bd=4, cursor="hand2",
                                    state=tk.DISABLED)
        self.freeze_btn.grid(row=0, column=1, padx=10)

        self.save_btn = tk.Button(button_frame, text="💾 Save Selected",
                                  command=self.save_selected_face,
                                  font=("Arial", 14, "bold"), bg="#9b59b6", fg="white",
                                  width=18, height=2, relief=tk.RAISED, bd=4, cursor="hand2",
                                  state=tk.DISABLED)
        self.save_btn.grid(row=0, column=2, padx=10)

        self.stop_btn = tk.Button(button_frame, text="⏹️ Stop",
                                  command=self.stop_camera,
                                  font=("Arial", 14, "bold"), bg="#e74c3c", fg="white",
                                  width=18, height=2, relief=tk.RAISED, bd=4, cursor="hand2")
        self.stop_btn.grid(row=0, column=3, padx=10)

        # Video display
        self.video_label = tk.Label(self.root, bg="black", cursor="hand2")
        self.video_label.pack(pady=10)
        self.video_label.bind("<Button-1>", self.on_frame_click)

        # Detected faces summary
        faces_frame = tk.Frame(self.root, bg="#16213e", relief=tk.RAISED, bd=3)
        faces_frame.pack(pady=10, padx=20, fill=tk.X)

        faces_title = tk.Label(faces_frame, text="👥 Detected Faces:",
                               font=("Arial", 18, "bold"), bg="#16213e", fg="#00ff88")
        faces_title.pack(anchor="w", padx=20, pady=5)

        self.faces_text = tk.Label(faces_frame, text="No faces detected",
                                   font=("Arial", 14), bg="#16213e", fg="#ffffff",
                                   justify=tk.LEFT, anchor="w")
        self.faces_text.pack(anchor="w", padx=20, pady=10, fill=tk.X)

        # Status
        self.status_label = tk.Label(self.root, text="Click 'Start Camera' to begin",
                                     font=("Arial", 14), bg="#1a1a2e", fg="#f39c12")
        self.status_label.pack(pady=5)

    def start_camera(self):
        """Start camera"""
        if not self.running:
            self.running = True
            self.start_btn.config(state=tk.DISABLED)
            self.freeze_btn.config(state=tk.NORMAL)
            self.save_btn.config(state=tk.NORMAL)

            # Open webcam (1 = external webcam)
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
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

            self.status_label.config(text="🎥 Camera active - detecting multiple faces...")

            # Start threads
            threading.Thread(target=self.capture_thread, daemon=True).start()
            self.update_frame()

    def capture_thread(self):
        """Capture frames in separate thread"""
        while self.running:
            if self.cap and self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret:
                    frame = cv2.flip(frame, 1)
                    with self.frame_lock:
                        self.latest_frame = frame

    def toggle_freeze(self):
        """Toggle freeze mode"""
        self.freeze = not self.freeze
        if self.freeze:
            self.freeze_btn.config(text="▶️ Resume", bg="#e67e22")
            self.status_label.config(text="❄️ Frame frozen - click on a face to select, then 'Save Selected'")
        else:
            self.freeze_btn.config(text="❄️ Freeze", bg="#3498db")
            self.status_label.config(text="🎥 Camera active - detecting multiple faces...")
            self.selected_face = None

    def on_frame_click(self, event):
        """Handle click on video frame to select a face"""
        if not self.freeze or not self.detected_faces:
            return

        # Get click coordinates (need to adjust for image scaling)
        click_x = event.x
        click_y = event.y

        # Find which face was clicked
        for i, face_data in enumerate(self.detected_faces):
            region = face_data['region']
            x, y, w, h = region['x'], region['y'], region['w'], region['h']

            # Check if click is within face bounds (accounting for image scaling)
            # This is approximate - would need exact scaling calculation for precision
            if x <= click_x * 2 <= x + w and y <= click_y * 2 <= y + h:
                self.selected_face = i
                self.status_label.config(
                    text=f"✅ Face #{i + 1} selected ({face_data['emotion'].upper()}) - Click 'Save Selected' to save"
                )
                break

    def save_selected_face(self):
        """Save the selected face as training example"""
        if self.selected_face is None or not self.detected_faces:
            self.status_label.config(text="⚠️ Please click on a face to select it first!")
            return

        face_data = self.detected_faces[self.selected_face]
        emotion = face_data['emotion']
        region = face_data['region']

        if self.current_frame is not None:
            # Crop the face from the frame
            x, y, w, h = region['x'], region['y'], region['w'], region['h']

            # Add padding
            padding = 20
            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(self.current_frame.shape[1], x + w + padding)
            y2 = min(self.current_frame.shape[0], y + h + padding)

            face_crop = self.current_frame[y1:y2, x1:x2]

            # המרה לשם התיקייה הנכון
            folder_name = emotion_mapping.get(emotion, emotion)

            # Save
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{DATA_FOLDER}/{folder_name}/face{self.selected_face + 1}_{timestamp}.jpg"
            cv2.imwrite(filename, cv2.cvtColor(face_crop, cv2.COLOR_RGB2BGR))

            self.status_label.config(text=f"✅ Saved Face #{self.selected_face + 1}: {filename}")
            print(f"💾 Saved: {filename}")

            self.selected_face = None

    def update_frame(self):
        """Update display with emotion detection"""
        if self.running and not self.freeze:
            with self.frame_lock:
                if self.latest_frame is not None:
                    frame = self.latest_frame.copy()
                else:
                    frame = None

            if frame is not None:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                try:
                    # Analyze emotions - will return list of faces
                    results = DeepFace.analyze(
                        rgb_frame,
                        actions=['emotion'],
                        enforce_detection=False,
                        detector_backend='opencv'
                    )

                    # Ensure results is a list
                    if not isinstance(results, list):
                        results = [results]

                    # Store detected faces
                    self.detected_faces = []
                    self.current_frame = rgb_frame.copy()

                    # Process each detected face
                    for i, result in enumerate(results):
                        emotion = result.get('dominant_emotion', 'neutral')
                        emotions_dict = result.get('emotion', {})
                        confidence = emotions_dict.get(emotion, 0)
                        region = result.get('region', {})

                        # Store face data
                        self.detected_faces.append({
                            'emotion': emotion,
                            'confidence': confidence,
                            'region': region,
                            'emotions_dict': emotions_dict
                        })

                        # Draw face box and emotion
                        if region:
                            x, y, w, h = region.get('x', 0), region.get('y', 0), region.get('w', 0), region.get('h', 0)

                            # Get color for emotion
                            color = color_map.get(emotion, (128, 128, 128))

                            # Draw rectangle around face (thicker if selected)
                            thickness = 6 if (self.freeze and self.selected_face == i) else 4
                            cv2.rectangle(rgb_frame, (x, y), (x + w, y + h), color, thickness)

                            # Draw face number
                            face_num = f"#{i + 1}"
                            cv2.circle(rgb_frame, (x + 15, y + 15), 20, color, -1)
                            cv2.putText(rgb_frame, face_num, (x + 5, y + 25),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

                            # Draw emotion label
                            label = f"{emotion.upper()}"
                            font_scale = 1.0
                            thickness_text = 2

                            # Background for text
                            (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX,
                                                                           font_scale, thickness_text)
                            cv2.rectangle(rgb_frame, (x, y - text_height - 12), (x + text_width + 10, y), color, -1)

                            # Text
                            cv2.putText(rgb_frame, label, (x + 5, y - 8),
                                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255),
                                        thickness_text, cv2.LINE_AA)

                            # Draw confidence bar
                            bar_width = int(w * (confidence / 100))
                            cv2.rectangle(rgb_frame, (x, y + h + 5), (x + bar_width, y + h + 15), color, -1)
                            cv2.rectangle(rgb_frame, (x, y + h + 5), (x + w, y + h + 15), color, 2)

                    # Update faces summary
                    if self.detected_faces:
                        faces_summary = f"Found {len(self.detected_faces)} face(s):\n"
                        for i, face in enumerate(self.detected_faces):
                            emoji = emoji_map.get(face['emotion'], '😐')
                            faces_summary += f"  {emoji} Face #{i + 1}: {face['emotion'].capitalize()} ({face['confidence']:.1f}%)\n"
                        self.faces_text.config(text=faces_summary)
                    else:
                        self.faces_text.config(text="No faces detected")

                except Exception as e:
                    print(f"⚠️ Detection error: {e}")
                    self.detected_faces = []
                    self.faces_text.config(text="Detection error - try different lighting")

                # Display frame
                img = Image.fromarray(rgb_frame)
                # Resize to fit window
                img.thumbnail((1200, 700), Image.Resampling.LANCZOS)
                imgtk = ImageTk.PhotoImage(image=img)
                self.video_label.imgtk = imgtk
                self.video_label.configure(image=imgtk)

        self.root.after(100, self.update_frame)

    def stop_camera(self):
        """Stop camera"""
        self.running = False
        if self.cap:
            self.cap.release()
        self.video_label.config(image='')
        self.start_btn.config(state=tk.NORMAL)
        self.freeze_btn.config(state=tk.DISABLED)
        self.save_btn.config(state=tk.DISABLED)
        self.status_label.config(text="Camera stopped")
        self.faces_text.config(text="No faces detected")


if __name__ == "__main__":
    root = tk.Tk()
    app = EmotionApp(root)
    root.mainloop()