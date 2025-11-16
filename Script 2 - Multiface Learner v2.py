import cv2
import os
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import datetime
import numpy as np
import threading
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
import mediapipe as mp

# ---- תיקיות ואימוג'ים ----
BASE_DIR = "data"
MODEL_PATH = os.path.join(BASE_DIR, "emotion_model.h5")
EMOTIONS = {
    "שמחה 😄": "happy",
    "עצב 😢": "sad",
    "מופתע 😮": "surprised",
    "כעס 😡": "angry",
    "פחד 😨": "fearful",
    "גועל 🤢": "disgusted"
}
EMOTION_HEB = list(EMOTIONS.keys())
EMOTION_FOLDERS = list(EMOTIONS.values())
IMG_SIZE = 96  # רק הפנים נשמרות ונלמדות - לא הרקע!

# צבעים מודרניים
COLOR_MAP = {
    "happy": (59, 130, 246),  # Blue
    "sad": (96, 165, 250),  # Light Blue
    "surprised": (251, 191, 36),  # Amber
    "angry": (220, 38, 38),  # Red
    "fearful": (168, 85, 247),  # Purple
    "disgusted": (34, 197, 94)  # Green
}

# יצירת תיקיות
os.makedirs(BASE_DIR, exist_ok=True)
for folder in EMOTION_FOLDERS:
    os.makedirs(os.path.join(BASE_DIR, folder), exist_ok=True)

# ---- Mediapipe Face Detection ----
mp_face_detection = mp.solutions.face_detection


def detect_all_faces(frame):
    """זיהוי כל הפרצופים בפריים - רק הפנים, לא הרקע!"""
    with mp_face_detection.FaceDetection(
            model_selection=1,
            min_detection_confidence=0.5
    ) as face_detection:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detection.process(frame_rgb)

        faces = []
        if results.detections:
            h, w = frame.shape[:2]
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                x1 = max(int(bbox.xmin * w), 0)
                y1 = max(int(bbox.ymin * h), 0)
                x2 = min(int((bbox.xmin + bbox.width) * w), w)
                y2 = min(int((bbox.ymin + bbox.height) * h), h)

                if x2 > x1 and y2 > y1:
                    faces.append({
                        'box': (x1, y1, x2 - x1, y2 - y1),
                        'confidence': detection.score[0]
                    })

        return faces


def crop_face(frame, box):
    """חיתוך רק הפנים - זה מה שנשמר ונלמד!"""
    x, y, w, h = box
    # הוספת padding קטן
    padding = 10
    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(frame.shape[1], x + w + padding)
    y2 = min(frame.shape[0], y + h + padding)

    face = frame[y1:y2, x1:x2]
    if face.size == 0:
        return None
    return cv2.resize(face, (IMG_SIZE, IMG_SIZE))


# ---- בניית מודל משופר ----
def build_model():
    """CNN שלומד רק מפנים - לא מרקע!"""
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),

        Conv2D(64, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),

        Conv2D(128, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),

        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(len(EMOTIONS), activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


# ---- אפליקציה ----
class EmotionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("🎭 מערכת זיהוי רגשות - Multi Face")

        # Get screen dimensions
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        window_width = int(screen_width * 0.85)
        window_height = int(screen_height * 0.85)

        self.root.geometry(f"{window_width}x{window_height}")
        self.root.configure(bg="#0f172a")

        # Variables
        self.cap = None
        self.running = False
        self.frozen = False
        self.current_frame = None
        self.detected_faces = []
        self.selected_face_idx = None
        self.model = None
        self.model_trained = False
        self.recognizing = False

        # Threading
        self.frame_lock = threading.Lock()
        self.latest_frame = None

        # בדיקה אם יש מודל שמור
        if os.path.exists(MODEL_PATH):
            try:
                self.model = load_model(MODEL_PATH)
                self.model_trained = True
            except:
                pass

        self.create_ui()
        self.try_open_camera()

    def create_ui(self):
        """יצירת ממשק מודרני"""
        # Header
        header = tk.Frame(self.root, bg="#1e293b", height=70)
        header.pack(fill=tk.X)
        header.pack_propagate(False)

        title = tk.Label(
            header, text="🎭 מערכת זיהוי רגשות",
            font=("Segoe UI", 26, "bold"), bg="#1e293b", fg="#f1f5f9"
        )
        title.pack(pady=15)

        # Main content
        content = tk.Frame(self.root, bg="#0f172a")
        content.pack(fill=tk.BOTH, expand=True, padx=30, pady=15)

        # Video frame
        video_container = tk.Frame(content, bg="#1e293b")
        video_container.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 20))

        self.video_label = tk.Label(video_container, bg="#000000", cursor="hand2")
        self.video_label.pack(fill=tk.BOTH, expand=True, padx=3, pady=3)
        self.video_label.bind("<Button-1>", self.on_frame_click)

        # Sidebar
        sidebar = tk.Frame(content, bg="#1e293b", width=350)
        sidebar.pack(side=tk.RIGHT, fill=tk.Y)
        sidebar.pack_propagate(False)

        # Controls
        controls = tk.Frame(sidebar, bg="#1e293b")
        controls.pack(pady=15, padx=15, fill=tk.X)

        self.freeze_btn = tk.Button(
            controls, text="❄  Freeze Frame",
            command=self.toggle_freeze,
            font=("Segoe UI", 11, "bold"), bg="#64748b", fg="white",
            relief=tk.FLAT, cursor="hand2", height=2, state=tk.DISABLED
        )
        self.freeze_btn.pack(fill=tk.X, pady=3)

        self.train_btn = tk.Button(
            controls, text="🏋️  Train Model",
            command=self.train_model,
            font=("Segoe UI", 11, "bold"), bg="#f59e0b", fg="white",
            relief=tk.FLAT, cursor="hand2", height=2
        )
        self.train_btn.pack(fill=tk.X, pady=3)

        self.recognize_btn = tk.Button(
            controls, text="🎯  Live Recognition",
            command=self.toggle_recognition,
            font=("Segoe UI", 11, "bold"), bg="#10b981", fg="white",
            relief=tk.FLAT, cursor="hand2", height=2,
            state=tk.NORMAL if self.model_trained else tk.DISABLED
        )
        self.recognize_btn.pack(fill=tk.X, pady=3)

        self.stop_btn = tk.Button(
            controls, text="⏹  Stop Camera",
            command=self.stop_camera,
            font=("Segoe UI", 11, "bold"), bg="#ef4444", fg="white",
            relief=tk.FLAT, cursor="hand2", height=2
        )
        self.stop_btn.pack(fill=tk.X, pady=3)

        # Separator
        tk.Frame(sidebar, bg="#334155", height=2).pack(fill=tk.X, padx=15, pady=12)

        # Faces info
        tk.Label(
            sidebar, text="👥 Detected Faces",
            font=("Segoe UI", 14, "bold"), bg="#1e293b", fg="#f1f5f9"
        ).pack(anchor="w", padx=15, pady=(5, 5))

        faces_scroll_frame = tk.Frame(sidebar, bg="#1e293b")
        faces_scroll_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=5)

        self.faces_info = tk.Label(
            faces_scroll_frame, text="Click 'Freeze' to capture",
            font=("Segoe UI", 10), bg="#1e293b", fg="#94a3b8",
            justify=tk.LEFT, anchor="nw", wraplength=320
        )
        self.faces_info.pack(fill=tk.BOTH, expand=True)

        # Emotion selection frame
        self.emotion_frame = tk.Frame(sidebar, bg="#1e293b")

        tk.Label(
            self.emotion_frame, text="בחר רגש לשמירה:",
            font=("Segoe UI", 12, "bold"), bg="#1e293b", fg="#f1f5f9"
        ).pack(pady=8)

        self.emotion_buttons_container = tk.Frame(self.emotion_frame, bg="#1e293b")
        self.emotion_buttons_container.pack(fill=tk.X, padx=10)

        # Status bar - תיקון: הזזה למעלה יותר
        status_bar = tk.Frame(self.root, bg="#1e293b", height=40)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X, before=content)
        status_bar.pack_propagate(False)

        self.status_label = tk.Label(
            status_bar, text="Ready",
            font=("Segoe UI", 10), bg="#1e293b", fg="#94a3b8", anchor="w"
        )
        self.status_label.pack(side=tk.LEFT, padx=20, pady=8)

    def try_open_camera(self):
        """ניסיון לפתוח מצלמה"""
        for cam_id in [1, 0]:
            self.cap = cv2.VideoCapture(cam_id, cv2.CAP_DSHOW)
            if not self.cap.isOpened():
                self.cap = cv2.VideoCapture(cam_id)

            if self.cap.isOpened():
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
                self.cap.set(cv2.CAP_PROP_FPS, 30)
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

                self.running = True
                threading.Thread(target=self.capture_thread, daemon=True).start()
                self.update_video()
                self.status_label.config(text=f"Camera {cam_id} active", fg="#10b981")
                self.freeze_btn.config(state=tk.NORMAL, bg="#10b981")
                return

        messagebox.showwarning("Warning", "No camera found!")

    def capture_thread(self):
        """Thread לקליטת פריימים"""
        while self.running:
            if self.cap and self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret:
                    frame = cv2.flip(frame, 1)
                    with self.frame_lock:
                        self.latest_frame = frame

    def toggle_freeze(self):
        """הקפאה/המשך"""
        self.frozen = not self.frozen
        if self.frozen:
            self.freeze_btn.config(text="▶  Resume", bg="#f59e0b")
            self.status_label.config(text="Frozen - Click on a face", fg="#f59e0b")
        else:
            self.freeze_btn.config(text="❄  Freeze Frame", bg="#10b981")
            self.status_label.config(text="Live", fg="#10b981")
            self.selected_face_idx = None
            self.emotion_frame.pack_forget()

    def on_frame_click(self, event):
        """בחירת פרצוף - תיקון: חישוב scaling נכון"""
        if not self.frozen or not self.detected_faces:
            return

        # קבל גודל התצוגה והפריים
        label_w = self.video_label.winfo_width()
        label_h = self.video_label.winfo_height()

        if self.current_frame is None or label_w <= 1:
            return

        frame_h, frame_w = self.current_frame.shape[:2]

        # חשב scaling (שומר על aspect ratio)
        scale = min(label_w / frame_w, label_h / frame_h)
        display_w = int(frame_w * scale)
        display_h = int(frame_h * scale)

        # מרכז התמונה
        offset_x = (label_w - display_w) // 2
        offset_y = (label_h - display_h) // 2

        # המר קליק לקואורדינטות מקוריות
        click_x = int((event.x - offset_x) / scale)
        click_y = int((event.y - offset_y) / scale)

        for i, face in enumerate(self.detected_faces):
            x, y, w, h = face['box']
            if x <= click_x <= x + w and y <= click_y <= y + h:
                self.selected_face_idx = i
                self.status_label.config(
                    text=f"Face #{i + 1} selected - Choose emotion below",
                    fg="#8b5cf6"
                )
                self.show_emotion_buttons()
                return

    def show_emotion_buttons(self):
        """הצגת כפתורי רגשות - תיקון: הצגה נכונה"""
        if self.selected_face_idx is None:
            self.status_label.config(text="Please select a face first", fg="#f59e0b")
            return

        # הצג את הפריים
        self.emotion_frame.pack(fill=tk.X, padx=15, pady=10)

        # נקה כפתורים קודמים
        for widget in self.emotion_buttons_container.winfo_children():
            widget.destroy()

        colors = {
            "happy": "#3b82f6", "sad": "#60a5fa", "surprised": "#fbbf24",
            "angry": "#dc2626", "fearful": "#a855f7", "disgusted": "#22c55e"
        }

        for heb, eng in EMOTIONS.items():
            btn = tk.Button(
                self.emotion_buttons_container, text=heb,
                font=("Segoe UI", 10, "bold"),
                bg=colors.get(eng, "#64748b"), fg="white",
                relief=tk.FLAT, cursor="hand2",
                command=lambda e=eng, h=heb: self.save_selected_face(e, h)
            )
            btn.pack(fill=tk.X, pady=2)

    def save_selected_face(self, emotion_folder, emotion_heb):
        """שמירת פרצוף נבחר - רק הפנים!"""
        if self.current_frame is None or self.selected_face_idx is None:
            return

        if self.selected_face_idx >= len(self.detected_faces):
            return

        box = self.detected_faces[self.selected_face_idx]['box']
        face = crop_face(self.current_frame, box)  # רק הפנים נשמרות!

        if face is None:
            messagebox.showerror("Error", "Failed to crop face")
            return

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{emotion_folder}_{timestamp}.jpg"
        filepath = os.path.join(BASE_DIR, emotion_folder, filename)
        cv2.imwrite(filepath, face)  # שומר רק את הפנים - 96x96!

        self.status_label.config(text=f"Saved: {emotion_heb}", fg="#10b981")
        print(f"💾 Saved face (96x96) to: {filepath}")

        self.selected_face_idx = None
        self.emotion_frame.pack_forget()
        self.frozen = False
        self.freeze_btn.config(text="❄  Freeze Frame", bg="#10b981")

    def update_video(self):
        """עדכון תצוגת וידאו - תיקון: scaling נכון"""
        if self.running and not self.frozen and not self.recognizing:
            with self.frame_lock:
                frame = self.latest_frame.copy() if self.latest_frame is not None else None

            if frame is not None:
                self.current_frame = frame.copy()
                display_frame = frame.copy()

                # זיהוי פרצופים
                self.detected_faces = detect_all_faces(frame)

                for i, face in enumerate(self.detected_faces):
                    x, y, w, h = face['box']
                    color = (59, 130, 246)

                    cv2.rectangle(display_frame, (x, y), (x + w, y + h), color, 3)
                    cv2.circle(display_frame, (x + 20, y + 20), 18, color, -1)
                    cv2.putText(display_frame, f"{i + 1}", (x + 13, y + 27),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                # עדכון מידע
                if self.detected_faces:
                    info = f"Found {len(self.detected_faces)} face(s)\n\n"
                    for i in range(len(self.detected_faces)):
                        info += f"Face #{i + 1}\n"
                    self.faces_info.config(text=info, fg="#f1f5f9")
                else:
                    self.faces_info.config(text="No faces detected", fg="#64748b")

                self.show_frame(display_frame)

        if self.running and not self.recognizing:
            self.root.after(30, self.update_video)

    def toggle_recognition(self):
        """הפעלה/כיבוי זיהוי חי"""
        if not self.model_trained:
            messagebox.showerror("Error", "Model not trained!")
            return

        self.recognizing = not self.recognizing
        self.frozen = False
        self.emotion_frame.pack_forget()

        if self.recognizing:
            self.recognize_btn.config(text="⏹  Stop Recognition", bg="#ef4444")
            self.status_label.config(text="Recognizing emotions...", fg="#10b981")
            self.update_recognition()
        else:
            self.recognize_btn.config(text="🎯  Live Recognition", bg="#10b981")
            self.status_label.config(text="Live", fg="#10b981")
            self.update_video()

    def update_recognition(self):
        """לולאת זיהוי רגשות"""
        if not self.recognizing or not self.running:
            return

        with self.frame_lock:
            frame = self.latest_frame.copy() if self.latest_frame is not None else None

        if frame is not None:
            display_frame = frame.copy()
            faces = detect_all_faces(frame)
            self.detected_faces = faces

            predictions = []
            for i, face_data in enumerate(faces):
                box = face_data['box']
                face_crop = crop_face(frame, box)  # חותך רק פנים

                if face_crop is not None:
                    face_norm = face_crop / 255.0
                    face_input = np.expand_dims(face_norm, axis=0)
                    preds = self.model.predict(face_input, verbose=0)[0]
                    idx = np.argmax(preds)
                    confidence = preds[idx] * 100
                    emotion = EMOTION_FOLDERS[idx]
                    emotion_heb = EMOTION_HEB[idx]

                    predictions.append({
                        'emotion': emotion,
                        'emotion_heb': emotion_heb,
                        'confidence': confidence
                    })

                    x, y, w, h = box
                    color = COLOR_MAP.get(emotion, (156, 163, 175))

                    cv2.rectangle(display_frame, (x, y), (x + w, y + h), color, 4)
                    cv2.circle(display_frame, (x + 22, y + 22), 20, color, -1)
                    cv2.putText(display_frame, f"{i + 1}", (x + 14, y + 29),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                    label = f"{emotion.upper()} {confidence:.0f}%"
                    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    cv2.rectangle(display_frame, (x, y - th - 15), (x + tw + 15, y), color, -1)
                    cv2.putText(display_frame, label, (x + 7, y - 7),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            if predictions:
                info = f"Recognizing {len(predictions)} face(s)\n\n"
                for i, pred in enumerate(predictions):
                    info += f"Face #{i + 1}:\n{pred['emotion_heb']}\n{pred['confidence']:.1f}%\n\n"
                self.faces_info.config(text=info, fg="#f1f5f9")
            else:
                self.faces_info.config(text="No faces detected", fg="#64748b")

            self.show_frame(display_frame)

        if self.recognizing and self.running:
            self.root.after(80, self.update_recognition)

    def show_frame(self, frame):
        """הצגת פריים - תיקון: שמירה על aspect ratio"""
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        label_w = self.video_label.winfo_width()
        label_h = self.video_label.winfo_height()

        if label_w > 1 and label_h > 1:
            # שמור על aspect ratio
            img.thumbnail((label_w, label_h), Image.Resampling.LANCZOS)

        imgtk = ImageTk.PhotoImage(img)
        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)

    def train_model(self):
        """אימון המודל - רק על פנים!"""
        images, labels = [], []

        total_count = 0
        for i, folder in enumerate(EMOTION_FOLDERS):
            folder_path = os.path.join(BASE_DIR, folder)
            count = 0
            for file in os.listdir(folder_path):
                if file.lower().endswith(('.jpg', '.png', '.jpeg')):
                    path = os.path.join(folder_path, file)
                    img = cv2.imread(path)
                    if img is not None:
                        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                        images.append(img)
                        labels.append(i)
                        count += 1
            print(f"📂 {folder}: {count} images")
            total_count += count

        if total_count < 20:
            messagebox.showerror("Error",
                                 f"Only {total_count} samples found!\n"
                                 f"Need at least 20 total samples.\n\n"
                                 f"Each emotion should have 3-5+ samples.\n"
                                 f"Current distribution above.")
            return

        X = np.array(images) / 255.0
        y = to_categorical(labels, num_classes=len(EMOTIONS))
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = build_model()

        early = EarlyStopping(monitor='val_accuracy', patience=8, restore_best_weights=True, verbose=1)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-7, verbose=1)

        self.status_label.config(text=f"Training on {total_count} face samples...", fg="#f59e0b")
        self.root.update()

        try:
            history = model.fit(
                X_train, y_train,
                validation_data=(X_test, y_test),
                epochs=50,  # יותר epochs
                batch_size=16,  # batch קטן יותר
                callbacks=[early, reduce_lr],
                verbose=1
            )

            model.save(MODEL_PATH)
            self.model = model
            self.model_trained = True

            acc = history.history['accuracy'][-1] * 100
            val_acc = history.history['val_accuracy'][-1] * 100

            self.status_label.config(text="Training complete!", fg="#10b981")
            self.recognize_btn.config(state=tk.NORMAL)

            messagebox.showinfo("Training Complete!",
                                f"Model trained successfully!\n\n"
                                f"Training Accuracy: {acc:.1f}%\n"
                                f"Validation Accuracy: {val_acc:.1f}%\n\n"
                                f"Total samples: {total_count}\n"
                                f"Model learns only from faces (96x96)\n"
                                f"Background is ignored!")

        except Exception as e:
            messagebox.showerror("Training Error", f"Training failed:\n{str(e)}")
            self.status_label.config(text="Training failed", fg="#ef4444")

    def stop_camera(self):
        """עצירת מצלמה"""
        self.running = False
        self.recognizing = False
        self.frozen = False

        if self.cap:
            self.cap.release()

        self.video_label.config(image='', bg="#000000")
        self.freeze_btn.config(state=tk.DISABLED, bg="#64748b")
        self.status_label.config(text="Camera stopped", fg="#64748b")
        self.faces_info.config(text="Camera not active", fg="#64748b")
        self.emotion_frame.pack_forget()

    def __del__(self):
        """סגירה נקייה"""
        self.running = False
        if self.cap and self.cap.isOpened():
            self.cap.release()


if __name__ == "__main__":
    root = tk.Tk()
    app = EmotionApp(root)
    root.mainloop()