import tkinter as tk
import cv2
import numpy as np
from PIL import Image, ImageTk

class CameraApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Love App")
        self.root.geometry("800x600")
        self.root.configure(bg='white')

        self.running = False
        self.cap = None
        self.mode = 'none'

        self.button_frame = tk.Frame(self.root, bg='white')
        self.button_frame.pack(pady=10)

        face_btn = tk.Button(self.button_frame, text="Face Love Mode", width=15, height=2,
                             font=("Arial", 12), bg='pink', command=self.face_love_mode)
        face_btn.pack(side=tk.LEFT, padx=5)

        self.stop_button = tk.Button(self.root, text="Stop Camera", width=15, height=2,
                                     font=("Arial", 12), bg='red', fg='white', command=self.stop_camera)
        self.stop_button.pack(pady=10)

        self.video_label = tk.Label(self.root)
        self.video_label.pack()

        # Load built-in face detector
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    def face_love_mode(self):
        self.mode = 'face'
        if not self.running:
            self.running = True
            self.cap = cv2.VideoCapture(0)
            self.update_frame()

    def stop_camera(self):
        self.running = False
        if self.cap:
            self.cap.release()
        self.video_label.config(image='')

    def update_frame(self):
        if self.running and self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret and self.mode == 'face':
                frame = self.detect_faces_and_draw_love(frame)

            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)

        if self.running:
            self.root.after(20, self.update_frame)

    def detect_faces_and_draw_love(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            # Draw a red heart shape (approximated by an ellipse)
            center = (x + w//2, y + h//2)
            axes = (w//2, h//2)
            cv2.ellipse(frame, center, axes, 0, 0, 360, (0, 0, 255), 2)

            # Add text "I love you" above the face
            cv2.putText(frame, "I love you", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, (0, 0, 255), 2, cv2.LINE_AA)

        return frame

# Run the app
if __name__ == "__main__":
    root = tk.Tk()
    app = CameraApp(root)
    root.mainloop()
