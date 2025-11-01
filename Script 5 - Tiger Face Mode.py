import tkinter as tk
import cv2
import numpy as np
from PIL import Image, ImageTk

class CameraApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Tiger Face Overlay")
        self.root.geometry("800x600")
        self.root.configure(bg='white')

        self.running = False
        self.cap = None
        self.mode = 'none'

        self.button_frame = tk.Frame(self.root, bg='white')
        self.button_frame.pack(pady=10)

        tiger_btn = tk.Button(self.button_frame, text="Tiger Mode", width=15, height=2,
                             font=("Arial", 12), bg='orange', command=self.tiger_mode)
        tiger_btn.pack(side=tk.LEFT, padx=5)

        self.stop_button = tk.Button(self.root, text="Stop Camera", width=15, height=2,
                                     font=("Arial", 12), bg='red', fg='white', command=self.stop_camera)
        self.stop_button.pack(pady=10)

        self.video_label = tk.Label(self.root)
        self.video_label.pack()

        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

        # Load the tiger face image (JPEG)
        self.tiger_img = cv2.imread("tiger.jpg")  # No alpha channel here

    def tiger_mode(self):
        self.mode = 'tiger'
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
            if ret and self.mode == 'tiger':
                frame = self.replace_faces_with_tiger(frame)

            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)

        if self.running:
            self.root.after(20, self.update_frame)

    def replace_faces_with_tiger(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            # Resize the tiger image to the size of the face
            resized_tiger = cv2.resize(self.tiger_img, (w, h))

            # Make sure the region of interest is within the frame bounds
            if y + h <= frame.shape[0] and x + w <= frame.shape[1]:
                frame[y:y+h, x:x+w] = resized_tiger

        return frame

# Run the app
if __name__ == "__main__":
    root = tk.Tk()
    app = CameraApp(root)
    root.mainloop()
