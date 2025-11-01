import tkinter as tk
import cv2
import numpy as np
import random
from PIL import Image, ImageTk

class CameraApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Camera Filters")
        self.root.geometry("800x600")
        self.root.configure(bg='white')

        self.current_filter = 'bw'
        self.cap = None
        self.running = False

        # כפתורים
        self.button_frame = tk.Frame(self.root, bg='white')
        self.button_frame.pack(pady=10)

        filters = [
            ("Black & White", 'bw'),
            ("Red", 'red'),
            ("Green", 'green'),
            ("Blue", 'blue'),
            ("Surprise", 'surprise')
        ]

        for label, mode in filters:
            btn = tk.Button(self.button_frame, text=label, width=12, height=2,
                            font=("Arial", 12, "bold"), command=lambda m=mode: self.set_filter(m))
            btn.pack(side=tk.LEFT, padx=10)

        self.stop_button = tk.Button(self.root, text="Stop Camera", width=15, height=2,
                                     font=("Arial", 12), bg='red', fg='white', command=self.stop_camera)
        self.stop_button.pack(pady=10)

        self.video_label = tk.Label(self.root)
        self.video_label.pack()

        self.set_filter('bw')  # להתחיל עם שחור לבן

    def set_filter(self, mode):
        self.current_filter = mode
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
            if ret:
                filtered = self.apply_filter(frame, self.current_filter)
                img = cv2.cvtColor(filtered, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(img)
                imgtk = ImageTk.PhotoImage(image=img)
                self.video_label.imgtk = imgtk
                self.video_label.configure(image=imgtk)

        # תזמן את הקריאה הבאה (כל 20 מילי־שניות ≈ 50fps)
        if self.running:
            self.root.after(20, self.update_frame)

    def apply_filter(self, frame, mode):
        if mode == 'bw':
            return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        elif mode == 'red':
            b, g, r = cv2.split(frame)
            return cv2.merge([b*0, g*0, r])
        elif mode == 'green':
            b, g, r = cv2.split(frame)
            return cv2.merge([b*0, g, r*0])
        elif mode == 'blue':
            b, g, r = cv2.split(frame)
            return cv2.merge([b, g*0, r*0])
        elif mode == 'surprise':
            rb = random.uniform(0.5, 2.0)
            rg = random.uniform(0.5, 2.0)
            rr = random.uniform(0.5, 2.0)
            b, g, r = cv2.split(frame)
            b = np.clip(b * rb, 0, 255).astype(np.uint8)
            g = np.clip(g * rg, 0, 255).astype(np.uint8)
            r = np.clip(r * rr, 0, 255).astype(np.uint8)
            return cv2.merge([b, g, r])
        return frame

# הפעלת התוכנית
if __name__ == "__main__":
    root = tk.Tk()
    app = CameraApp(root)
    root.mainloop()
