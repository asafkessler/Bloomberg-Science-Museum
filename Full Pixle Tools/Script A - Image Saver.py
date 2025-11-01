import tkinter as tk
import cv2
from PIL import Image, ImageTk
import os
from datetime import datetime
import winsound

# צליל (ל-Windows)
try:
    def play_shutter():
        winsound.MessageBeep()  # צליל מערכת פשוט
except ImportError:
    from playsound import playsound
    def play_shutter():
        # נדרשת קובץ shutter.wav בתיקייה הנוכחית
        playsound("shutter.wav")

class CameraCaptureApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Camera Capture")
        self.root.geometry("800x600")

        self.cap = cv2.VideoCapture(0)
        self.running = True

        self.video_label = tk.Label(self.root)
        self.video_label.pack()

        self.capture_button = tk.Button(self.root, text="Capture & Save",
                                        font=("Arial", 12), command=self.capture_image)
        self.capture_button.pack(pady=10)

        self.message_label = tk.Label(self.root, text="", font=("Arial", 12), fg="green")
        self.message_label.pack()

        self.quit_button = tk.Button(self.root, text="Quit", font=("Arial", 12),
                                     bg='red', fg='white', command=self.quit_app)
        self.quit_button.pack()

        self.update_frame()

    def update_frame(self):
        if self.running and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                self.last_frame = frame.copy()
                img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(img)
                imgtk = ImageTk.PhotoImage(image=img)
                self.video_label.imgtk = imgtk
                self.video_label.configure(image=imgtk)
        if self.running:
            self.root.after(20, self.update_frame)

    def capture_image(self):
        if hasattr(self, 'last_frame'):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"captured_{timestamp}.jpg"
            downloads_dir = os.path.join(os.path.expanduser("~"), "Downloads")
            save_path = os.path.join(downloads_dir, filename)

            cv2.imwrite(save_path, self.last_frame)
            print(f"✅ Image saved to: {save_path}")

            # צליל
            try:
                play_shutter()
            except:
                print("🔇 Failed to play sound")

            # הבהוב רקע זמני (לבן)
            original_bg = self.video_label.cget("background")
            self.video_label.configure(background='white')
            self.root.after(150, lambda: self.video_label.configure(background=original_bg))

            # הודעה זמנית
            self.message_label.config(text="✅ Image captured and saved!")
            self.root.after(2000, lambda: self.message_label.config(text=""))

    def quit_app(self):
        self.running = False
        if self.cap:
            self.cap.release()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = CameraCaptureApp(root)
    root.mainloop()
