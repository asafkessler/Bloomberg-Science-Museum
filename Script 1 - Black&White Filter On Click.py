import tkinter as tk
import cv2
import threading
import numpy as np
from tkinter import scrolledtext


class EducationalCameraApp:
    def __init__(self, root):
        self.root = root
        self.root.title("🎓 Learn: Color to Black & White")
        self.root.geometry("600x700")
        self.root.configure(bg='#2c3e50')

        self.camera_running = False
        self.cap = None
        self.selected_pixel = None  # Store selected pixel coordinates

        # Title
        title = tk.Label(self.root, text="🎨 Color to Grayscale Converter",
                         font=("Arial", 20, "bold"), bg='#2c3e50', fg='white')
        title.pack(pady=20)

        # Explanation box
        self.create_explanation_box()

        # Buttons frame
        button_frame = tk.Frame(self.root, bg='#2c3e50')
        button_frame.pack(pady=20)

        # Start camera button
        self.start_btn = tk.Button(button_frame, text="▶️ Start Camera",
                                   command=self.start_camera,
                                   font=("Arial", 14, "bold"),
                                   bg='#27ae60', fg='white',
                                   width=15, height=2,
                                   relief=tk.RAISED, bd=4,
                                   cursor='hand2',
                                   activebackground='#229954')
        self.start_btn.pack(side=tk.LEFT, padx=10)

        # Stop camera button
        self.stop_btn = tk.Button(button_frame, text="⏹️ Stop Camera",
                                  command=self.stop_camera,
                                  font=("Arial", 14, "bold"),
                                  bg='#e74c3c', fg='white',
                                  width=15, height=2,
                                  relief=tk.RAISED, bd=4,
                                  cursor='hand2',
                                  activebackground='#c0392b',
                                  state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=10)

        # Instructions label
        self.instruction_label = tk.Label(self.root,
                                          text="💡 Tip: Click on any pixel in the COLOR image to see its calculation!",
                                          font=("Arial", 11, "bold"), bg='#2c3e50', fg='#f39c12',
                                          wraplength=550)
        self.instruction_label.pack(pady=5)

        # Real-time calculation display
        self.calc_frame = tk.Frame(self.root, bg='#34495e', relief=tk.RAISED, bd=3)
        self.calc_frame.pack(pady=10, padx=20, fill=tk.X)

        calc_title = tk.Label(self.calc_frame, text="🔢 Live Calculation",
                              font=("Arial", 12, "bold"), bg='#34495e', fg='#ecf0f1')
        calc_title.pack(pady=5)

        self.calc_text = tk.Label(self.calc_frame,
                                  text="Click 'Start Camera' then click on any pixel in the color image!",
                                  font=("Courier", 10), bg='#34495e', fg='#3498db',
                                  justify=tk.LEFT, wraplength=550)
        self.calc_text.pack(pady=10, padx=10)

    def create_explanation_box(self):
        """Create the educational explanation box"""
        explain_frame = tk.Frame(self.root, bg='white', relief=tk.SUNKEN, bd=3)
        explain_frame.pack(pady=10, padx=20, fill=tk.BOTH, expand=True)

        explain_title = tk.Label(explain_frame, text="📚 How Does Color → Grayscale Work?",
                                 font=("Arial", 13, "bold"), bg='white', fg='#2c3e50')
        explain_title.pack(pady=5)

        explanation = scrolledtext.ScrolledText(explain_frame, wrap=tk.WORD,
                                                font=("Arial", 11), height=10,
                                                bg='#ecf0f1', fg='#2c3e50',
                                                relief=tk.FLAT)
        explanation.pack(pady=5, padx=10, fill=tk.BOTH, expand=True)

        text = """כל פיקסל בתמונה צבעונית מורכב מ-3 ערכים:
🔴 Red (אדום) - ערך בין 0-255
🟢 Green (ירוק) - ערך בין 0-255  
🔵 Blue (כחול) - ערך בין 0-255

צבעים בסיסיים:
⬛ שחור = RGB(0, 0, 0) - כל הערוצים כבויים
⬜ לבן = RGB(255, 255, 255) - כל הערוצים דלוקים במלואם
⚪ אפור = RGB(128, 128, 128) - כל הערוצים באותה עוצמה

המרה לשחור-לבן (Grayscale):
אנחנו משתמשים בנוסחה מתמטית:

Gray = 0.299 × Red + 0.587 × Green + 0.114 × Blue

למה המשקלים האלה? 👀
העין האנושית רגישה בצורה שונה לצבעים:
• הכי רגישה לירוק (58.7%) - לכן המשקל הגבוה ביותר
• בינוני לאדום (29.9%)
• הכי פחות רגישה לכחול (11.4%)

דוגמה מעשית:
אם פיקסל הוא RGB(200, 100, 50):
Gray = 0.299×200 + 0.587×100 + 0.114×50
Gray = 59.8 + 58.7 + 5.7 = 124.2 ≈ 124

התוצאה: אפור בהיר עם ערך 124
(0 = שחור מוחלט, 255 = לבן מוחלט)

💡 טיפ: לחץ על פיקסלים שונים בתמונה הצבעונית כדי 
לראות איך הנוסחה עובדת על צבעים שונים!
"""

        explanation.insert(1.0, text)
        explanation.config(state=tk.DISABLED)

    def start_camera(self):
        """Start the camera in a separate thread"""
        if not self.camera_running:
            self.camera_running = True
            self.start_btn.config(state=tk.DISABLED)
            self.stop_btn.config(state=tk.NORMAL)
            threading.Thread(target=self.run_camera, daemon=True).start()

    def stop_camera(self):
        """Stop the camera"""
        self.camera_running = False
        if self.cap:
            self.cap.release()
            cv2.destroyAllWindows()
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.calc_text.config(text="Camera stopped. Click 'Start Camera' to begin again!")
        self.selected_pixel = None

    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse clicks on the image"""
        if event == cv2.EVENT_LBUTTONDOWN:
            frame, display_width = param
            h, w = frame.shape[:2]

            # Check if click is in the left half (color image area on display)
            if x < display_width:
                # Store the selected pixel coordinates (already in correct scale)
                self.selected_pixel = (x, y)

    def run_camera(self):
        """Run the camera with B&W filter and show calculations"""
        # Open webcam (1 = external webcam)
        self.cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(1)  # Fallback

        if not self.cap.isOpened():
            self.calc_text.config(text="❌ Cannot open webcam!")
            self.camera_running = False
            return

        # Optimize settings
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)

        window_name = 'Click on COLOR image to analyze pixels - Press Q to close'
        cv2.namedWindow(window_name)

        while self.camera_running:
            ret, frame = self.cap.read()
            if not ret:
                break

            # Apply grayscale filter
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Convert gray back to BGR for display
            gray_bgr = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2BGR)

            # Resize for side-by-side (larger size now)
            h, w = frame.shape[:2]
            frame_display = cv2.resize(frame, (w, h))
            gray_display = cv2.resize(gray_bgr, (w, h))

            # Draw selection circle if pixel is selected
            if self.selected_pixel:
                x, y = self.selected_pixel
                if 0 <= x < w and 0 <= y < h:
                    # Draw circle on color image
                    cv2.circle(frame_display, (x, y), 10, (0, 255, 0), 2)
                    cv2.circle(frame_display, (x, y), 2, (0, 255, 0), -1)

                    # Draw crosshair
                    cv2.line(frame_display, (x - 15, y), (x + 15, y), (0, 255, 0), 1)
                    cv2.line(frame_display, (x, y - 15), (x, y + 15), (0, 255, 0), 1)

                    # Show calculation
                    self.show_pixel_calculation(frame, x, y)

            # Add labels
            cv2.putText(frame_display, "ORIGINAL (Click here!)", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(gray_display, "GRAYSCALE", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            # Combine side by side
            combined = np.hstack((frame_display, gray_display))

            # Set mouse callback with frame and display width
            cv2.setMouseCallback(window_name, self.mouse_callback, (frame, w))

            cv2.imshow(window_name, combined)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.stop_camera()

    def show_pixel_calculation(self, frame, x, y):
        """Show calculation for selected pixel"""
        h, w = frame.shape[:2]

        # Validate coordinates
        if y >= h or x >= w:
            return

        # Get BGR values
        b, g, r = frame[y, x]

        # Calculate grayscale using the same formula OpenCV uses
        gray_value = int(0.299 * r + 0.587 * g + 0.114 * b)

        # Create calculation text
        calc = f"""Selected Pixel @ ({x}, {y}):

🔴 Red   = {r}
🟢 Green = {g}
🔵 Blue  = {b}

📐 Calculation:
Gray = (0.299 × {r}) + (0.587 × {g}) + (0.114 × {b})
Gray = {0.299 * r:.1f} + {0.587 * g:.1f} + {0.114 * b:.1f}
Gray = {gray_value}

✨ Result: Grayscale = {gray_value} (0=Black, 255=White)"""

        self.calc_text.config(text=calc)


# Create and run the app
if __name__ == "__main__":
    root = tk.Tk()
    app = EducationalCameraApp(root)
    root.mainloop()