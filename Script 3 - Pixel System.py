import tkinter as tk
from tkinter import Toplevel
import cv2
from PIL import Image, ImageTk
import numpy as np
import os


class PixelatedFilterBuilder:
    def __init__(self, root):
        self.root = root
        self.root.title("🎮 Pixelated Image Filter Builder (64×64)")

        # Get screen size
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        window_width = int(screen_width * 0.85)
        window_height = int(screen_height * 0.85)

        self.root.geometry(f"{window_width}x{window_height}")
        self.root.configure(bg='#2c3e50')

        # Variables
        self.current_image = None  # Original 64x64 PIL Image
        self.pixel_size = 64
        self.pixel_scale = 8  # Display scale

        # Main title
        title = tk.Label(self.root, text="🎨 Pixelated Filter Builder - 64×64 Resolution",
                         font=("Arial", 24, "bold"), bg='#2c3e50', fg='white')
        title.pack(pady=15)

        # Create layout
        self.create_main_container()

    def create_main_container(self):
        """Create main container with left and right panels"""
        main_container = tk.Frame(self.root, bg='#2c3e50')
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # Left side - Image display
        self.create_left_panel(main_container)

        # Right side - Controls
        self.create_right_panel(main_container)

    def create_left_panel(self, parent):
        """Create left panel with image display"""
        left_frame = tk.Frame(parent, bg='#34495e', relief=tk.RAISED, bd=3)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)

        # Title
        title = tk.Label(left_frame, text="📸 64×64 Pixelated Image",
                         font=("Arial", 16, "bold"), bg='#34495e', fg='white')
        title.pack(pady=10)

        # Status label
        self.status_label = tk.Label(left_frame,
                                     text="👉 Select an image source from the right panel",
                                     font=("Arial", 13, "bold"),
                                     bg='#34495e', fg='#f39c12')
        self.status_label.pack(pady=5)

        # Canvas for image
        self.canvas = tk.Canvas(left_frame, width=512, height=512, bg='black',
                                highlightthickness=2, highlightbackground='#3498db')
        self.canvas.pack(pady=10)
        self.canvas.bind("<Button-1>", self.on_pixel_click)

        # Pixel info
        self.pixel_info = tk.Label(left_frame, text="",
                                   font=("Courier", 12, "bold"),
                                   bg='#34495e', fg='#3498db', pady=10,
                                   wraplength=500)
        self.pixel_info.pack()

    def create_right_panel(self, parent):
        """Create right panel with controls"""
        right_frame = tk.Frame(parent, bg='#34495e', relief=tk.RAISED, bd=3, width=400)
        right_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=5)
        right_frame.pack_propagate(False)

        # Image source buttons
        source_frame = tk.LabelFrame(right_frame, text="📸 Image Sources",
                                     font=("Arial", 14, "bold"),
                                     bg='#34495e', fg='white', relief=tk.RAISED, bd=2)
        source_frame.pack(pady=10, padx=10, fill=tk.X)

        # Capture from camera
        capture_btn = tk.Button(source_frame, text="📷 Capture from Camera",
                                command=self.capture_from_camera,
                                font=("Arial", 12, "bold"),
                                bg='#27ae60', fg='white',
                                height=2, relief=tk.RAISED, bd=3, cursor='hand2')
        capture_btn.pack(pady=5, padx=10, fill=tk.X)

        # Load Tiger
        tiger_btn = tk.Button(source_frame, text="🐯 Load Tiger (64×64)",
                              command=self.load_tiger,
                              font=("Arial", 12, "bold"),
                              bg='#e67e22', fg='white',
                              height=2, relief=tk.RAISED, bd=3, cursor='hand2')
        tiger_btn.pack(pady=5, padx=10, fill=tk.X)

        # Load Einstein
        einstein_btn = tk.Button(source_frame, text="🧠 Load Einstein (64×64)",
                                 command=self.load_einstein,
                                 font=("Arial", 12, "bold"),
                                 bg='#3498db', fg='white',
                                 height=2, relief=tk.RAISED, bd=3, cursor='hand2')
        einstein_btn.pack(pady=5, padx=10, fill=tk.X)

        # Reset button
        reset_btn = tk.Button(source_frame, text="🔄 Reset / Clear",
                              command=self.reset_all,
                              font=("Arial", 12, "bold"),
                              bg='#c0392b', fg='white',
                              height=2, relief=tk.RAISED, bd=3, cursor='hand2')
        reset_btn.pack(pady=5, padx=10, fill=tk.X)

        # Filter builder
        filter_frame = tk.LabelFrame(right_frame, text="🎨 RGB Filter Builder",
                                     font=("Arial", 14, "bold"),
                                     bg='#34495e', fg='white', relief=tk.RAISED, bd=2)
        filter_frame.pack(pady=10, padx=10, fill=tk.BOTH, expand=True)

        # Info label
        info = tk.Label(filter_frame,
                        text="Set RGB values (0-255)\nto create custom filters:",
                        font=("Arial", 11), bg='#34495e', fg='#ecf0f1',
                        justify=tk.LEFT)
        info.pack(pady=10)

        # RGB input fields
        self.rgb_entries = {}
        colors = [
            ('Red 🔴', 'red', '#ff4444'),
            ('Green 🟢', 'green', '#44ff44'),
            ('Blue 🔵', 'blue', '#4444ff')
        ]

        for label_text, color_name, color_hex in colors:
            frame = tk.Frame(filter_frame, bg='#34495e')
            frame.pack(pady=8, padx=15, fill=tk.X)

            label = tk.Label(frame, text=label_text,
                             font=("Arial", 13, "bold"),
                             bg='#34495e', fg=color_hex, width=12, anchor='w')
            label.pack(side=tk.LEFT, padx=5)

            entry = tk.Entry(frame, font=("Arial", 14, "bold"),
                             width=8, justify='center',
                             bg='#ecf0f1', relief=tk.SUNKEN, bd=2)
            entry.insert(0, "255")
            entry.pack(side=tk.LEFT, padx=5)

            self.rgb_entries[color_name] = entry

        # Preset filters
        preset_label = tk.Label(filter_frame, text="Quick Presets:",
                                font=("Arial", 11, "bold"),
                                bg='#34495e', fg='#ecf0f1')
        preset_label.pack(pady=(15, 5))

        preset_buttons_frame = tk.Frame(filter_frame, bg='#34495e')
        preset_buttons_frame.pack(pady=5)

        presets = [
            ("Reset (255,255,255)", lambda: self.set_preset(255, 255, 255)),
            ("Only Red", lambda: self.set_preset(255, 0, 0)),
            ("Only Green", lambda: self.set_preset(0, 255, 0)),
            ("Only Blue", lambda: self.set_preset(0, 0, 255)),
            ("No Red", lambda: self.set_preset(0, 255, 255)),
            ("Dark (128,128,128)", lambda: self.set_preset(128, 128, 128))
        ]

        for i, (text, cmd) in enumerate(presets):
            btn = tk.Button(preset_buttons_frame, text=text, command=cmd,
                            font=("Arial", 9), bg='#95a5a6', fg='black',
                            width=18, relief=tk.RAISED, bd=2, cursor='hand2')
            btn.grid(row=i // 2, column=i % 2, padx=3, pady=3)

        # Apply filter button - BIGGER AND MORE VISIBLE
        apply_btn = tk.Button(filter_frame, text="✨ APPLY FILTER & SHOW RESULT ✨",
                              command=self.apply_filter,
                              font=("Arial", 15, "bold"),
                              bg='#27ae60', fg='white',
                              height=3, relief=tk.RAISED, bd=5, cursor='hand2',
                              activebackground='#229954')
        apply_btn.pack(pady=15, padx=15, fill=tk.X)

    def set_preset(self, r, g, b):
        """Set RGB preset values"""
        self.rgb_entries['red'].delete(0, tk.END)
        self.rgb_entries['red'].insert(0, str(r))
        self.rgb_entries['green'].delete(0, tk.END)
        self.rgb_entries['green'].insert(0, str(g))
        self.rgb_entries['blue'].delete(0, tk.END)
        self.rgb_entries['blue'].insert(0, str(b))

    def capture_from_camera(self):
        """Capture image from camera and pixelate to 64x64"""
        cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

        if not cap.isOpened():
            cap = cv2.VideoCapture(1)

        if not cap.isOpened():
            self.status_label.config(text="❌ Cannot open webcam!")
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

        ret, frame = cap.read()
        cap.release()

        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame_rgb)
            # Pixelate to 64x64
            pixelated = pil_img.resize((self.pixel_size, self.pixel_size), Image.Resampling.NEAREST)
            self.load_image(pixelated)
            self.status_label.config(text="📷 Camera captured! 64×64 pixels - Click to explore")

    def load_tiger(self):
        """Load tiger.jpg and pixelate"""
        if os.path.exists("tiger.jpg"):
            img = Image.open("tiger.jpg").convert("RGB")
            pixelated = img.resize((self.pixel_size, self.pixel_size), Image.Resampling.NEAREST)
            self.load_image(pixelated)
            self.status_label.config(text="🐯 Tiger loaded! 64×64 pixels - Click to explore")
        else:
            self.status_label.config(text="❌ tiger.jpg not found!")

    def load_einstein(self):
        """Load einstein.png and pixelate"""
        if os.path.exists("einstein.png"):
            img = Image.open("einstein.png").convert("RGB")
            pixelated = img.resize((self.pixel_size, self.pixel_size), Image.Resampling.NEAREST)
            self.load_image(pixelated)
            self.status_label.config(text="🧠 Einstein loaded! 64×64 pixels - Click to explore")
        else:
            self.status_label.config(text="❌ einstein.png not found!")

    def load_image(self, pil_image):
        """Load 64x64 PIL image"""
        self.current_image = pil_image
        self.display_image()

    def display_image(self):
        """Display 64x64 image scaled up"""
        if self.current_image is None:
            return

        # Scale up with nearest neighbor to show large pixels
        display_size = self.pixel_size * self.pixel_scale
        scaled = self.current_image.resize((display_size, display_size), Image.Resampling.NEAREST)

        self.tk_image = ImageTk.PhotoImage(scaled)

        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)

    def on_pixel_click(self, event):
        """Handle click on pixel"""
        if self.current_image is None:
            return

        # Calculate which pixel was clicked
        pixel_x = event.x // self.pixel_scale
        pixel_y = event.y // self.pixel_scale

        if 0 <= pixel_x < self.pixel_size and 0 <= pixel_y < self.pixel_size:
            r, g, b = self.current_image.getpixel((pixel_x, pixel_y))
            gray = int(0.299 * r + 0.587 * g + 0.114 * b)

            info = f"""Pixel Position: [{pixel_x}, {pixel_y}] (of 64×64 grid)
RGB Values: R={r}, G={g}, B={b}
Hex Color: #{r:02X}{g:02X}{b:02X}
Grayscale: {gray} (0=Black, 255=White)"""

            self.pixel_info.config(text=info)

    def apply_filter(self):
        """Apply RGB filter to current image"""
        if self.current_image is None:
            self.status_label.config(text="❌ No image loaded! Select an image source first.")
            return

        try:
            r_mult = int(self.rgb_entries['red'].get()) / 255.0
            g_mult = int(self.rgb_entries['green'].get()) / 255.0
            b_mult = int(self.rgb_entries['blue'].get()) / 255.0

            if not all(0 <= val <= 255 for val in [r_mult * 255, g_mult * 255, b_mult * 255]):
                raise ValueError

        except ValueError:
            self.status_label.config(text="❌ Invalid RGB values! Use numbers 0-255")
            return

        # Apply filter
        img_array = np.array(self.current_image).astype(np.float32)
        img_array[..., 0] *= r_mult  # R
        img_array[..., 1] *= g_mult  # G
        img_array[..., 2] *= b_mult  # B
        img_array = np.clip(img_array, 0, 255).astype(np.uint8)

        filtered_img = Image.fromarray(img_array)

        # Show result
        self.show_filtered_window(filtered_img)

    def show_filtered_window(self, filtered_img):
        """Show filtered image in new window"""
        win = Toplevel(self.root)
        win.title("✨ Filtered Result - 64×64 Pixels")
        win.geometry("700x800")
        win.configure(bg='#2c3e50')

        # Get filter values for display
        r_val = self.rgb_entries['red'].get()
        g_val = self.rgb_entries['green'].get()
        b_val = self.rgb_entries['blue'].get()

        title = tk.Label(win,
                         text=f"🎨 Filtered Image\nFilter Applied: RGB({r_val}, {g_val}, {b_val})",
                         font=("Arial", 16, "bold"), bg='#2c3e50', fg='white')
        title.pack(pady=15)

        # Canvas
        canvas = tk.Canvas(win, width=512, height=512, bg='black',
                           highlightthickness=2, highlightbackground='#e74c3c')
        canvas.pack(pady=10)

        # Display scaled image
        display_size = self.pixel_size * self.pixel_scale
        scaled = filtered_img.resize((display_size, display_size), Image.Resampling.NEAREST)
        tk_img = ImageTk.PhotoImage(scaled)

        canvas.create_image(0, 0, anchor=tk.NW, image=tk_img)
        canvas.image = tk_img

        # Pixel info
        pixel_label = tk.Label(win, text="👆 Click on any pixel to see its RGB values",
                               font=("Courier", 12, "bold"), bg='#2c3e50', fg='#3498db',
                               pady=10, wraplength=650)
        pixel_label.pack()

        def on_click(event):
            pixel_x = event.x // self.pixel_scale
            pixel_y = event.y // self.pixel_scale

            if 0 <= pixel_x < self.pixel_size and 0 <= pixel_y < self.pixel_size:
                r, g, b = filtered_img.getpixel((pixel_x, pixel_y))
                gray = int(0.299 * r + 0.587 * g + 0.114 * b)

                info = f"Pixel [{pixel_x},{pixel_y}] | RGB=({r},{g},{b}) | Hex=#{r:02X}{g:02X}{b:02X} | Gray={gray}"
                pixel_label.config(text=info)

        canvas.bind("<Button-1>", on_click)

    def reset_all(self):
        """Reset everything"""
        self.current_image = None
        self.canvas.delete("all")

        # Reset RGB values
        for entry in self.rgb_entries.values():
            entry.delete(0, tk.END)
            entry.insert(0, "255")

        self.status_label.config(text="👉 Select an image source from the right panel")
        self.pixel_info.config(text="")


if __name__ == "__main__":
    root = tk.Tk()
    app = PixelatedFilterBuilder(root)
    root.mainloop()