import tkinter as tk
from tkinter import filedialog, Toplevel
from PIL import Image, ImageTk
import numpy as np

class PixelMatrixApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Pixel Matrix Filter")
        self.root.geometry("1000x700")
        self.image = None
        self.resized_image = None
        self.scale = 8
        self.pixel_size = 64

        # Canvas ראשי להצגת התמונה
        self.img_canvas = tk.Canvas(self.root, width=512, height=512)
        self.img_canvas.pack(side=tk.LEFT, padx=20, pady=20)
        self.img_canvas.bind("<Button-1>", self.on_pixel_click)

        # פאנל ימני
        self.right_panel = tk.Frame(self.root)
        self.right_panel.pack(side=tk.RIGHT, padx=20, fill=tk.Y)

        self.load_button = tk.Button(self.right_panel, text="Load Image", font=("Arial", 18, "bold"), command=self.load_image)
        self.load_button.pack(pady=20)

        self.rgb_label = tk.Label(self.right_panel, text="Pixel RGB: ---", font=("Arial", 18))
        self.rgb_label.pack(pady=20)

        self.create_text_inputs()

        self.go_button = tk.Button(self.right_panel, text="Go!", font=("Arial", 20, "bold"), command=self.apply_filter, bg="#4CAF50", fg="white")
        self.go_button.pack(pady=30)

    def create_text_inputs(self):
        self.entries = {}
        for color in ["Red", "Green", "Blue"]:
            label = tk.Label(self.right_panel, text=f"{color} value (0-255):", font=("Arial", 18))
            label.pack(pady=(10, 5))
            entry = tk.Entry(self.right_panel, font=("Arial", 18), width=6, justify='center')
            entry.insert(0, "255")
            entry.pack(pady=(0, 15))
            self.entries[color] = entry

    def load_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.png *.jpeg")])
        if file_path:
            self.image = Image.open(file_path).convert("RGB")
            self.image = self.image.resize((self.pixel_size, self.pixel_size), Image.NEAREST)
            self.draw_image()

    def draw_image(self):
        self.resized_image = self.image.resize((self.pixel_size * self.scale, self.pixel_size * self.scale), Image.NEAREST)
        self.tk_img = ImageTk.PhotoImage(self.resized_image)
        self.img_canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_img)

    def on_pixel_click(self, event):
        if self.image:
            x = event.x // self.scale
            y = event.y // self.scale
            if 0 <= x < self.pixel_size and 0 <= y < self.pixel_size:
                pixel = self.image.getpixel((x, y))
                self.rgb_label.config(text=f"Pixel RGB: {pixel}")

    def apply_filter(self):
        if self.image is None:
            return

        try:
            r_mult = int(self.entries["Red"].get()) / 255.0
            g_mult = int(self.entries["Green"].get()) / 255.0
            b_mult = int(self.entries["Blue"].get()) / 255.0
        except ValueError:
            self.rgb_label.config(text="Invalid RGB values!")
            return

        img_array = np.array(self.image).astype(np.float32)
        img_array[..., 0] *= r_mult
        img_array[..., 1] *= g_mult
        img_array[..., 2] *= b_mult
        img_array = np.clip(img_array, 0, 255).astype(np.uint8)

        filtered_img = Image.fromarray(img_array)
        self.show_filtered_image_window(filtered_img)

    def show_filtered_image_window(self, img):
        win = Toplevel(self.root)
        win.title("Filtered Image")

        # גודל תצוגה חדש
        display_width = 600
        display_height = 600

        canvas = tk.Canvas(win, width=display_width, height=display_height, bg="black")
        canvas.pack(pady=(10, 5))

        large_img = img.resize((display_width, display_height), Image.NEAREST)
        tk_img = ImageTk.PhotoImage(large_img)
        canvas.create_image(0, 0, anchor=tk.NW, image=tk_img)
        canvas.image = tk_img

        # כיתוב הדרכה
        info_label = tk.Label(win, text="Click on the image to see pixel RGB values",
                              font=("Arial", 14), fg="white", bg="black")
        info_label.pack(pady=(10, 0))

        rgb_result_label = tk.Label(win, text="Pixel RGB: ---",
                                    font=("Arial", 16), bg="white", width=25)
        rgb_result_label.pack(pady=(10, 15))

        # שמירה של התמונה לחישוב ערכי RGB
        self.filtered_image = img.copy()

        def on_click(event):
            x = int(event.x * self.filtered_image.width / display_width)
            y = int(event.y * self.filtered_image.height / display_height)
            if 0 <= x < self.filtered_image.width and 0 <= y < self.filtered_image.height:
                rgb = self.filtered_image.getpixel((x, y))
                rgb_result_label.config(text=f"Pixel ({x},{y}) RGB: {rgb}")

        canvas.bind("<Button-1>", on_click)

if __name__ == "__main__":
    root = tk.Tk()
    app = PixelMatrixApp(root)
    root.mainloop()
