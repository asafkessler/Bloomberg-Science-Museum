import tkinter as tk
from PIL import Image, ImageTk, ImageDraw, ImageFont
import cv2
import os
from deepface import DeepFace
from datetime import datetime
import threading
import numpy as np
from collections import Counter
import json

# בדיקת תיקיית data קיימת
DATA_FOLDER = "data"
OUTPUT_FOLDER = "emotion_art"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

emotion_dirs = ['angry', 'disgusted', 'fearful', 'happy', 'sad', 'surprised', 'neutral']

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
    'angry': (255, 0, 0),  # Red
    'disgust': (0, 128, 0),  # Green
    'fear': (128, 0, 128),  # Purple
    'happy': (255, 215, 0),  # Gold
    'sad': (0, 0, 255),  # Blue
    'surprise': (255, 255, 0),  # Yellow
    'neutral': (128, 128, 128)  # Gray
}

# תיאורים פואטיים לכל רגש
emotion_descriptions = {
    'angry': 'Flames of passion burning bright',
    'disgust': 'A bitter taste of discontent',
    'fear': 'Shadows dancing in the night',
    'happy': 'Sunshine breaking through the clouds',
    'sad': 'Tears falling like gentle rain',
    'surprise': 'Lightning striking unexpectedly',
    'neutral': 'Calm waters of tranquility'
}


class EmotionArtGenerator:
    def __init__(self, root):
        self.root = root
        self.root.title("🎨 AI Emotion Art Generator")

        # Get screen size
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        window_width = int(screen_width * 0.9)
        window_height = int(screen_height * 0.9)

        self.root.geometry(f"{window_width}x{window_height}")
        self.root.configure(bg="#0a0a0a")

        # Variables
        self.running = False
        self.cap = None
        self.freeze = False

        # מאגר רגשות - כל הרגשות שנתפסו
        self.emotion_timeline = []  # [{emotion, confidence, timestamp, face_id}]
        self.emotion_stats = {}  # {emotion: total_confidence}

        # Threading
        self.frame_lock = threading.Lock()
        self.latest_frame = None

        # UI
        self.create_ui()

    def create_ui(self):
        """Create user interface"""
        # Title
        title = tk.Label(self.root, text="🎨 AI Emotion Art Generator",
                         font=("Arial", 28, "bold"), bg="#0a0a0a", fg="#ff00ff")
        title.pack(pady=20)

        # Subtitle
        subtitle = tk.Label(self.root,
                            text="Capture the emotions of the moment and transform them into unique AI-generated art!",
                            font=("Arial", 14), bg="#0a0a0a", fg="#00ffff")
        subtitle.pack(pady=5)

        # Main container
        main_container = tk.Frame(self.root, bg="#0a0a0a")
        main_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

        # Left side - Video
        left_frame = tk.Frame(main_container, bg="#1a1a1a", relief=tk.RAISED, bd=3)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))

        video_title = tk.Label(left_frame, text="📹 Live Emotion Capture",
                               font=("Arial", 18, "bold"), bg="#1a1a1a", fg="#00ff88")
        video_title.pack(pady=10)

        self.video_label = tk.Label(left_frame, bg="black")
        self.video_label.pack(pady=10, padx=10)

        # Control buttons
        button_frame = tk.Frame(left_frame, bg="#1a1a1a")
        button_frame.pack(pady=15)

        self.start_btn = tk.Button(button_frame, text="▶️ Start Capture",
                                   command=self.start_camera,
                                   font=("Arial", 13, "bold"), bg="#27ae60", fg="white",
                                   width=15, height=2, relief=tk.RAISED, bd=4, cursor="hand2")
        self.start_btn.grid(row=0, column=0, padx=8)

        self.generate_btn = tk.Button(button_frame, text="🎨 Generate Art",
                                      command=self.generate_art,
                                      font=("Arial", 13, "bold"), bg="#9b59b6", fg="white",
                                      width=15, height=2, relief=tk.RAISED, bd=4, cursor="hand2",
                                      state=tk.DISABLED)
        self.generate_btn.grid(row=0, column=1, padx=8)

        self.stop_btn = tk.Button(button_frame, text="⏹️ Stop",
                                  command=self.stop_camera,
                                  font=("Arial", 13, "bold"), bg="#e74c3c", fg="white",
                                  width=15, height=2, relief=tk.RAISED, bd=4, cursor="hand2")
        self.stop_btn.grid(row=0, column=2, padx=8)

        # Right side - Statistics & Preview
        right_frame = tk.Frame(main_container, bg="#1a1a1a", relief=tk.RAISED, bd=3)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        stats_title = tk.Label(right_frame, text="📊 Emotion Analysis",
                               font=("Arial", 18, "bold"), bg="#1a1a1a", fg="#00ff88")
        stats_title.pack(pady=10)

        # Statistics display
        self.stats_frame = tk.Frame(right_frame, bg="#0d1117", relief=tk.SUNKEN, bd=2)
        self.stats_frame.pack(pady=10, padx=20, fill=tk.BOTH, expand=True)

        self.stats_canvas = tk.Canvas(self.stats_frame, bg="#0d1117", highlightthickness=0)
        self.stats_canvas.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Emotion counter
        counter_frame = tk.Frame(right_frame, bg="#1a1a1a")
        counter_frame.pack(pady=10)

        tk.Label(counter_frame, text="Total Emotions Captured:",
                 font=("Arial", 12, "bold"), bg="#1a1a1a", fg="#ffffff").pack()

        self.emotion_count_label = tk.Label(counter_frame, text="0",
                                            font=("Arial", 24, "bold"), bg="#1a1a1a", fg="#ffff00")
        self.emotion_count_label.pack()

        # Preview area
        preview_title = tk.Label(right_frame, text="🖼️ Art Preview",
                                 font=("Arial", 16, "bold"), bg="#1a1a1a", fg="#00ff88")
        preview_title.pack(pady=10)

        self.preview_label = tk.Label(right_frame, bg="#000000",
                                      text="Art will appear here\nafter generation",
                                      font=("Arial", 14), fg="#666666")
        self.preview_label.pack(pady=10, padx=20)

        # Status
        self.status_label = tk.Label(self.root, text="Click 'Start Capture' to begin your emotional journey",
                                     font=("Arial", 13), bg="#0a0a0a", fg="#f39c12")
        self.status_label.pack(pady=10)

    def start_camera(self):
        """Start camera and emotion capture"""
        if not self.running:
            self.running = True
            self.start_btn.config(state=tk.DISABLED)
            self.generate_btn.config(state=tk.NORMAL)

            # Reset emotion data
            self.emotion_timeline = []
            self.emotion_stats = {}
            self.update_stats_display()

            # Open webcam
            self.cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
            if not self.cap.isOpened():
                self.cap = cv2.VideoCapture(1)
            if not self.cap.isOpened():
                self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            if not self.cap.isOpened():
                self.cap = cv2.VideoCapture(0)

            if not self.cap.isOpened():
                self.status_label.config(text="❌ Cannot open webcam!")
                self.running = False
                return

            # Optimize settings
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            self.cap.set(cv2.CAP_PROP_FPS, 30)

            self.status_label.config(text="🎥 Capturing emotions... Express yourself freely!")

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

    def update_frame(self):
        """Update display with emotion detection"""
        if self.running:
            with self.frame_lock:
                if self.latest_frame is not None:
                    frame = self.latest_frame.copy()
                else:
                    frame = None

            if frame is not None:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                try:
                    # Analyze emotions
                    results = DeepFace.analyze(
                        rgb_frame,
                        actions=['emotion'],
                        enforce_detection=False,
                        detector_backend='opencv',
                        silent=True
                    )

                    if not isinstance(results, list):
                        results = [results]

                    # Process each detected face
                    for i, result in enumerate(results):
                        emotion = result.get('dominant_emotion', 'neutral')
                        emotions_dict = result.get('emotion', {})
                        confidence = emotions_dict.get(emotion, 0)
                        region = result.get('region', {})

                        # שמירת הרגש במאגר
                        self.emotion_timeline.append({
                            'emotion': emotion,
                            'confidence': confidence,
                            'timestamp': datetime.now(),
                            'emotions_dict': emotions_dict
                        })

                        # עדכון סטטיסטיקות
                        if emotion not in self.emotion_stats:
                            self.emotion_stats[emotion] = 0
                        self.emotion_stats[emotion] += confidence

                        # Draw on frame
                        if region:
                            x, y, w, h = region.get('x', 0), region.get('y', 0), region.get('w', 0), region.get('h', 0)

                            # Get color - convert from RGB to BGR for OpenCV
                            color_rgb = color_map.get(emotion, (128, 128, 128))
                            color_bgr = (color_rgb[2], color_rgb[1], color_rgb[0])

                            # Draw glowing effect
                            for thickness in range(8, 2, -1):
                                alpha = 0.3 + (8 - thickness) * 0.1
                                overlay = rgb_frame.copy()
                                cv2.rectangle(overlay, (x, y), (x + w, y + h), color_bgr, thickness)
                                cv2.addWeighted(overlay, alpha, rgb_frame, 1 - alpha, 0, rgb_frame)

                            # Draw emotion emoji
                            emoji = emoji_map.get(emotion, '😐')
                            cv2.putText(rgb_frame, emoji, (x + w // 2 - 30, y - 20),
                                        cv2.FONT_HERSHEY_SIMPLEX, 2, color_bgr, 3, cv2.LINE_AA)

                            # Draw emotion label with glow
                            label = f"{emotion.upper()}"
                            font_scale = 1.2
                            thickness_text = 3

                            (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX,
                                                                           font_scale, thickness_text)

                            # Background with transparency
                            overlay = rgb_frame.copy()
                            cv2.rectangle(overlay, (x, y - text_height - 20),
                                          (x + text_width + 20, y - 5), color_bgr, -1)
                            cv2.addWeighted(overlay, 0.7, rgb_frame, 0.3, 0, rgb_frame)

                            cv2.putText(rgb_frame, label, (x + 10, y - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255),
                                        thickness_text, cv2.LINE_AA)

                    # Update statistics display
                    self.update_stats_display()

                except Exception as e:
                    print(f"⚠️ Detection error: {e}")

                # Display frame
                img = Image.fromarray(rgb_frame)
                img.thumbnail((800, 600), Image.Resampling.LANCZOS)
                imgtk = ImageTk.PhotoImage(image=img)
                self.video_label.imgtk = imgtk
                self.video_label.configure(image=imgtk)

        self.root.after(100, self.update_frame)

    def update_stats_display(self):
        """Update statistics visualization"""
        # Clear canvas
        self.stats_canvas.delete("all")

        # Update counter
        self.emotion_count_label.config(text=str(len(self.emotion_timeline)))

        if not self.emotion_stats:
            self.stats_canvas.create_text(200, 150, text="No emotions detected yet...",
                                          font=("Arial", 14), fill="#666666")
            return

        # Calculate percentages
        total = sum(self.emotion_stats.values())
        sorted_emotions = sorted(self.emotion_stats.items(), key=lambda x: x[1], reverse=True)

        # Draw bars
        y_offset = 20
        max_width = 350

        for emotion, value in sorted_emotions:
            percentage = (value / total) * 100
            bar_width = (value / total) * max_width

            # Color
            color_rgb = color_map.get(emotion, (128, 128, 128))
            color_hex = '#{:02x}{:02x}{:02x}'.format(color_rgb[0], color_rgb[1], color_rgb[2])

            # Draw bar
            self.stats_canvas.create_rectangle(10, y_offset, 10 + bar_width, y_offset + 30,
                                               fill=color_hex, outline="white", width=2)

            # Emoji and label
            emoji = emoji_map.get(emotion, '😐')
            label = f"{emoji} {emotion.capitalize()}: {percentage:.1f}%"
            self.stats_canvas.create_text(max_width + 60, y_offset + 15,
                                          text=label, font=("Arial", 12, "bold"),
                                          fill="white", anchor="w")

            y_offset += 50

    def generate_art(self):
        """Generate artistic representation of captured emotions"""
        if not self.emotion_timeline:
            self.status_label.config(text="⚠️ No emotions captured yet! Start the camera first.")
            return

        self.status_label.config(text="🎨 Generating your unique emotion art... Please wait!")
        self.generate_btn.config(state=tk.DISABLED)

        # Generate in separate thread to not freeze UI
        threading.Thread(target=self._generate_art_thread, daemon=True).start()

    def _generate_art_thread(self):
        """Generate art in separate thread"""
        try:
            # Analyze emotions
            total_emotions = len(self.emotion_timeline)
            emotion_percentages = {}

            for emotion, value in self.emotion_stats.items():
                total = sum(self.emotion_stats.values())
                emotion_percentages[emotion] = (value / total) * 100

            # Sort by dominance
            dominant_emotions = sorted(emotion_percentages.items(),
                                       key=lambda x: x[1], reverse=True)[:3]

            # Create art
            art_image = self.create_emotion_art(dominant_emotions, total_emotions)

            # Save
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{OUTPUT_FOLDER}/emotion_art_{timestamp}.png"
            art_image.save(filename)

            # Display preview in small window
            preview = art_image.copy()
            preview.thumbnail((400, 400), Image.Resampling.LANCZOS)
            preview_tk = ImageTk.PhotoImage(preview)
            self.preview_label.config(image=preview_tk, text="")
            self.preview_label.image = preview_tk

            # Open fullscreen art display
            self.root.after(0, lambda: self.show_fullscreen_art(art_image, filename))

            # Update status
            self.root.after(0, lambda: self.status_label.config(
                text=f"✅ Art generated! Saved as: {filename}"))
            self.root.after(0, lambda: self.generate_btn.config(state=tk.NORMAL))

            print(f"🎨 Art saved: {filename}")

        except Exception as e:
            print(f"❌ Error generating art: {e}")
            self.root.after(0, lambda: self.status_label.config(
                text=f"❌ Error generating art: {str(e)}"))
            self.root.after(0, lambda: self.generate_btn.config(state=tk.NORMAL))

    def create_emotion_art(self, dominant_emotions, total_emotions):
        """Create artistic emoji representation of dominant emotion"""
        # Get the most dominant emotion
        main_emotion = dominant_emotions[0][0] if dominant_emotions else 'neutral'
        main_percentage = dominant_emotions[0][1] if dominant_emotions else 100

        # Create canvas
        width, height = 1200, 1200
        image = Image.new('RGB', (width, height), color='#0a0a0a')
        draw = ImageDraw.Draw(image)

        # Get emotion properties
        emoji = emoji_map.get(main_emotion, '😐')
        color = color_map.get(main_emotion, (128, 128, 128))
        description = emotion_descriptions.get(main_emotion, "An emotional moment")

        # Create gradient background based on emotion color
        for y in range(height):
            # Gradient from dark to emotion color and back
            factor = abs(y - height / 2) / (height / 2)
            r = int(color[0] * (1 - factor * 0.7))
            g = int(color[1] * (1 - factor * 0.7))
            b = int(color[2] * (1 - factor * 0.7))
            draw.line([(0, y), (width, y)], fill=(r, g, b))

        # Add radial glow effect around center
        center_x, center_y = width // 2, height // 2
        for radius in range(500, 0, -10):
            alpha = int(50 * (radius / 500))
            glow_color = tuple([min(255, c + alpha) for c in color])
            draw.ellipse([center_x - radius, center_y - radius,
                          center_x + radius, center_y + radius],
                         fill=None, outline=glow_color, width=2)

        # Draw decorative circles/particles around
        np.random.seed(42)
        for _ in range(30):
            x = np.random.randint(100, width - 100)
            y = np.random.randint(100, height - 100)
            r = np.random.randint(5, 20)
            alpha = np.random.randint(50, 150)
            particle_color = tuple([min(255, c + alpha) for c in color])
            draw.ellipse([x - r, y - r, x + r, y + r], fill=particle_color)

        # Try to use a large emoji font or draw the emoji artistically
        try:
            # Try to load a large font for emoji
            emoji_size = 500
            emoji_font = ImageFont.truetype("seguiemj.ttf", emoji_size)  # Windows emoji font
        except:
            try:
                emoji_font = ImageFont.truetype("Apple Color Emoji.ttc", 500)  # Mac emoji font
            except:
                try:
                    emoji_font = ImageFont.truetype("NotoColorEmoji.ttf", 500)  # Linux emoji font
                except:
                    # Fallback - create artistic text representation
                    emoji_font = None

        # Draw the giant emoji in center
        if emoji_font:
            # Draw emoji with shadow effect
            shadow_offset = 10
            # Shadow
            draw.text((center_x + shadow_offset, center_y + shadow_offset), emoji,
                      font=emoji_font, anchor="mm", fill=(0, 0, 0))
            # Main emoji
            draw.text((center_x, center_y), emoji, font=emoji_font, anchor="mm")
        else:
            # Fallback: draw artistic representation with shapes
            self._draw_artistic_emoji(draw, main_emotion, center_x, center_y, color)

        # Draw decorative frame
        frame_thickness = 15
        for i in range(frame_thickness):
            alpha = int(255 * (1 - i / frame_thickness))
            frame_color = tuple([min(255, int(c * (1 - i / frame_thickness * 0.5))) for c in color])
            draw.rectangle([i, i, width - i, height - i], outline=frame_color, width=3)

        # Add title at top
        try:
            title_font = ImageFont.truetype("arial.ttf", 70)
            subtitle_font = ImageFont.truetype("arial.ttf", 40)
            text_font = ImageFont.truetype("arial.ttf", 35)
        except:
            title_font = ImageFont.load_default()
            subtitle_font = ImageFont.load_default()
            text_font = ImageFont.load_default()

        # Main title with shadow
        title_text = main_emotion.upper()
        shadow_offset = 5
        draw.text((center_x + shadow_offset, 100 + shadow_offset), title_text,
                  font=title_font, anchor="mm", fill=(0, 0, 0))
        draw.text((center_x, 100), title_text, font=title_font, anchor="mm", fill='#ffffff')

        # Percentage
        percentage_text = f"{main_percentage:.1f}% Dominant"
        draw.text((center_x, 180), percentage_text, font=subtitle_font, anchor="mm", fill='#ffff00')

        # Poetic description at bottom
        draw.text((center_x, height - 180), description, font=text_font, anchor="mm", fill='#ffffff')

        # Statistics
        stats_text = f"Captured {total_emotions} emotional moments"
        draw.text((center_x, height - 120), stats_text, font=text_font, anchor="mm", fill='#00ffff')

        # Timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        small_font = ImageFont.truetype("arial.ttf", 25) if emoji_font else ImageFont.load_default()
        draw.text((center_x, height - 60), timestamp, font=small_font, anchor="mm", fill='#888888')

        # Add secondary emotions as small indicators
        if len(dominant_emotions) > 1:
            y_pos = height - 250
            draw.text((center_x, y_pos), "Other emotions detected:",
                      font=text_font, anchor="mm", fill='#aaaaaa')
            y_pos += 50
            for i in range(1, min(3, len(dominant_emotions))):
                emotion, percentage = dominant_emotions[i]
                emoji_small = emoji_map.get(emotion, '😐')
                text = f"{emoji_small} {emotion.capitalize()}: {percentage:.1f}%"
                draw.text((center_x, y_pos), text, font=text_font, anchor="mm", fill='#cccccc')
                y_pos += 45

        return image

    def _draw_artistic_emoji(self, draw, emotion, center_x, center_y, color):
        """Draw artistic emoji representation using shapes (fallback)"""
        size = 400

        # Draw face circle
        draw.ellipse([center_x - size, center_y - size,
                      center_x + size, center_y + size],
                     fill=color, outline='#ffffff', width=10)

        # Draw features based on emotion
        if emotion == 'happy':
            # Eyes
            eye_y = center_y - 100
            draw.ellipse([center_x - 150, eye_y - 40, center_x - 80, eye_y + 40], fill='#000000')
            draw.ellipse([center_x + 80, eye_y - 40, center_x + 150, eye_y + 40], fill='#000000')
            # Smile
            draw.arc([center_x - 200, center_y - 50, center_x + 200, center_y + 250],
                     start=0, end=180, fill='#000000', width=20)

        elif emotion == 'sad':
            # Eyes
            eye_y = center_y - 100
            draw.ellipse([center_x - 150, eye_y - 40, center_x - 80, eye_y + 40], fill='#000000')
            draw.ellipse([center_x + 80, eye_y - 40, center_x + 150, eye_y + 40], fill='#000000')
            # Frown
            draw.arc([center_x - 200, center_y + 50, center_x + 200, center_y + 350],
                     start=180, end=0, fill='#000000', width=20)

        elif emotion == 'angry':
            # Angry eyes
            eye_y = center_y - 80
            draw.polygon([center_x - 180, eye_y - 50, center_x - 60, eye_y,
                          center_x - 60, eye_y + 40, center_x - 140, eye_y + 40],
                         fill='#000000')
            draw.polygon([center_x + 180, eye_y - 50, center_x + 60, eye_y,
                          center_x + 60, eye_y + 40, center_x + 140, eye_y + 40],
                         fill='#000000')
            # Angry mouth
            draw.arc([center_x - 150, center_y + 80, center_x + 150, center_y + 220],
                     start=180, end=0, fill='#000000', width=15)

        elif emotion == 'surprise':
            # Wide eyes
            eye_y = center_y - 100
            draw.ellipse([center_x - 160, eye_y - 50, center_x - 60, eye_y + 50], fill='#000000')
            draw.ellipse([center_x + 60, eye_y - 50, center_x + 160, eye_y + 50], fill='#000000')
            # Open mouth
            draw.ellipse([center_x - 80, center_y + 80, center_x + 80, center_y + 220], fill='#000000')

        else:  # neutral, fear, disgust
            # Simple eyes
            eye_y = center_y - 100
            draw.ellipse([center_x - 140, eye_y - 30, center_x - 90, eye_y + 30], fill='#000000')
            draw.ellipse([center_x + 90, eye_y - 30, center_x + 140, eye_y + 30], fill='#000000')
            # Neutral mouth
            draw.line([center_x - 120, center_y + 150, center_x + 120, center_y + 150],
                      fill='#000000', width=15)

    def show_fullscreen_art(self, art_image, filename):
        """Display art in fullscreen window with spectacular effects"""
        # Create new fullscreen window
        art_window = tk.Toplevel(self.root)
        art_window.title("🎨 Your Emotion Masterpiece")

        # Gradient-like background
        art_window.configure(bg='#0a0a0a')

        # Make it fullscreen
        art_window.attributes('-fullscreen', True)
        art_window.attributes('-topmost', True)

        # Get screen size
        screen_width = art_window.winfo_screenwidth()
        screen_height = art_window.winfo_screenheight()

        # Create canvas for animated background
        bg_canvas = tk.Canvas(art_window, width=screen_width, height=screen_height,
                              bg='#0a0a0a', highlightthickness=0)
        bg_canvas.pack(fill=tk.BOTH, expand=True)

        # Draw animated stars in background
        stars = []
        for _ in range(50):
            x = np.random.randint(0, screen_width)
            y = np.random.randint(0, screen_height)
            size = np.random.randint(2, 6)
            star = bg_canvas.create_oval(x, y, x + size, y + size, fill='white', outline='')
            stars.append((star, np.random.uniform(0.3, 1.0)))

        def animate_stars(alpha_index=0):
            if art_window.winfo_exists():
                for star, base_alpha in stars:
                    alpha = base_alpha * (0.5 + 0.5 * np.sin(alpha_index * 0.1))
                    gray_val = int(255 * alpha)
                    color = '#{:02x}{:02x}{:02x}'.format(gray_val, gray_val, gray_val)
                    bg_canvas.itemconfig(star, fill=color)
                art_window.after(50, lambda: animate_stars(alpha_index + 1))

        animate_stars()

        # Create main frame on top of canvas
        main_frame = tk.Frame(bg_canvas, bg='#0a0a0a')
        bg_canvas.create_window(screen_width // 2, screen_height // 2, window=main_frame)

        # Title with glow effect
        title_frame = tk.Frame(main_frame, bg='#0a0a0a')
        title_frame.pack(pady=40)

        title = tk.Label(title_frame, text="✨ YOUR EMOTION MASTERPIECE ✨",
                         font=("Arial", 52, "bold"), bg='#0a0a0a', fg='#ff00ff')
        title.pack()

        subtitle = tk.Label(title_frame, text="A unique AI-generated visualization of your emotional journey",
                            font=("Arial", 26), bg='#0a0a0a', fg='#00ffff')
        subtitle.pack(pady=15)

        # Display the art - scale to fit screen while maintaining aspect ratio
        display_image = art_image.copy()

        # Calculate size to fit screen (leave space for title and buttons)
        max_width = screen_width - 300
        max_height = screen_height - 500

        # Scale image
        img_width, img_height = display_image.size
        scale = min(max_width / img_width, max_height / img_height)
        new_width = int(img_width * scale)
        new_height = int(img_height * scale)

        display_image = display_image.resize((new_width, new_height), Image.Resampling.LANCZOS)

        # Create image with glowing border
        image_container = tk.Frame(main_frame, bg='#0a0a0a')
        image_container.pack(pady=30)

        # Multi-layer border for glow effect
        for i in range(3, 0, -1):
            border_frame = tk.Frame(image_container,
                                    bg=f'#{int(255 * i / 3):02x}00{int(255 * i / 3):02x}',
                                    bd=i * 2, relief=tk.RAISED)
            if i == 3:
                border_frame.pack(padx=i * 3, pady=i * 3)
            else:
                border_frame.pack()

        photo = ImageTk.PhotoImage(display_image)
        image_label = tk.Label(border_frame, image=photo, bg='#000000')
        image_label.image = photo  # Keep reference
        image_label.pack()

        # Info text with icon
        info_frame = tk.Frame(main_frame, bg='#0a0a0a')
        info_frame.pack(pady=25)

        timestamp_display = datetime.now().strftime("%B %d, %Y at %H:%M:%S")
        info_text = f"💾 Saved as: {filename}"
        info_label = tk.Label(info_frame, text=info_text,
                              font=("Arial", 20, "bold"), bg='#0a0a0a', fg='#00ff88')
        info_label.pack()

        time_label = tk.Label(info_frame, text=f"Created on {timestamp_display}",
                              font=("Arial", 16), bg='#0a0a0a', fg='#888888')
        time_label.pack(pady=5)

        # Buttons with hover effects
        button_frame = tk.Frame(main_frame, bg='#0a0a0a')
        button_frame.pack(pady=35)

        def close_window():
            art_window.destroy()

        def save_again():
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            new_filename = f"{OUTPUT_FOLDER}/emotion_art_{timestamp}_copy.png"
            art_image.save(new_filename)
            info_label.config(text=f"💾 Also saved as: {new_filename}")

        def on_enter_close(e):
            close_btn.config(bg='#c0392b', relief=tk.SUNKEN)

        def on_leave_close(e):
            close_btn.config(bg='#e74c3c', relief=tk.RAISED)

        def on_enter_save(e):
            save_btn.config(bg='#2980b9', relief=tk.SUNKEN)

        def on_leave_save(e):
            save_btn.config(bg='#3498db', relief=tk.RAISED)

        close_btn = tk.Button(button_frame, text="✖️ Close (ESC)",
                              command=close_window,
                              font=("Arial", 22, "bold"), bg='#e74c3c', fg='white',
                              width=18, height=2, relief=tk.RAISED, bd=6, cursor="hand2")
        close_btn.pack(side=tk.LEFT, padx=25)
        close_btn.bind('<Enter>', on_enter_close)
        close_btn.bind('<Leave>', on_leave_close)

        save_btn = tk.Button(button_frame, text="💾 Save Another Copy",
                             command=save_again,
                             font=("Arial", 22, "bold"), bg='#3498db', fg='white',
                             width=18, height=2, relief=tk.RAISED, bd=6, cursor="hand2")
        save_btn.pack(side=tk.LEFT, padx=25)
        save_btn.bind('<Enter>', on_enter_save)
        save_btn.bind('<Leave>', on_leave_save)

        # Keyboard shortcuts
        art_window.bind('<Escape>', lambda e: close_window())
        art_window.bind('<space>', lambda e: close_window())
        art_window.bind('<Return>', lambda e: close_window())
        art_window.bind('q', lambda e: close_window())
        art_window.bind('Q', lambda e: close_window())

        # Add pulsing effect to title
        def pulse_title(color_index=0):
            if art_window.winfo_exists():
                colors = ['#ff00ff', '#ff3399', '#ff66cc', '#ff3399']
                title.config(fg=colors[color_index % len(colors)])
                art_window.after(400, lambda: pulse_title((color_index + 1) % len(colors)))

        pulse_title()

        # Pulse subtitle too
        def pulse_subtitle(color_index=0):
            if art_window.winfo_exists():
                colors = ['#00ffff', '#00ccff', '#0099ff', '#00ccff']
                subtitle.config(fg=colors[color_index % len(colors)])
                art_window.after(600, lambda: pulse_subtitle((color_index + 1) % len(colors)))

        pulse_subtitle()

        # Instructions with keyboard shortcuts
        instructions = tk.Label(main_frame,
                                text="⌨️ Keyboard: ESC / SPACE / ENTER / Q to close",
                                font=("Arial", 16), bg='#0a0a0a', fg='#666666')
        instructions.pack(pady=15)

    def stop_camera(self):
        """Stop camera"""
        self.running = False
        if self.cap:
            self.cap.release()
        self.video_label.config(image='', text="Camera stopped",
                                fg="#666666", font=("Arial", 16))
        self.start_btn.config(state=tk.NORMAL)
        self.status_label.config(text="Camera stopped. Click 'Generate Art' to create your masterpiece!")


if __name__ == "__main__":
    root = tk.Tk()
    app = EmotionArtGenerator(root)
    root.mainloop()