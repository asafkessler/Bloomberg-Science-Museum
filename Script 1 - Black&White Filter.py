# This is a sample Python script.

import cv2
import tkinter as tk
from tkinter import Label
from PIL import Image, ImageTk

def update_frame():
    ret, frame = cap.read()
    if ret:
        # Apply a simple filter, for example converting to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Convert to ImageTk format
        img = ImageTk.PhotoImage(image=Image.fromarray(gray))

        # Update label with the new image
        label.imgtk = img
        label.config(image=img)

    # Repeat after 10 milliseconds
    root.after(10, update_frame)

# Set up the main Tkinter window
root = tk.Tk()
root.title("Real-Time Filter App")

# Set up the video capture (0 is usually the default webcam)
cap = cv2.VideoCapture(0)

# Set up a label to display the video feed
label = Label(root)
label.pack()

# Start the update loop
update_frame()

# Run the Tkinter event loop
root.mainloop()

# Release the video capture when the app is closed
cap.release()
