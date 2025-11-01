import tkinter as tk
import cv2
import threading

def open_camera_with_bw_filter():
    """Opens the webcam and applies a black-and-white (grayscale) filter."""
    cap = cv2.VideoCapture(0)  # 0 = default camera
    if not cap.isOpened():
        print("Cannot open camera")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame. Exiting...")
            break

        # Apply black-and-white (grayscale) filter
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Show the frame
        cv2.imshow('Black & White Camera', gray_frame)

        # Exit loop if user presses 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def on_button_click():
    """Starts the camera with BW filter in a separate thread to keep GUI responsive."""
    threading.Thread(target=open_camera_with_bw_filter, daemon=True).start()

# Create the main window
root = tk.Tk()
root.title("My Button Screen")
root.geometry("300x200")  # Set window size (width x height)

# Create a button widget
button = tk.Button(root, text="Click Me!", command=on_button_click)
button.pack(pady=50)  # Add some padding for better appearance

# Start the Tkinter event loop
root.mainloop()
