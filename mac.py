import time
import cv2
import numpy as np
import tkinter as tk
from tkinter import messagebox
from threading import Thread
import queue
import pyautogui

# Load Haar cascades
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_eye.xml')

# Webcam setup
cap = cv2.VideoCapture(0)

# Calibration data
calibration_data = {"neutral": None, "up": None, "down": None}
threshold = 20
is_running = False
frame_queue = queue.Queue()


def detect_pupil(eye_image):
    blurred = cv2.GaussianBlur(eye_image, (5, 5), 0)
    _, thresholded = cv2.threshold(blurred, 50, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(
        thresholded, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        (x, y), radius = cv2.minEnclosingCircle(largest_contour)
        return int(x), int(y)
    return None


def determine_scroll_action(pupil_position):
    if not pupil_position or not all(calibration_data.values()):
        return None
    distances = {
        pos: np.linalg.norm(np.array(pupil_position) -
                            np.array(calibration_data[pos]))
        for pos in calibration_data
    }
    closest = min(distances, key=distances.get)
    if distances[closest] < threshold:
        if closest == "up":
            return 100  # Scroll up by 100 pixels
        elif closest == "down":
            return -100  # Scroll down by 100 pixels
    return 0  # No scrolling


def calibrate_eye_position(position, canvas, instruction_label):
    instruction_label.config(
        text=f"Look {position.upper()} and press 'c' in the camera window.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame.")
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.3, minNeighbors=5)
        for (x, y, w, h) in faces:
            face_region = gray[y:y + h, x:x + w]
            eyes = eye_cascade.detectMultiScale(
                face_region, scaleFactor=1.1, minNeighbors=10)
            for (ex, ey, ew, eh) in eyes:
                eye_region = face_region[ey:ey + eh, ex:ex + ew]
                pupil_center = detect_pupil(eye_region)
                cv2.rectangle(frame, (x + ex, y + ey),
                              (x + ex + ew, y + ey + eh), (0, 255, 0), 2)
                if pupil_center:
                    cv2.circle(
                        frame, (x + ex + pupil_center[0], y + ey + pupil_center[1]), 5, (0, 0, 255), -1)
        cv2.imshow("Calibration", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'):
            calibration_data[position] = pupil_center
            cv2.destroyWindow("Calibration")
            instruction_label.config(
                text=f"{position.capitalize()} calibrated!")
            return


def track():
    """Background thread to process video frames and detect pupil positions."""
    while is_running:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame.")
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.3, minNeighbors=5)
        for (x, y, w, h) in faces:
            face_region = gray[y:y + h, x:x + w]
            eyes = eye_cascade.detectMultiScale(
                face_region, scaleFactor=1.1, minNeighbors=10)
            for (ex, ey, ew, eh) in eyes:
                eye_region = face_region[ey:ey + eh, ex:ex + ew]
                pupil_position = detect_pupil(eye_region)
                scroll_amount = determine_scroll_action(pupil_position)
                if scroll_amount:
                    pyautogui.scroll(scroll_amount)

                    scroll_action_text = ""
                    # Update scroll action text based on scroll amount
                    if scroll_amount > 0:
                        scroll_action_text = "Scrolling Up"
                        pyautogui.scroll(scroll_amount)
                    elif scroll_amount < 0:
                        scroll_action_text = "Scrolling Down"
                        pyautogui.scroll(scroll_amount)
                    else:
                        scroll_action_text = "No Action"

                canvas.delete("all") # clear screen
                # Update the instruction_label with the current action
                instruction_label.config(text=f"Current Action: {
                                     scroll_action_text}")
                canvas.create_text(150, 100, text=f"Action: {
                    scroll_action_text}", font=("Helvetica", 20))

                if pupil_position:
                    cv2.circle(
                        frame, (x + ex + pupil_position[0], y + ey + pupil_position[1]), 5, (0, 0, 255), -1)
        # Send the processed frame to the main thread for display
        frame_queue.put(frame)
        time.sleep(0.1)  # Prevent excessive CPU usage


def start_tracking(canvas, instruction_label):
    """Start the tracking process."""
    global is_running
    is_running = True
    instruction_label.config(text="Tracking started. Press 'Stop' to end.")
    Thread(target=track, daemon=True).start()


def stop_tracking():
    """Stop the tracking process."""
    global is_running
    is_running = False
    print("Tracking stopped.")


def display_frames():
    """Display video frames from the queue on the main thread."""
    if not frame_queue.empty():
        frame = frame_queue.get()
        cv2.imshow("Eye Tracker", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        stop_tracking()
        root.destroy()
    root.after(10, display_frames)  # Repeat after 10ms


# GUI Setup
root = tk.Tk()
root.title("Eye Tracker")
canvas = tk.Canvas(root, width=300, height=200, bg="white")
canvas.pack()
instruction_label = tk.Label(
    root, text="Press 'Calibrate' to begin.", font=("Helvetica", 14))
instruction_label.pack()

tk.Button(root, text="Calibrate Neutral", command=lambda: calibrate_eye_position(
    "neutral", canvas, instruction_label)).pack()
tk.Button(root, text="Calibrate Up", command=lambda: calibrate_eye_position(
    "up", canvas, instruction_label)).pack()
tk.Button(root, text="Calibrate Down", command=lambda: calibrate_eye_position(
    "down", canvas, instruction_label)).pack()
tk.Button(root, text="Start Tracking", command=lambda: start_tracking(
    canvas, instruction_label)).pack()
tk.Button(root, text="Stop Tracking", command=stop_tracking).pack()

# Add safe shutdown logic
root.protocol("WM_DELETE_WINDOW", lambda: [stop_tracking(), root.destroy()])
root.after(10, display_frames)  # Start displaying frames
root.mainloop()

# Cleanup resources
cap.release()
cv2.destroyAllWindows()