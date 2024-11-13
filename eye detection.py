# being able to detect the movement of the eyes 
# importing openCV and numpy 
import cv2 
import numpy as np 

# Store the reference images in memory
reference_images = {}

# Function to capture the neutral and downward gaze reference
def capture_gaze_reference(frame, neutral_position=False):
    height, width, _ = frame.shape
    if neutral_position:
        # Neutral gaze: Draw a dot at the center
        cv2.circle(frame, (width//2, height//2), 10, (0, 0, 255), -1)
        # Get text size
        text = "Look at dot, q when done"
        (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)

        # Center the text
        text_x = (width - text_width) // 2
        text_y = (height // 2) - 20

        cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # Save neutral gaze image
        reference_images['neutral'] = frame.copy()  
    else:
        # Downward gaze: Draw a dot at the bottom
        cv2.circle(frame, (width//2, height - 50), 10, (0, 0, 255), -1)
        # Get text size
        text = "Look at dot, q when done"
        (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)

        # Center the text
        text_x = (width - text_width) // 2
        text_y = (height // 2) - 20

        cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # Save downward gaze image
        reference_images['downward'] = frame.copy()  

    cv2.imshow("Frame", frame)
    cv2.waitKey(0)
    

# Prompt the user for screen resolution, with a default of 1920x1080
user_input = input("Enter screen resolution (e.g. 1920x1080, enter to use default): ")

# Set default resolution
screen_width, screen_height = (1920, 1080)

# If user provides input, split and update the resolution
if user_input:
    try:
        screen_width, screen_height = map(int, user_input.split('x'))
    except ValueError:
        print("Invalid input. Using default resolution.")

# Create a window and set its size
cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Frame", screen_width, screen_height)

# Start of the main program
cap = cv2.VideoCapture(0)

# Ask the user to look neutrally
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame")
        break
    capture_gaze_reference(frame, neutral_position=True)  # Neutral gaze reference
    break  # After neutral gaze capture, move to next step

# Ask the user to look downward
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame")
        break
    capture_gaze_reference(frame, neutral_position=False)  # Downward gaze reference
    break  # After downward gaze capture, move to main loop

# End of initialization of eyes

# using the Haar cascades pre-defined models 
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

# capturing the webcam 
# cap = cv2.VideoCapture("eye_webcam.flv")
# cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Falied to capture frame")
        break

    # converting to grey scale for haar cascades 
    gray_scale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # first detecting the face 
    face = face_cascade.detectMultiScale(gray_scale, scaleFactor=1.3, minNeighbors=5, minSize=(50,50))

    # Print number of faces detected
    # print(f"Number of faces detected: {len(face)}")

    # visualizing the face detection by drawing rectabgles 
    for(x,y,w,h) in face:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (255,0,0), 2)

        # extracting the face region 
        face_region = gray_scale[y:y+h, x:x+w]
        color_region = frame[y:y+h, x:x+w]

        # applying the eye detection to the face region 
        eyes = eye_cascade.detectMultiScale(face_region, scaleFactor=1.1, minNeighbors=10, minSize=(20,20))

        # draw visual around the eyes 
        for(x2, y2, w2, h2) in eyes:
            cv2.rectangle(color_region, (x2,y2), (x2+w2, y2 + h2), (0, 255,0), 2)

            # zooming into eye region 
            eye_region = color_region[y2:y2+h2, x2:x2+w2]
            zoomed_eye = cv2.resize(eye_region, (w2* 4, h2 *4))

            # display the zoomed in version in a seperate window 
            cv2.imshow("Zoomed eye", zoomed_eye)
    

    cv2.imshow("Frame", frame)
    # if cv2.getWindowProperty('frame', cv2.WND_PROP_VISIBLE) < 1:
    #     break
    # Wait for key press or window close
    key = cv2.waitKey(30) & 0xFF

    # Exit on pressing 'q' or when the window is closed
    if key == ord('q') or key == 27:
        break

cap.release()
cv2.destroyAllWindows()
