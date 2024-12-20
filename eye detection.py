# being able to detect the movement of the eyes 
# importing openCV and numpy 
import cv2 
import numpy as np 

# using the Haar cascades pre-defined models 
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

# capturing the webcam 
# cap = cv2.VideoCapture("eye_webcam.flv")
cap = cv2.VideoCapture(0)

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
    print(f"Number of faces detected: {len(face)}")

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
