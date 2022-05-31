#import libraries
import cv2

#import Haarcascade classifier for face detection (frontal face only)
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

video_capture = cv2.VideoCapture(0)
#VideoCapture - denotes video is taken from a device
# 0 - the device is webcam (default)

#when running in windows, this causes as warning (but works) because of a bug in backend API
# in default backend MSMF (Microsoft Media Foundation). You can add the parameter cv2.CAP_DSHOW
#in the VideoCapture function to use DirectShow as backend 
#video_capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()   #read returns boolean, frame (image)
    #converting to gray_scale (since detectMultiScale function expects gray_scale frame input)
    image_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  #BGR to RGB  
    #since it is webcam, minimum size is 100x100 so that farther faces will not be detected
    #minimum number of neighbors for each rectange is set as 5
    detections = face_detector.detectMultiScale(image_gray, minSize=(100, 100), minNeighbors=5)

    # Draw a rectangle around the faces
    for (x, y, w, h) in detections: 
        #x - location of frame from left
        #y - location of frame from top (not from bottom)
        #w - width of frame
        #h - height of frame
        print(w, h)
        #rectangle color is green (BGR) in the parameter and the border width of rectangle is 2px
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Video', frame)

#exit if 'q' is pressed (lower case only)
#technically, waitkey return integer (ASCII) of keypressed
#comparing it with ASCII of letter 'q'
    if (cv2.waitKey(1) == ord('q')):
        break

video_capture.release() #close video capturing device (here, webcam)
cv2.destroyAllWindows() #close all windows created so far (here only one video capturing window)