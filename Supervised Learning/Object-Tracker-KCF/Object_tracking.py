#import library
import cv2

tracker = cv2.TrackerKCF_create()   #create tracker
video = cv2.VideoCapture('street.mp4')    #capture the give file

ok, frame = video.read() #ok - boolean variable, frame - frame
#select ROI (Region of interest i.e, object to track) and hit enter or space to accept the selected values
bbox = cv2.selectROI(frame) #x,y,width,height (lets select the initial location of object of interest)
ok = tracker.init(frame, bbox)  #initialise the tracker with the initial location

#record video
size = (int(video.get(3)), int(video.get(4)))   #getting video dimensions
fps = video.get(cv2.CAP_PROP_FPS)   #getting FPS of the video
result = cv2.VideoWriter('Street_Tracker.avi', cv2.VideoWriter_fourcc(*'MJPG'),fps, size) #video writer
#filename - race_Tracker.avi
#cv2.VideoWriter_fourcc(*'MJPG') - compress the video in Motion JPEG format
#cv2.VideoWriter_fourcc(*'MJPG') - compress the video in Motion JPEG format; *'XVID' - better compression
#fps - frame per second
#size - dimension of the video

while(True):    #loop until object lost or video ends
    ok, frame = video.read()    #read a frame
    
    if not ok:  #if frame is not available (video ended)
        break
    
    ok, bbox = tracker.update(frame)    #update the tracker to the new position of the object
    
    if(ok): #if frame available after updating tracker
        (x,y,w,h) = [int(v) for v in bbox]  #get locations and dimensions of the object
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2, 1)   #draw rectangle around object; 1 - line type
    else:   #if frame not available
        cv2.putText(frame, 'Error', (100,80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)    #put text
    
    cv2.imshow('Tracker', frame)    #display the frame
    result.write(frame)     #write the frame to recording video file
    if(cv2.waitKey(10) & 0xFF == 27): #ESC key
        break
    
    
video.release() #close video capturing device (here, webcam)
cv2.destroyAllWindows() #close all windows created so far (here only one video capturing window)
result.release()    #release the video writer
