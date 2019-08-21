import cv2
import numpy as np
from timeit import default_timer as timer

class Camera():
    def __init__(self):
        # Load the cascade
        self.face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        # For timer
        self.empty_start=0
        self.last_cx=2000

    def detectFace(self, frame):
        font = cv2.FONT_HERSHEY_SIMPLEX

        height, width, _ = frame.shape
        # Convert into grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Detect faces
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 10)
        # Directions vector for drone control
        directions = np.zeros(4)
        # Draw rectangle around the faces
        if len(faces)>0:          
            self.empty_start=timer()
            cx=0
            cy=0
            msg=""
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                cx=cx+x+w/2
                cy=cy+y+h/2

            cx=cx/len(faces)
            cy=cy/len(faces)

            # right-left
            if cx > width*0.6:
                msg=msg+"rgt, "
                directions[0]=1
            if cx < width*0.4:
                msg=msg+"lft, "
                directions[0]=-1
            # forward-backward
            w=faces[0][2]
            if w>width*0.2:
                msg=msg+"bck, "
                directions[1]=-1
            if w<width*0.15:
                msg=msg+"fwd, "
                directions[1]=1
            # up-down
            if cy < height*0.25:
                msg=msg+"up, "
                directions[2]=1
            if cy > height*0.45:
                msg=msg+"dwn"
                directions[2]=-1

            self.last_cx=cx

            cv2.rectangle(frame, (int(cx)-2, int(cy)-2), (int(cx)+4, int(cy)+4), (0, 255, 0), 4)
            
            cv2.putText(frame,msg,(10,height-15),font,1,(0,0,255),2)
        elif timer()-self.empty_start>2:
            if self.last_cx>width/2:
                directions[3]=1
            else:
                directions[3]=-1

        return frame, directions

# cap = cv2.VideoCapture(0)

# cam = Camera()

# while True:
#     # Read the input image
#     ret, frame = cap.read()

#     frame, directions = cam.detectFace(frame)

#     print(directions)

#     # Display the output
#     cv2.imshow('img', frame)

#     c = cv2.waitKey(1)

#     if c == 27:
#         break

# cap.release()
# # finally, close the window
# cv2.destroyAllWindows()