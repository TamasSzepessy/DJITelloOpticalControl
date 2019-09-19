import cv2
import numpy as np
from timeit import default_timer as timer
import time

class Camera():
    def __init__(self):
        # Load the cascade
        self.face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        # For timer
        self.empty_start=0
        self.last_cx=2000

        self.font = cv2.FONT_HERSHEY_SIMPLEX

        # for calibration
        self.db=0
        self.chbEdgeLength = 0.0254

        self.start=True
        self.tstart=0
        self.calib=False

        # termination criteria
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(8,5,0)
        self.objp = np.zeros((6*9,3), np.float32)
        self.objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)*self.chbEdgeLength

        # arrays to store object points and image points from all the images
        self.objpoints = [] # 3d point in real world space
        self.imgpoints = [] # 2d points in image plane.

        # for loading camera matrices
        self.not_loaded=True

        # for aruco markers
        self.markerEdge=0.0910 # ArUco marker edge length in meters

    def detectFace(self, frame):

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
            
            cv2.putText(frame,msg,(10,height-15),self.font,1,(0,0,255),2)
        elif timer()-self.empty_start>2:
            if self.last_cx>width/2:
                directions[3]=1
            else:
                directions[3]=-1

        return frame, directions

    def calibrator(self,frame):
        if self.calib==False:
            gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            # find the chessboard corners
            ret, corners = cv2.gpu.findChessboardCorners(gray, (9,6),None)
        
        if self.db<20:
            # if found, add object points, image points (after refining them)
            if ret == True and self.start:
                self.start=False
                self.tstart=time.time()
                self.objpoints.append(self.objp)
            
                corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),self.criteria)
                self.imgpoints.append(corners2)
            
                # draw and display the corners
                frame = cv2.drawChessboardCorners(frame, (9,6), corners2,ret)
                self.db=self.db+1
            elif ret == True and time.time()-self.tstart>0.5:
                self.tstart=time.time()
                self.objpoints.append(self.objp)
            
                corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),self.criteria)
                self.imgpoints.append(corners2)
            
                # draw and display the corners
                frame = cv2.drawChessboardCorners(frame, (9,6), corners2,ret)
                self.db=self.db+1
            else:
                if ret==True:
                    corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),self.criteria)
                    frame = cv2.drawChessboardCorners(frame, (9,6), corners2,ret)
                else:                
                    cv2.putText(frame, "Please show chessboard.", (0,64), self.font, 1, (0,0,255),2,cv2.LINE_AA)
        else:
            if self.calib==False: # save the camera matrices first
                self.calib=True
                ret, self.mtx, self.dist, self.rvecs, self.tvecs = cv2.calibrateCamera(self.objpoints, self.imgpoints, gray.shape[::-1],None,None)
                h, w = frame.shape[:2]
                self.newcameramtx, self.roi=cv2.getOptimalNewCameraMatrix(self.mtx,self.dist,(w,h),1,(w,h))
                np.savez("camcalib", ret=ret, mtx=self.mtx, dist=self.dist, rvecs=self.rvecs, tvecs=self.tvecs)
            
            # undistort
            frame = cv2.undistort(frame, self.mtx, self.dist, None, self.newcameramtx)
            # crop the image
            x,y,w,h = self.roi
            frame = frame[y:y+h, x:x+w]
            cv2.putText(frame, "Camera calibrated.", (0,64), self.font, 1, (0,255,0),2,cv2.LINE_AA)

        return frame

    def aruco(self, frame):
        # Get the calibrated camera matrices
        if self.not_loaded:
            with np.load('camcalib.npz') as X:
                self.mtx = X['mtx']
                self.dist = X['dist']
            self.not_loaded=False

        h, w = frame.shape[:2]
        newcameramtx, roi=cv2.getOptimalNewCameraMatrix(self.mtx,self.dist,(w,h),1,(w,h))

        # Undistort
        frame = cv2.undistort(frame, self.mtx, self.dist, None, newcameramtx)

        # Crop image
        x,y,w,h = roi
        frame = frame[y:y+h, x:x+w]

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_7X7_100)
        parameters = cv2.aruco.DetectorParameters_create()

        # Detecting markers: get corners and IDs
        corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

        if np.all(ids != None):
            ### IDs found
            # Pose estimation with marker edge length
            rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(corners, self.markerEdge, self.mtx, self.dist)
            
            # array of [x, y, z] coordinates of the markers
            transl=tvec.reshape(len(tvec),3)

            # Error map: List to store z coordinates
            markers_z_list = []

            for i in range(0, ids.size):
                cv2.aruco.drawAxis(frame, self.mtx, self.dist, rvec[i], tvec[i], 0.1)  # Draw axis
                
                # Z component of tvec to string
                strg = str(ids[i][0]) + ' z=' + str(round(transl[i][2],3))

                # Store z coordinates for error map
                markers_z_list.append(str(round(transl[i][2],3)))

                # Writing text to 4th corner
                cv2.putText(frame, strg, (int(corners[i][0][2][0]),int(corners[i][0][2][1])), self.font, 0.5, (255,0,0),1,cv2.LINE_AA)

            # Draw square around the markers
            cv2.aruco.drawDetectedMarkers(frame, corners)

            # Draw direction arrow for the first marker found
            # (ideally only 1 marker is seen while testing this)
            radius = 0.05
            x = transl[0][0]
            xstr = str(round(x,3))
            if x < -radius :
                arrow = "-->"
            elif x > radius :
                arrow = "<--"
            else:
                arrow = "OK"
            
            cv2.putText(frame, arrow + "  x=" + xstr, (0,64), self.font, 1, (0,255,0),2,cv2.LINE_AA)

        else:
            ### No IDs found
            cv2.putText(frame, "No Ids", (0,64), self.font, 1, (0,0,255),2,cv2.LINE_AA)

        return frame

        

cap = cv2.VideoCapture(0)

cam = Camera()

while True:
    ret, frame = cap.read()
    #cuFrame=cv2.cuda_GpuMat(frame)

    #frame, directions = cam.detectFace(frame)
    #print(directions)

    frame = cam.aruco(frame)
    
    #frame=cuFrame.download()
    cv2.imshow('img', frame)

    c = cv2.waitKey(1)

    if c == 27:
        break

cap.release()
cv2.destroyAllWindows()