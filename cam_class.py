import cv2
import numpy as np
from timeit import default_timer as timer
import time
from collections import deque
from plot3d import Plotting
from marker_class import Markers
import transformations as tf
import math

class Camera():
    def __init__(self):
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
        self.markerEdge=0.0957 # ArUco marker edge length in meters
        self.seenMarkers=Markers()

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

    def writeBattery(self, frame, bat):
        w=frame.shape[1]
        h=frame.shape[0]
        if bat < 25:
            cv2.putText(frame, "Battery: "+str(bat), (w-170,h-10), self.font, 0.8, (0,0,255),2,cv2.LINE_AA)
        elif bat < 50:
            cv2.putText(frame, "Battery: "+str(bat), (w-170,h-10), self.font, 0.8, (0,255,255),2,cv2.LINE_AA)
        else:
            cv2.putText(frame, "Battery: "+str(bat), (w-170,h-10), self.font, 0.8, (0,255,0),2,cv2.LINE_AA)

        return frame

    def aruco(self, frame, ImageGet, CoordGet, CoordReset):
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
        #origFrame=np.copy(frame)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_7X7_100)
        parameters = cv2.aruco.DetectorParameters_create()

        # Detecting markers: get corners and IDs
        corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

        id_list=[]

        if CoordReset:
            print("coords reset")
            self.seenMarkers.nullCoords()        

        if np.all(ids != None):
            ### IDs found
            # Pose estimation with marker edge length
            rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(corners, self.markerEdge, self.mtx, self.dist)

            for i in range(0, ids.size):
                cv2.aruco.drawAxis(frame, self.mtx, self.dist, rvec[i], tvec[i], 0.1)  # Draw axis
                
                # correct translational values for rotation
                R, _ = cv2.Rodrigues(rvec[i])
                tvec[i]=np.matmul(tvec[i], R)
                # print(tvec[i])
                
                # convert rotation vector to Euler angles
                # x-y síktól függ a markerhelyzet
                # függőleges: piros -x (x - pitch), zöld y (z - yaw), kék z (y - roll)
                # vízszintes: piros x (x - pitch), zöld y (y - roll), kék z (z - yaw)
                rvec[i] = tf.rotationVectorToEulerAngles(rvec[i])*180/math.pi
                # print(rvec[i]*180/math.pi)

                id_list.append(ids[i][0])
                self.seenMarkers.appendMarker(ids[i][0],tvec[i],rvec[i])

            self.seenMarkers.getMov(id_list, tvec, rvec, CoordGet)
            # Draw square around the markers
            cv2.aruco.drawDetectedMarkers(frame, corners)
        else:
            self.seenMarkers.getMov(id_list, np.zeros((1,3)), np.zeros((1,3)), CoordGet)
            ### No IDs found
            cv2.putText(frame, "No Ids", (0,64), self.font, 1, (0,0,255),2,cv2.LINE_AA)

        return frame



cap = cv2.VideoCapture(0)

cam = Camera()
getImages=False
getCoords=False
resetCoords=False

while True:
    ret, frame = cap.read()
    #cuFrame=cv2.cuda_GpuMat(frame)

    #frame, directions = cam.detectFace(frame)
    #print(directions)

    frame = cam.aruco(frame, getImages, getCoords, resetCoords)
    
    #frame=cuFrame.download()
    cv2.imshow('img', frame)

    c = cv2.waitKey(1)

    resetCoords=False

    if c == 27:
        break
    if c == ord("g") or c == ord("G"):
        getImages = True
        resetCoords = True
        print("getim")
        continue
    if c == ord("c") or c == ord("C"):
        if getCoords:
            getCoords=False
        else:
            getCoords = True
            resetCoords = True
            print("getco")
        continue
    if c == ord("d") or c == ord("D"):
        resetCoords = True
        continue


cap.release()
cv2.destroyAllWindows()