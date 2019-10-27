import cv2
import numpy as np
from timeit import default_timer as timer
import time
from collections import deque
from marker_class import Markers
import transformations as tf
import math
import threading
import queue
from pid import PID

# Edge to screen ratio (for filtering)
EDGE = 0.02
# Chessboard edge length in meters
CHB_SIDE = 0.0254
# Marker edge length in meters
MARKER_SIDE = 0.0957
# Time delay
DELAY = 1.5

class Camera():
    def __init__(self, S, dir_queue):
        self.font = cv2.FONT_HERSHEY_SIMPLEX

        # for calibration
        self.db=0
        self.chbEdgeLength = CHB_SIDE

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
        self.not_loaded = True

        # for aruco markers
        self.markerEdge=MARKER_SIDE # ArUco marker edge length in meters
        self.seenMarkers=Markers(MARKER_SIDE)
        self.getFirst = True

        # drone speed
        self.speed = S
        self.amplify = 10
        self.dir_queue = dir_queue
        self.t_lost = 0.
        self.last_marker_pos = 1.

        # controller
        self.yaw_pid = PID(0.1, 0.00001, 0.001)
        self.v_pid = PID(0.3, 0.00001, 0.0001)
        self.vz_pid = PID(0.8, 0.00001, 0.0001)
        self.target = np.array([[0., 0., 0.8, 0.]])

    def calibrator(self,frame):
        if self.calib==False:
            gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            # find the chessboard corners
            ret, corners = cv2.findChessboardCorners(gray, (9,6),None)
        
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

    def aruco(self, frame, CoordGet, CoordReset):
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

        # List for all currently seen IDs
        id_list=[]

        if CoordReset:
            print("Coordinates reset")
            self.seenMarkers.nullCoords()

        if np.all(ids != None):
            ### IDs found
            # Pose estimation with marker edge length
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, self.markerEdge, self.mtx, self.dist)

            for i in range(0, ids.size):
                cv2.aruco.drawAxis(frame, self.mtx, self.dist, rvecs[i], tvecs[i], 0.1)  # Draw axis

                id_list.append(ids[i][0])

            # Draw square around the markers
            cv2.aruco.drawDetectedMarkers(frame, corners)

            if self.getFirst:
                if 1 in id_list:
                    ind = id_list.index(1)
                    self.lookForOrigin(tvecs[ind], rvecs[ind])
                    self.t_lost = timer()
                elif timer()-self.t_lost > 2:
                    if self.last_marker_pos >= 0:
                        self.dir_queue.put([0, 0, 0, self.speed*2])
                    else:
                        self.dir_queue.put([0, 0, 0, -self.speed*2])
            else:
                # Reject markers, which have corners on the edges of the frame
                id_list, tvecs, rvecs = self.filterCorners(w, h, corners, id_list, tvecs, rvecs)

                if len(id_list) > 0:
                    self.seenMarkers.appendMarker(id_list, tvecs, rvecs)

                    #_, _, plotIMG = self.seenMarkers.getCoords(id_list, tvecs, rvecs)
                    self.seenMarkers.getMov(id_list, tvecs, rvecs, CoordGet)
        else:
            #self.seenMarkers.getMov(id_list, np.zeros((1,3)), np.zeros((1,3)), CoordGet)
            ### No IDs found
            cv2.putText(frame, "No Ids", (0,64), self.font, 1, (0,0,255),2,cv2.LINE_AA)

            if timer()-self.t_lost > 2:
                if self.last_marker_pos >= 0:
                    self.dir_queue.put([0, 0, 0, self.speed*2])
                else:
                    self.dir_queue.put([0, 0, 0, -self.speed*2])

        return frame

    # Look for origin marker and set directions
    def lookForOrigin(self, tvec, rvec):
        self.last_marker_pos = tvec[0][0]
        # tvec = np.transpose(tvec)
        # rvec = np.transpose(rvec)
        # R = cv2.Rodrigues(rvec)[0]
        # tvec = -R.T.dot(tvec)
        # rvec = -R.T.dot(rvec)
        # tvec = np.transpose(tvec)
        # rvec = np.transpose(rvec)
        rvec = tf.rotationVectorToEulerAngles(rvec)*180/math.pi
        # print(tvec)
        # print(rvec)

        directions = np.zeros((1,4))
        A = self.amplify*self.speed

        err_yaw = rvec[0][1] - self.target[0][3]
        directions[0][3] = self.speed*self.yaw_pid.control(err_yaw)

        err_x = self.target[0][0] - tvec[0][0]
        directions[0][0] = -A*self.v_pid.control(err_x)
        err_y = self.target[0][2] - tvec[0][2]
        directions[0][1] = -A*self.v_pid.control(err_y)
        err_z = self.target[0][1] - tvec[0][1]
        directions[0][2] = A*self.vz_pid.control(err_z)

        self.dir_queue.put(directions[0].tolist())
        
    # Filter out marker corners on the frame's edges
    def filterCorners(self, w, h, corners, id_list, tvecs, rvecs):
        length = len(id_list)
        rejectIDs = []
        mask = np.ones(length)
        
        for i in range(length):
            for j in range(4):
                xc = corners[i][0][j][0]
                yc = corners[i][0][j][1]
                if (xc < w*EDGE) or (xc > w-w*EDGE) or (yc < h*EDGE) or (yc > h-h*EDGE):
                    rejectIDs.append(id_list[i])
                    mask[i]=0
                    #print("ID "+str(id_list[i])+" rejected")
                    break

        for i in range(len(rejectIDs)):
            id_list.remove(rejectIDs[i])
        
        okay = np.where(mask > 0)
        tvecs = np.r_[tvecs[okay]]
        rvecs = np.r_[rvecs[okay]]

        return id_list, tvecs, rvecs

'''
cap = cv2.VideoCapture(0)

direc_queue = queue.Queue()
cam = Camera(15, direc_queue)
getCoords = False
resetCoords = False

while True:
    ret, frame = cap.read()
    #cuFrame=cv2.cuda_GpuMat(frame)

    #frame, directions = cam.detectFace(frame)
    #print(directions)

    frame = cam.aruco(frame, getCoords, resetCoords)
    
    #frame=cuFrame.download()
    cv2.imshow('img', frame)

    if not direc_queue.empty():
        x, y, z, yaw = direc_queue.get()
        print([int(x), int(y), int(z), int(yaw)])

    c = cv2.waitKey(1)

    resetCoords=False

    if c == 27:
        break
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
'''

