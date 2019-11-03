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
from targeter import TargetDefine

# Edge to screen ratio (for filtering)
EDGE = 0.02
# Chessboard edge length in meters
CHB_SIDE = 0.0254
# Marker edge length in meters
MARKER_SIDE = 0.11
# Time delay
DELAY = 1.5
# Error allowed
ERROR = 0.15

class Camera():
    def __init__(self, S, dir_queue, cam_data, getCoords_event, navigate_event, END_event):
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.cam_data = cam_data

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

        # drone navigation
        self.speed = S
        self.amplify = 10
        self.dir_queue = dir_queue
        self.getCoords_event = getCoords_event
        self.navigate_event = navigate_event
        self.END_event = END_event
        self.resetCoords = False
        self.t_lost = 0.
        self.t_inPos = 0.
        self.last_marker_pos = 1.
        self.beenThere = []
        self.TargetID = 1
        self.findNew = False
        self.MarkerTarget = TargetDefine()

        # controller
        self.yaw_pid = PID(0.1, 0.00001, 0.001)
        self.v_pid = PID(0.5, 0.00001, 0.0001)
        self.vz_pid = PID(0.8, 0.00001, 0.0001)
        self.TargetPos = np.array([[0., 0., 1., 0.]])

        # for aruco markers
        self.markerEdge=MARKER_SIDE # ArUco marker edge length in meters
        self.seenMarkers=Markers(MARKER_SIDE, self.getCoords_event)
        self.getFirst = True

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

    def aruco(self, frame, CoordGet, CoordReset, angles_tof):
        # Get the calibrated camera matrices
        if self.not_loaded:
            with np.load(self.cam_data) as X:
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

        if CoordReset or self.resetCoords:
            print("Coordinates reset")
            self.resetCoords = False
            self.seenMarkers.nullCoords()

        if self.getCoords_event.is_set():
            CoordGet = True

        if np.all(ids != None):
            ### IDs found
            # Pose estimation with marker edge length
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, self.markerEdge, self.mtx, self.dist)

            for i in range(0, ids.size):
                cv2.aruco.drawAxis(frame, self.mtx, self.dist, rvecs[i], tvecs[i], 0.1)  # Draw axis

                id_list.append(ids[i][0])

            # Draw square around the markers
            cv2.aruco.drawDetectedMarkers(frame, corners)
            
            if self.navigate_event.is_set():
                if not self.findNew:
                    frame = self.drawCenter(frame, id_list, corners, w, h)
                    self.navigateToMarker(id_list, tvecs, rvecs)
                else:
                    self.dir_queue.put([0, 0, 0, self.speed*2])
                    for ID in id_list:
                        if ID not in self.beenThere:
                            self.TargetID = ID
                            self.TargetPos = self.MarkerTarget.changeTarget(ID)
                            self.findNew = False
                            break

                if not self.getFirst:
                    # Reject markers, which have corners on the edges of the frame
                    id_list, tvecs, rvecs = self.filterCorners(w, h, corners, id_list, tvecs, rvecs)

                    if len(id_list) > 0:
                        angles = np.array([[angles_tof[0], angles_tof[1], angles_tof[2]]])
                        tof = angles_tof[3]
                        self.seenMarkers.appendMarker(id_list, tvecs, rvecs, angles, tof)

                        #_, _, plotIMG = self.seenMarkers.getCoords(id_list, tvecs, rvecs)
                        self.seenMarkers.getMov(id_list, tvecs, rvecs, angles)

                    if not self.getCoords_event.is_set():
                        self.getFirst = True
        else:
            ### No IDs found
            cv2.putText(frame, "No Ids", (0,64), self.font, 1, (0,0,255),2,cv2.LINE_AA)
            if timer()-self.t_lost > 2:
                if self.last_marker_pos >= 0:
                    self.dir_queue.put([0, 0, 0, self.speed*2])
                else:
                    self.dir_queue.put([0, 0, 0, -self.speed*2])

        return frame

    # Look for origin marker and set directions
    def navigateToMarker(self, seen_id_list, tvecs, rvecs):
        if self.TargetID not in seen_id_list:
            if timer()-self.t_lost > 2:
                if self.last_marker_pos >= 0:
                    self.dir_queue.put([0, 0, 0, self.speed*2])
                else:
                    self.dir_queue.put([0, 0, 0, -self.speed*2])
        else:
            # select needed vectors
            ind = seen_id_list.index(self.TargetID)
            tvec = tvecs[ind]
            rvec = rvecs[ind]
            # only selected vectors from now
            self.last_marker_pos = tvec[0][0]
            rvec = tf.rotationVectorToEulerAngles(rvec)*180/math.pi
            # flip the yaw angle if marker is upside down
            if abs(rvec[0][2]) > 90:
                rvec[0][1] = -rvec[0][1]
            # print(tvec)
            # print(rvec)

            directions = [0., 0., 0., 0.]
            A = self.amplify*self.speed

            err_yaw = rvec[0][1] - self.TargetPos[0][3]
            directions[3] = self.speed/2*self.yaw_pid.control(err_yaw)

            # tvecs in camera coordinates
            err_x = self.TargetPos[0][0] - tvec[0][0]
            directions[0] = -A*self.v_pid.control(err_x)
            err_y = self.TargetPos[0][2] - tvec[0][2]
            directions[1] = -A*self.v_pid.control(err_y)
            err_z = self.TargetPos[0][1] - tvec[0][1]
            directions[2] = A*self.vz_pid.control(err_z)

            if abs(err_x) < ERROR and abs(err_y) < ERROR and abs(err_z) < 0.1 and abs(err_yaw) < 5:
                if timer()-self.t_inPos > 1:
                    if self.TargetID == 1:
                        self.getFirst = False
                        print("Positioned to origin, begin navigation")
                        self.resetCoords = True
                        self.getCoords_event.set()
                    if self.TargetID == 50:
                        print("End of flight")
                        self.getCoords_event.clear()
                        self.resetNavigators()
                        self.navigate_event.clear()
                        self.END_event.set()
                    if not self.getFirst:
                        self.beenThere.append(self.TargetID)
                        self.changeObjective(seen_id_list, tvecs)
                        if not self.findNew:
                            self.TargetPos = self.MarkerTarget.changeTarget(self.TargetID)
                            # reset PIDs
                            self.yaw_pid.reset()
                            self.v_pid.reset()
                            self.vz_pid.reset()
                            print("Target changed to "+str(self.TargetID))
                        else:
                            print("Searching for new marker...") 
            else:
                self.t_inPos = timer()

            self.t_lost = timer()
            self.dir_queue.put(directions)

    def resetNavigators(self):
        self.beenThere = []
        self.TargetID = 1
        self.yaw_pid.reset()
        self.v_pid.reset()
        self.vz_pid.reset()
        self.t_lost = 0.
        self.t_inPos = 0.
        self.last_marker_pos = 1.
        self.TargetPos = np.array([[0., 0., 0.8, 0.]])
    
    def changeObjective(self, seen_id_list, tvecs):
        length = len(seen_id_list)
        dist_minimum = 10000
        min_ind = self.TargetID
        for i in range(length):
            if np.linalg.norm(tvecs[i]) < dist_minimum and seen_id_list[i] not in self.beenThere:
                dist_minimum = np.linalg.norm(tvecs[i])
                min_ind = seen_id_list[i]

        if self.TargetID == min_ind:
            self.findNew = True
        else:
            self.TargetID = min_ind

    def drawCenter(self, frame, seen_id_list, corners, w, h):
        if self.TargetID not in seen_id_list:
            pass
        else:
            ind = seen_id_list.index(self.TargetID)
            cx = int((corners[ind][0][0][0]+corners[ind][0][1][0]+corners[ind][0][2][0]+corners[ind][0][3][0])/4)
            cy = int((corners[ind][0][0][1]+corners[ind][0][1][1]+corners[ind][0][2][1]+corners[ind][0][3][1])/4)
            cv2.line(frame, (int(w/2), int(h/2)), (cx, cy), (0,255,255), 3)
        
        return frame

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
getCoords_event = threading.Event()
getCoords_event.clear()
cam = Camera(15, direc_queue, 'camcalib_webcam.npz', getCoords_event)
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
        #print([x, y, z, yaw])

    c = cv2.waitKey(1)

    resetCoords=False

    if c == 27:
        break
    if c == ord("c") or c == ord("C"):
        if getCoords:
            getCoords = False
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
