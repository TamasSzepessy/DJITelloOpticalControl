import cv2 as cv
import numpy as np
import time

class Calib:
    def __init__(self):
        self.db=0
        self.chbEdgeLength = 0.0254

        self.font = cv.FONT_HERSHEY_SIMPLEX
        self.start=True
        self.tstart=0
        self.calib=False

        # termination criteria
        self.criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(8,5,0)
        self.objp = np.zeros((6*9,3), np.float32)
        self.objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)*self.chbEdgeLength

        # arrays to store object points and image points from all the images
        self.objpoints = [] # 3d point in real world space
        self.imgpoints = [] # 2d points in image plane.

    def calibrator(self,frame):
        if self.calib==False:
            gray = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
            # find the chess board corners
            ret, corners = cv.findChessboardCorners(gray, (9,6),None)
        
        if self.db<20:
            # if found, add object points, image points (after refining them)
            if ret == True and self.start:
                self.start=False
                self.tstart=time.time()
                self.objpoints.append(self.objp)
            
                corners2 = cv.cornerSubPix(gray,corners,(11,11),(-1,-1),self.criteria)
                self.imgpoints.append(corners2)
            
                # draw and display the corners
                frame = cv.drawChessboardCorners(frame, (9,6), corners2,ret)
                #cv.imshow('frame',frame)
                self.db=self.db+1
            elif ret == True and time.time()-self.tstart>0.5:
                self.tstart=time.time()
                self.objpoints.append(self.objp)
            
                corners2 = cv.cornerSubPix(gray,corners,(11,11),(-1,-1),self.criteria)
                self.imgpoints.append(corners2)
            
                # draw and display the corners
                frame = cv.drawChessboardCorners(frame, (9,6), corners2,ret)
                #cv.imshow('frame',frame)
                self.db=self.db+1
            else:
                if ret==True:
                    corners2 = cv.cornerSubPix(gray,corners,(11,11),(-1,-1),self.criteria)
                    frame = cv.drawChessboardCorners(frame, (9,6), corners2,ret)
                else:                
                    cv.putText(frame, "Please show chessboard.", (0,64), self.font, 1, (0,0,255),2,cv.LINE_AA)
                
                #cv.imshow('frame',frame)
        else:
            if self.calib==False: # save the camera matrices first
                self.calib=True
                ret, self.mtx, self.dist, self.rvecs, self.tvecs = cv.calibrateCamera(self.objpoints, self.imgpoints, gray.shape[::-1],None,None)
                h, w = frame.shape[:2]
                self.newcameramtx, self.roi=cv.getOptimalNewCameraMatrix(self.mtx,self.dist,(w,h),1,(w,h))
                np.savez("camcalib", ret=ret, mtx=self.mtx, dist=self.dist, rvecs=self.rvecs, tvecs=self.tvecs)
            
            # undistort
            udst = cv.undistort(frame, self.mtx, self.dist, None, self.newcameramtx)
            # crop the image
            x,y,w,h = self.roi
            udst = udst[y:y+h, x:x+w]
            cv.putText(udst, "Camera calibrated.", (0,64), self.font, 1, (0,255,0),2,cv.LINE_AA)
            #cv.imshow('frame',udst)

        return frame