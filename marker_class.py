import numpy as np
from plot3d import Plotting
from timeit import default_timer as timer
import time
import math
import transformations as tf
import cv2

class Markers():
    def __init__(self):
        # intiialise arrays
        self.ids = []
        self.tvec_origin = []
        self.rvec_origin = []

        # the first markers orientation
        self.orientation = np.zeros((2,3))

        # for logging
        self.OpenedFile=False

        # plotting
        self.plotter = Plotting()

    def appendMarker(self, seen_id_list, tvec, rvec):
        for n_id in seen_id_list:
            n_index = seen_id_list.index(n_id)
            if n_id not in self.ids and len(self.ids) == 0:
                self.ids.append(n_id)
                self.tvec_origin.append(tf.compensateTranslation(rvec[n_index], tvec[n_index], 0))
                self.rvec_origin.append(rvec[n_index])
                # convert rotation vector to Euler angles
                # x-y síktól függ a markerhelyzet
                # függőleges: piros -x (x - pitch), zöld y (z - yaw), kék z (y - roll)
                # vízszintes: piros x (x - pitch), zöld y (y - roll), kék z (z - yaw)
                rvec_Euler = tf.rotationVectorToEulerAngles(rvec[0][n_index])*180/math.pi
                if abs(rvec_Euler[0]) <= 150: # horizontal
                    self.orientation = np.array([[1, 1, -1],[0, 1, 2]])
                elif abs(rvec_Euler[0]) > 150: # vertical
                    self.orientation = np.array([[-1, 1, 1],[0, 2, 1]])

                print(self.orientation)
            elif n_id not in self.ids and len(self.ids) > 0 and len(seen_id_list) >= 2:
                for m_id in seen_id_list:
                    if m_id in self.ids: # n is to be added, m is already in list
                        m_index = seen_id_list.index(m_id)
                        tf.TransformBetweenMarkers(tvec[m_index], tvec[n_index], rvec[m_index], rvec[n_index])
                        self.ids.append(n_id)
                        # self.tvec_origin.append(1)
                        # self.rvec_origin.append(1)
                        # print("m_id: "+str(tf.compensateTranslation(rvec[n_index],tvec[n_index],0)))
                        # print(diff_o+tf.compensateTranslation(rvec[n_index],tvec_n,0))
                        break

    # Reset the coordinate system
    def nullCoords(self):
        self.ids = []
        self.tvec_origin = []
        self.rvec_origin = []
        self.orientation = []
        
    # Calculate the difference of seen marker from origin
    def dCoords(self, seen_id_list, tvecs, rvecs):
        length=len(seen_id_list)
        dtv=np.zeros((1,3))
        drv=np.zeros((1,3))
        for i in range(length):
            if seen_id_list[i] in self.ids:
                ind = self.ids.index(seen_id_list[i])
                if ind == 0:
                    dtv=dtv+(tf.compensateTranslation(rvecs[i],tvecs[i],0)-self.tvec_origin[ind])
                else:
                    tvecs[i]=tf.compensateTranslation(rvecs[i], tvecs[i], 0)
                    tvecs[i]=tf.compensateTranslation(self.rvec_origin[ind], tvecs[i], 1)
                    dtv=dtv+(tvecs[i]-self.tvec_origin[ind])

                drv=drv+(rvecs[i]-self.rvec_origin[ind])
        if length>0:
            dtv=dtv/length
            drv=drv/length

        return dtv, drv

    def getMov(self, seen_id_list, tvecs, rvecs, CoordGet):
        if CoordGet and not self.OpenedFile:
            dtv, drv=self.dCoords(seen_id_list,tvecs,rvecs)
            self.start=timer()
            self.t=[0]
            self.tvec_all=dtv
            self.rvec_all=drv
            self.OpenedFile=True
        elif CoordGet and self.OpenedFile:
            dtv, drv=self.dCoords(seen_id_list,tvecs,rvecs)
            self.t=np.append(self.t,[timer()-self.start],axis=0)
            self.tvec_all=np.append(self.tvec_all,dtv,axis=0)
            self.rvec_all=np.append(self.rvec_all,drv,axis=0)
        elif not CoordGet and self.OpenedFile:
            timestr = time.strftime("%Y%m%d_%H%M%S")
            np.savez("results/movement_"+timestr, t=self.t, tvecs=self.tvec_all,
                        rvecs=self.rvec_all, orientation=self.orientation)
            self.OpenedFile=False
            print("saved")
            self.plotter.plotout("results/movement_"+timestr+".npz")