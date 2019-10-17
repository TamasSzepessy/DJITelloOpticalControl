import numpy as np
from plot3d import Plotting
from timeit import default_timer as timer
import time
import math
import transformations as tf
import cv2

# limit for averaging
ALLOW_LIMIT = 30

class Markers():
    def __init__(self):
        # intiialise arrays
        self.ids = []
        self.tvec_origin = []
        self.rvec_origin = []
        self.dRot = []
        self.allow_use = []

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
                self.tvec_origin.append(np.array([[0, 0, 0]]))
                self.rvec_origin.append(rvec[n_index])
                self.dRot.append(np.array([[1,0,0],[0,1,0],[0,0,1]]))
                self.allow_use.append(ALLOW_LIMIT)
                # x-y síktól függ a markerhelyzet
                # függőleges: piros -x (x - pitch), zöld y (z - yaw), kék z (y - roll)
                # vízszintes: piros x (x - pitch), zöld y (y - roll), kék z (z - yaw)
                rvec_Euler = tf.rotationVectorToEulerAngles(rvec[0][n_index])*180/math.pi
                if abs(rvec_Euler[0]) <= 150: # horizontal
                    self.orientation = np.array([[1, 1, -1],[0, 1, 2]])
                elif abs(rvec_Euler[0]) > 150: # vertical
                    self.orientation = np.array([[-1, 1, 1],[0, 2, 1]])
                #print(self.orientation)
            elif n_id not in self.ids and len(self.ids) > 0 and len(seen_id_list) >= 2:
                self.ids.append(n_id)
                self.tvec_origin.append(np.array([[0, 0, 0]]))
                self.rvec_origin.append(np.array([[0, 0, 0]]))
                self.dRot.append(np.array([[0,0,0],[0,0,0],[0,0,0]]))
                self.allow_use.append(0)
                for m_id in seen_id_list:
                    if m_id in self.ids and m_id != n_id and self.allow_use[self.ids.index(m_id)]==ALLOW_LIMIT:
                        # n is to be added, m is already in list
                        m_index = seen_id_list.index(m_id)
                        m_index_list = self.ids.index(m_id)
                        n_index_list = self.ids.index(n_id)
                        t, R, a = tf.getTransformations(tvec[m_index], tvec[n_index], rvec[m_index], rvec[n_index],
                                                        self.tvec_origin[m_index_list], self.tvec_origin[n_index_list],
                                                        self.dRot[m_index_list], self.dRot[n_index_list],
                                                        self.allow_use[n_index_list], ALLOW_LIMIT)
                        self.tvec_origin[n_index_list] = t
                        self.dRot[n_index_list] = R
                        self.allow_use[n_index_list] = a
                        break
            elif n_id in self.ids and self.allow_use[self.ids.index(n_id)]<ALLOW_LIMIT:
                for m_id in seen_id_list:
                    if m_id in self.ids and m_id != n_id and self.allow_use[self.ids.index(m_id)]==ALLOW_LIMIT:
                        # n is to be added, m is already in list
                        m_index = seen_id_list.index(m_id)
                        m_index_list = self.ids.index(m_id)
                        n_index_list = self.ids.index(n_id)
                        t, R, a = tf.getTransformations(tvec[m_index], tvec[n_index], rvec[m_index], rvec[n_index],
                                                        self.tvec_origin[m_index_list], self.tvec_origin[n_index_list],
                                                        self.dRot[m_index_list], self.dRot[n_index_list],
                                                        self.allow_use[n_index_list], ALLOW_LIMIT)
                        self.tvec_origin[n_index_list] = t
                        self.dRot[n_index_list] = R
                        self.allow_use[n_index_list] = a
                        break

    def getCoords(self, seen_id_list, tvecs, rvecs):
        length = len(seen_id_list)
        len_diff = 0
        dtv = np.zeros((1,3))
        drv = np.zeros((1,3))
        for i in range(length):
            if seen_id_list[i] in self.ids:
                ind = self.ids.index(seen_id_list[i])
                if self.allow_use[ind] == ALLOW_LIMIT:
                    dtv = dtv+tf.calculatePos(tvecs[i], rvecs[i], self.tvec_origin[ind], self.dRot[ind])
                else:
                    len_diff = len_diff + 1

        length = length - len_diff
        if length>0:
            dtv=dtv/length
            drv=drv/length
        
        # print(dtv)
        # time.sleep(0.2)

        return dtv, drv
    
    # Reset the coordinate system
    def nullCoords(self):
        self.ids = []
        self.tvec_origin = []
        self.rvec_origin = []
        self.dRot = []
        self.allow_use = []
        self.orientation = np.zeros((2,3))
        
    # # Calculate the difference of seen marker from origin
    # def dCoords(self, seen_id_list, tvecs, rvecs):
    #     length = len(seen_id_list)
    #     len_diff = 0
    #     dtv = np.zeros((1,3))
    #     drv = np.zeros((1,3))
    #     for i in range(length):
    #         if seen_id_list[i] in self.ids:
    #             ind = self.ids.index(seen_id_list[i])
    #             if self.allow_use[ind] == ALLOW_LIMIT:
    #                 dtv = dtv+tf.calculatePos(tvecs[i], rvecs[i], self.tvec_origin[ind], self.dRot[ind])
    #             else:
    #                 len_diff = len_diff + 1
    #             # if ind == 0:
    #             #     dtv=dtv+(tf.compensateTranslation(rvecs[i],tvecs[i],0)-self.tvec_origin[0])
    #             # else:
    #             #     tvecs[i]=tf.compensateTranslation(rvecs[i], tvecs[i], 0)
    #             #     tvecs[i]=tf.compensateTranslation(self.rvec_origin[ind], tvecs[i], 1)
    #             #     dtv=dtv+(tvecs[i]-self.tvec_origin[ind])

    #             drv=drv+(rvecs[i]-self.rvec_origin[ind])

    #     length = length - len_diff
    #     if length>0:
    #         dtv=dtv/length
    #         drv=drv/length

    #     return dtv, drv

    def getMov(self, seen_id_list, tvecs, rvecs, CoordGet):
        if CoordGet and not self.OpenedFile:
            dtv, drv=self.getCoords(seen_id_list,tvecs,rvecs)
            self.start=timer()
            self.t=[0]
            self.tvec_all=dtv
            self.rvec_all=drv
            self.OpenedFile=True
        elif CoordGet and self.OpenedFile:
            dtv, drv=self.getCoords(seen_id_list,tvecs,rvecs)
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