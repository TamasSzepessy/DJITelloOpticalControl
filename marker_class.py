import numpy as np
from plot3d import Plotting
from timeit import default_timer as timer
import time

class Markers():
    def __init__(self):
        self.ids=[]
        self.tvec_o=[]
        self.rvec_o=[]
        self.marker_end=[]
        self.marker_terminating=[]

        self.iterateImgGet=31
        self.OpenedFile=False

        self.plotter = Plotting()

    def appendMarker(self, m_id, tvec, rvec):
        if m_id not in self.ids:
            self.ids.append(m_id)
            self.tvec_o.append(tvec)
            self.rvec_o.append(rvec)
            self.marker_end.append(0)
            self.marker_terminating.append(False)

    def lostMarker(self, seenMarkerList):
        for i in range(len(self.ids)):
            if self.ids[i] not in seenMarkerList and not self.marker_terminating[i]:
                #print("terminator")
                self.marker_end[i]=timer()
                self.marker_terminating[i]=True
            elif self.ids[i] in seenMarkerList and self.marker_terminating[i]:
                #print("ide is belépett")
                self.marker_terminating[i]=False

    def delMarker(self):
        toDelete=[]
        for i in range(len(self.ids)):
            if self.marker_terminating[i] and timer()-self.marker_end[i]>0.5:
                toDelete.append(i)

        toDelete.sort(reverse=True)

        for i in range(len(toDelete)):
            ind=toDelete[i]
            #print(str(ind)+" deleted")
            self.ids.pop(ind)
            self.tvec_o.pop(ind)
            self.rvec_o.pop(ind)
            self.marker_end.pop(ind)
            self.marker_terminating.pop(ind)

    def nullCoords(self):
        self.ids=[]
        self.tvec_o=[]
        self.rvec_o=[]
        self.marker_end=[]
        self.marker_terminating=[]     

    def writePos(self, id_list, tvecs, rvecs, ImageGet, origFrame):
        length=len(id_list)
        dtv, drv=self.dCoords(id_list,tvecs,rvecs)
        with open("results/marker_coords.csv", "a") as marker_coords:
                dtv=np.round(dtv,4)
                drv=np.round(drv,4)
                coords_string=str(-dtv[0][0])+";"+str(-dtv[0][2])+";"+str(dtv[0][1])+";"+str(-drv[0][0])+";"+str(-drv[0][1])+";"+str(-drv[0][2])+";"
                marker_coords.write(str(length)+";"+coords_string+"\n")

        # if ImageGet:
        #     self.iterateImgGet=0

        # if self.iterateImgGet<30:
        #     if self.iterateImgGet==0:
        #         self.start=timer()
        #         self.t=[0]
        #         self.tvec_all=dtv
        #         self.rvec_all=drv
        #     else:
        #         self.t=np.append(self.t,[timer()-self.start],axis=0)
        #         self.tvec_all=np.append(self.tvec_all,dtv,axis=0)
        #         self.rvec_all=np.append(self.rvec_all,drv,axis=0)

        #     cv2.imwrite("results/img_"+str(self.iterateImgGet)+".jpg", origFrame)

        #     self.iterateImgGet=self.iterateImgGet+1

        # if self.iterateImgGet==30:
        #     timestr = time.strftime("%Y%m%d_%H%M%S")
        #     np.savez("results/movement"+timestr, t=self.t, tvecs=self.tvec_all, rvecs=self.rvec_all)
        #     # print("saved")
        #     # print(self.t)
        #     # print(np.round(self.tvec_all,4))
        #     # print(np.round(self.rvec_all,4))
        #     self.iterateImgGet=self.iterateImgGet+1

    def dCoords(self, id_list, tvecs, rvecs):
        length=len(id_list)
        dtv=np.zeros((1,3))
        drv=np.zeros((1,3))
        for i in range(length):
            ind = self.ids.index(id_list[i])
            dtv=dtv+(tvecs[i]-self.tvec_o[ind])
            drv=drv+(rvecs[i]-self.rvec_o[ind])
        if length>0:
            dtv=dtv/length
            drv=drv/length

        return dtv, drv

    def getMov(self, id_list, tvecs, rvecs, CoordGet):
        if CoordGet and not self.OpenedFile:
            dtv, drv=self.dCoords(id_list,tvecs,rvecs)
            self.start=timer()
            self.t=[0]
            self.tvec_all=dtv
            self.rvec_all=drv
            self.OpenedFile=True
        elif CoordGet and self.OpenedFile:
            dtv, drv=self.dCoords(id_list,tvecs,rvecs)
            self.t=np.append(self.t,[timer()-self.start],axis=0)
            self.tvec_all=np.append(self.tvec_all,dtv,axis=0)
            self.rvec_all=np.append(self.rvec_all,drv,axis=0)
        elif not CoordGet and self.OpenedFile:
            timestr = time.strftime("%Y%m%d_%H%M%S")
            np.savez("results/movement_"+timestr, t=self.t, tvecs=self.tvec_all, rvecs=self.rvec_all)
            self.OpenedFile=False
            print("saved")
            self.plotter.plotout("results/movement_"+timestr+".npz")