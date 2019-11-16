import numpy as np
import matplotlib
# matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
from matplotlib.animation import FuncAnimation
from scipy.spatial.transform import Rotation
import math
import cv2
import time
import csv
from pykalman import KalmanFilter

SHOW_ANG = False
SHOW_COMP = False
ANIM = False

AVG_MAX = 12
SMOOTHER = 5000
SET_EQUAL = False

FILE = 'flight_03'
ARUCO_PATH = 'results/MoCap/'+FILE+'.npz'
MOCAP_PATH = 'results/MoCap/'+FILE+'.csv'
START = 1000
STEP = 4

SHOW_START = 0

class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        FancyArrowPatch.draw(self, renderer)

class Plotting():
    def __init__(self, MARKER_SIDE):
        self.markerEdge = MARKER_SIDE
        self.fig = plt.figure(1)
        if ANIM:
            self.ax_AR = self.fig.add_subplot(111, projection='3d')
        else:
            self.ax_AR = self.fig.add_subplot(121, projection='3d')
            self.ax_MC = self.fig.add_subplot(122, projection='3d')
        
        self.bx_prev, self.by_prev, self.bz_prev = self.plotCoordSys(np.array([[0, 0, 0]]), np.array([[0.,0.,0.]]), False, 1)
        self.index = 0

    def animate(self, xp, yp, zp, rvec):        
        self.markerEdge = self.markerEdge*5
        self.xt, self.yt, self.zt = 0, 0, 0
        def update(frame):
            if frame is None: return
            else:
                if self.index < len(xp):
                    try:
                        self.bx_prev.remove()
                        self.by_prev.remove()
                        self.bz_prev.remove()
                    except:
                        pass
                    rvec_act = np.array([[float(rvec[self.index][0]),float(rvec[self.index][1]),float(rvec[self.index][2])]])
                    bx, by, bz = self.plotCoordSys(np.array([[xp[self.index],yp[self.index],zp[self.index]]]), rvec_act, True, 1)
                    self.ax_AR.add_artist(bx)
                    self.ax_AR.add_artist(by)
                    self.ax_AR.add_artist(bz)
                    self.bx_prev, self.by_prev, self.bz_prev = bx, by, bz
           
                    if (((self.index+1) % AVG_MAX) != 0):
                        self.xt += xp[self.index]
                        self.yt += yp[self.index]
                        self.zt += zp[self.index]
                    else:
                        self.xt = (self.xt + xp[self.index])/AVG_MAX
                        self.yt = (self.yt + yp[self.index])/AVG_MAX
                        self.zt = (self.zt + zp[self.index])/AVG_MAX
                        self.ax_AR.plot([self.xt], [self.yt], [self.zt], 'm.')
                        self.xt, self.yt, self.zt = 0, 0, 0

                    self.index += 1
                    print(100*self.index/len(xp))
                else:
                    pass

                return

        animation = FuncAnimation(self.fig, update, frames=len(xp), interval=1, repeat=False, save_count=len(xp))
        animation.save('results/'+FILE+'.mp4', fps=25.)
        print("Animation saved")
        self.index = 0
        self.markerEdge = self.markerEdge/5

    def plot_MC(self, file, showAngles):
        with open(file, 'rt', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter=',')
            MC_data = list(reader)

        MC_data = np.array(MC_data[7:], dtype=float)
        #MC_time = MC_data[START::STEP,1]
        MC_xop = np.average(MC_data[START::STEP,5])
        MC_yop = np.average(MC_data[START::STEP,6])
        MC_zop = np.average(MC_data[START::STEP,7])
        MC_xor = np.average(MC_data[START::STEP,2])
        MC_yor = np.average(MC_data[START::STEP,3])
        MC_zor = np.average(MC_data[START::STEP,4])
        bx, by, bz = self.plotCoordSys(np.array([[0,0,0]]), np.array([[0.,0.,0.]]), False, 10)
        self.ax_MC.add_artist(bx)
        self.ax_MC.add_artist(by)
        self.ax_MC.add_artist(bz)
        
        R_MC = self.RotateXYZ(-MC_xor, -MC_yor, -MC_zor)
        MC_o = R_MC.dot(np.array([[MC_xop],[MC_yop],[MC_zop]]))
        R_MCh = self.RotateHom(MC_o[0,0], MC_o[1,0], MC_o[2,0], -MC_xor, -MC_yor, -MC_zor)

        MC_xp, MC_yp, MC_zp = MC_data[START::STEP,12], MC_data[START::STEP,13], MC_data[START::STEP,14]
        MC_pos = np.transpose(R_MCh.dot(np.column_stack((MC_xp, MC_yp, MC_zp, np.ones((len(MC_xp),1)))).T))
        MC_xp, MC_yp, MC_zp = MC_pos[:,0], MC_pos[:,1], MC_pos[:,2]

        if not SET_EQUAL:
            self.ax_MC.set_xlim([min((0,min(MC_xp))),max((0,max(MC_xp)))])
            self.ax_MC.set_ylim([min((0,min(MC_yp))),max((0,max(MC_yp)))])
            self.ax_MC.set_zlim([min((0,min(MC_zp))),max((0,max(MC_zp)))])
        else:
            minval, maxval = min((0,min(MC_xp),min(MC_yp),min(MC_zp))), max((0,max(MC_xp),max(MC_yp),max(MC_zp)))
            self.ax_MC.set_xlim([minval,maxval])
            self.ax_MC.set_ylim([minval,maxval])
            self.ax_MC.set_zlim([minval,maxval])
        # self.ax_MC.set_xlim([0.5, 1])
        # self.ax_MC.set_ylim([-0.5, -1])
        # self.ax_MC.set_zlim([0.2, 0.7])

        MC_xr = MC_data[START::STEP, 9]
        MC_yr = MC_data[START::STEP,10]
        MC_zr = MC_data[START::STEP,11]
        self.ax_MC.plot(MC_xp[SHOW_START+START:],MC_yp[SHOW_START+START:],MC_zp[SHOW_START+START:],'k--')

        self.ax_MC.set_xlabel("X [m]")
        self.ax_MC.set_ylabel("Y [m]")
        self.ax_MC.set_zlabel("Z [m]")

        if showAngles:
            for i in range(len(MC_xp[SHOW_START+START:])):
                i += SHOW_START+START
                if ((i % (AVG_MAX*10)) == 0):
                    rvec_act = np.array([[MC_xr[i]+90,MC_yr[i],MC_zr[i]]])
                    bx, by, bz = self.plotCoordSys(np.array([[MC_xp[i],MC_yp[i],MC_zp[i]]]), rvec_act, True, 1)
                    self.ax_MC.add_artist(bx)
                    self.ax_MC.add_artist(by)
                    self.ax_MC.add_artist(bz)

    def plot_AR(self, AR_file, MC_file, showAngles, showComponents):
        if not ANIM:
            self.plot_MC(MC_file, showAngles)
        
        with np.load(AR_file) as X:
            tvec = X['tvecs']
            rvec = X['rvecs']
            t_origin = X['t_origin']
            r_origin = X['t_origin']
            orientation = X['orientation']
            t = X['t']
        
        m = self.plotMarkers(t_origin, r_origin, orientation, 10)

        xp = orientation[0][0]*tvec[:,orientation[1][0]]
        yp = orientation[0][1]*tvec[:,orientation[1][1]]
        zp = orientation[0][2]*tvec[:,orientation[1][2]]

        if not SET_EQUAL:
            self.ax_AR.set_xlim([min((m[0],min(xp))),max((m[1],max(xp)))])
            self.ax_AR.set_ylim([min((m[2],min(yp))),max((m[3],max(yp)))])
            self.ax_AR.set_zlim([min((m[4],min(zp))),max((m[5],max(zp)))])
        else:
            minval, maxval = min((m[0],m[2],m[4],min(xp),min(yp),min(zp))), max((m[1],m[3],m[5],max(xp),max(yp),max(zp)))
            self.ax_AR.set_xlim([minval,maxval])
            self.ax_AR.set_ylim([minval,maxval])
            self.ax_AR.set_zlim([minval,maxval])
        # self.ax_AR.set_xlim([0.5, 1])
        # self.ax_AR.set_ylim([-0.5, -1])
        # self.ax_AR.set_zlim([0, 0.5])

        measurements = np.column_stack((xp,yp,zp))

        avg = np.zeros((1,3))
        minv, maxv = 10000, 0
        for i in range(AVG_MAX):
            avg += measurements[i]
            if np.linalg.norm(measurements[i]) > np.linalg.norm(maxv):
                maxv = measurements[i]
            if np.linalg.norm(measurements[i]) < np.linalg.norm(minv):
                minv = measurements[i]
        
        avg=(avg-maxv-minv)/(AVG_MAX-2)

        initial_state_mean = [avg[0, 0], 0, avg[0, 1], 0, avg[0, 2], 0]
        dt = 0.04
        transition_matrix = [[1, dt, 0, 0, 0, 0],
                             [0, 1, 0, 0, 0, 0],
                             [0, 0, 1, dt, 0, 0],
                             [0, 0, 0, 1, 0, 0],
                             [0, 0, 0, 0, 1, dt],
                             [0, 0, 0, 0, 0, 1]]

        observation_matrix = [[1, 0, 0, 0, 0, 0],
                              [0, 0, 1, 0, 0, 0],
                              [0, 0, 0, 0, 1, 0]]

        kf1 = KalmanFilter(transition_matrices = transition_matrix,
                        observation_matrices = observation_matrix,
                        initial_state_mean = initial_state_mean)

        kf1 = kf1.em(measurements, n_iter=5)
        (smoothed_state_means, smoothed_state_covariances) = kf1.smooth(measurements)

        print("KF1 done")

        kf2 = KalmanFilter(transition_matrices = transition_matrix,
                        observation_matrices = observation_matrix,
                        initial_state_mean = initial_state_mean,
                        observation_covariance = SMOOTHER*kf1.observation_covariance,
                  em_vars=['transition_covariance', 'initial_state_covariance'])

        kf2 = kf2.em(measurements, n_iter=5)
        (smoothed_state_means, smoothed_state_covariances) = kf2.smooth(measurements)

        print("KF2 done")

        if showComponents:
            times = t
            plt.figure(2)
            plt.xlabel("t [s]")
            plt.ylabel("X [m]")
            plt.plot(times, measurements[:, 0], 'r-', times, smoothed_state_means[:, 0], 'b--')
            plt.figure(3)
            plt.xlabel("t [s]")
            plt.ylabel("Y [m]")
            plt.plot(times, measurements[:, 1], 'g-', times, smoothed_state_means[:, 2], 'r--')
            plt.figure(4)
            plt.xlabel("t [s]")
            plt.ylabel("Z [m]")
            plt.plot(times, measurements[:, 2], 'b-', times, smoothed_state_means[:, 4], 'r--')

        xp = smoothed_state_means[:, 0]
        yp = smoothed_state_means[:, 2]
        zp = smoothed_state_means[:, 4]

        self.ax_AR.set_xlabel("X [m]")
        self.ax_AR.set_ylabel("Y [m]")
        self.ax_AR.set_zlabel("Z [m]")

        if ANIM:
            self.animate(xp, yp, zp, rvec)
        else:
            if showAngles:
                for i in range(len(xp[SHOW_START:])):
                    i += SHOW_START
                    if ((i % (AVG_MAX*2)) == 0):
                        rvec_act = np.array([[float(rvec[i][0]),float(rvec[i][1]),float(rvec[i][2])]])
                        bx, by, bz = self.plotCoordSys(np.array([[xp[i],yp[i],zp[i]]]), rvec_act, True, 1)
                        self.ax_AR.add_artist(bx)
                        self.ax_AR.add_artist(by)
                        self.ax_AR.add_artist(bz)



            self.ax_AR.plot(xp[SHOW_START:], yp[SHOW_START:], zp[SHOW_START:], 'k--')
            plt.tight_layout()
            plt.show()

    def plotMarkers(self, t_origin, r_origin, orientation, MUT):
        xo = orientation[0][0]*t_origin[:,0,orientation[1][0]]
        yo = orientation[0][1]*t_origin[:,0,orientation[1][1]]
        zo = orientation[0][2]*t_origin[:,0,orientation[1][2]]
        rxo = orientation[0][0]*r_origin[:,0,orientation[1][0]]
        ryo = orientation[0][1]*r_origin[:,0,orientation[1][1]]
        rzo = orientation[0][2]*r_origin[:,0,orientation[1][2]]

        # add the marker origins
        for i in range(t_origin.shape[0]):
            if i == 0:
                # print(t_origin[i])
                bx, by, bz = self.plotCoordSys(np.array([[xo[i],yo[i],zo[i]]]), np.array([[0.,0.,0.]]), False, MUT)
            else:
                bx, by, bz = self.plotCoordSys(np.array([[xo[i],yo[i],zo[i]]]), np.array([[rxo[i],ryo[i],rzo[i]]]), False, MUT)
            self.ax_AR.add_artist(bx)
            self.ax_AR.add_artist(by)
            self.ax_AR.add_artist(bz)

        return [min(xo), max(xo), min(yo), max(yo), min(zo), max(zo)]

    def RotateXYZ(self, pitch, roll, yaw):
        pitch, roll, yaw = [pitch*math.pi/180, roll*math.pi/180, yaw*math.pi/180]
        RotX=np.array([[1, 0, 0],[0, math.cos(pitch), -math.sin(pitch)],[0, math.sin(pitch), math.cos(pitch)]])
        RotY=np.array([[math.cos(roll), 0, math.sin(roll)],[0, 1, 0],[-math.sin(roll), 0, math.cos(roll)]])
        RotZ=np.array([[math.cos(yaw), -math.sin(yaw), 0],[math.sin(yaw), math.cos(yaw), 0],[0, 0, 1]])
        Rot = RotX.dot(RotY.dot(RotZ))

        return Rot

    def RotateHom(self, xo, yo, zo, xr, yr, zr):
        R = self.RotateXYZ(xr, yr, zr)
        vo = np.array([[xo],[yo],[zo]])
        R_hom = np.column_stack((R,vo))
        R_hom = np.row_stack((R_hom, np.array([[0,0,0,1]])))
        
        return R_hom

    def plotCoordSys(self, origin, rot, euler, MUT):
        bases = np.array([[1, 0, 0],[0, 1, 0], [0, 0, 1]])
        bases = bases * self.markerEdge
        if not euler:
            R = cv2.Rodrigues(rot)[0]
        else:
            R = self.RotateXYZ(rot[0][0], rot[0][1], rot[0][2])
        
        bases = R.dot(bases)

        ox = origin[0][0]
        oy = origin[0][1]
        oz = origin[0][2]
        
        coord_arrow_X = Arrow3D((ox,ox+bases[0][0]),(oy,oy+bases[1][0]),(oz,oz+bases[2][0]), mutation_scale=MUT, lw=1, arrowstyle="-|>", color="r")
        coord_arrow_Y = Arrow3D((ox,ox+bases[0][1]),(oy,oy+bases[1][1]),(oz,oz+bases[2][1]), mutation_scale=MUT, lw=1, arrowstyle="-|>", color="g")
        coord_arrow_Z = Arrow3D((ox,ox+bases[0][2]),(oy,oy+bases[1][2]),(oz,oz+bases[2][2]), mutation_scale=MUT, lw=1, arrowstyle="-|>", color="b")

        return coord_arrow_X, coord_arrow_Y, coord_arrow_Z


plotter = Plotting(.11)
plotter.plot_AR(ARUCO_PATH, MOCAP_PATH, SHOW_ANG, SHOW_COMP)