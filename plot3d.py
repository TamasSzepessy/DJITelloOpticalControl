import numpy as np
import matplotlib
# matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
from scipy import interpolate
from scipy.spatial.transform import Rotation
import math
import cv2
import time
from pykalman import KalmanFilter
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

AVG_MAX = 12
SMOOTHER = 5000

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
        fig = plt.figure(1)
        self.ax = fig.add_subplot(111, projection='3d')

    def plotout(self, file, showAngles, showComponents):
        with np.load(file) as X:
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

        self.ax.set_xlim([min((m[0],min(xp))),max((m[1],max(xp)))])
        self.ax.set_ylim([min((m[2],min(yp))),max((m[3],max(yp)))])
        self.ax.set_zlim([min((m[4],min(zp))),max((m[5],max(zp)))])

        measurements = tvec

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

        print("kf1 done")

        kf2 = KalmanFilter(transition_matrices = transition_matrix,
                        observation_matrices = observation_matrix,
                        initial_state_mean = initial_state_mean,
                        observation_covariance = SMOOTHER*kf1.observation_covariance,
                  em_vars=['transition_covariance', 'initial_state_covariance'])

        kf2 = kf2.em(measurements, n_iter=5)
        (smoothed_state_means, smoothed_state_covariances) = kf2.smooth(measurements)

        print("kf2 done")

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

        if showAngles:
            for i in range(len(xp)):
                if ((i % (AVG_MAX*2)) == 0):
                    rvec_act = np.array([[float(rvec[i][0]),float(rvec[i][1]),float(rvec[i][2])]])
                    bx, by, bz = self.plotCoordSys(np.array([[xp[i],yp[i],zp[i]]]), rvec_act, True, 1)
                    self.ax.add_artist(bx)
                    self.ax.add_artist(by)
                    self.ax.add_artist(bz)

        self.ax.set_xlabel("X [m]")
        self.ax.set_ylabel("Y [m]")
        self.ax.set_zlabel("Z [m]")

        self.ax.plot(xp, yp, zp, 'k--')
        plt.show()

    def plotRT(self, file):
        with np.load(file) as X:
            t = X['t']
            tvec = X['tvecs']
            rvec = X['rvecs']
            t_origin = X['t_origin']
            r_origin = X['t_origin']
            orientation = X['orientation']

        plt.ion()
        fig = plt.figure()
        self.ax = fig.add_subplot(111, projection='3d')

        xo = orientation[0][0]*t_origin[:,0,orientation[1][0]]
        yo = orientation[0][1]*t_origin[:,0,orientation[1][1]]
        zo = orientation[0][2]*t_origin[:,0,orientation[1][2]]
        rxo = orientation[0][0]*r_origin[:,0,orientation[1][0]]
        ryo = orientation[0][1]*r_origin[:,0,orientation[1][1]]
        rzo = orientation[0][2]*r_origin[:,0,orientation[1][2]]

        maxval=max((max(xo),max(yo),max(zo)))
        minval=min((min(xo),min(yo),min(zo)))

        # add the marker origins
        for i in range(t_origin.shape[0]):
            if i == 0:
                # print(t_origin[i])
                bx, by, bz = self.plotCoordSys(np.array([[xo[i],yo[i],zo[i]]]), np.array([[0.,0.,0.]]), False)
            else:
                bx, by, bz = self.plotCoordSys(np.array([[xo[i],yo[i],zo[i]]]), np.array([[rxo[i],ryo[i],rzo[i]]]), False)
            self.ax.add_artist(bx)
            self.ax.add_artist(by)
            self.ax.add_artist(bz)

        xp = orientation[0][0]*tvec[:,orientation[1][0]]
        yp = orientation[0][1]*tvec[:,orientation[1][1]]
        zp = orientation[0][2]*tvec[:,orientation[1][2]]
        
        okay = np.where(np.abs(np.diff(xp)) + np.abs(np.diff(yp)) + np.abs(np.diff(zp)) > 0)
        xp = np.r_[xp[okay], xp[-1]]
        yp = np.r_[yp[okay], yp[-1]]
        zp = np.r_[zp[okay], zp[-1]]

        maxval=max((max(xp),max(yp),max(zp), maxval))
        minval=min((min(xp),min(yp),min(zp), minval))

        self.ax.set_xlim([min((min(xo),min(xp))),max((max(xo),max(xp)))])
        self.ax.set_ylim([min((min(yo),min(yp))),max((max(yo),max(yp)))])
        self.ax.set_zlim([min((min(zo),min(zp))),max((max(zo),max(zp)))])

        self.markerEdge = self.markerEdge*5
        xp_new = np.array([[0.]])
        yp_new = np.array([[0.]])
        zp_new = np.array([[0.]])

        bx_prev, by_prev, bz_prev = self.plotCoordSys(np.array([[0, 0, 0]]), np.array([[0.,0.,0.]]), False)
        i = 0
        while(True):
            try:
                bx_prev.remove()
                by_prev.remove()
                bz_prev.remove()
            except:
                pass

            if i < len(xp):
                rvec_act = np.array([[float(rvec[i][0]),float(rvec[i][1]),float(rvec[i][2])]])
                # print(rvec_act)
                # r = Rotation.from_euler('xyz', rvec_act)
                # rvec_act = r.as_rotvec()
                # Moving a coordinate system on the camera positions (origin, rotation) given
                bx, by, bz = self.plotCoordSys(np.array([[xp[i],yp[i],zp[i]]]), rvec_act, True)
                self.ax.add_artist(bx)
                self.ax.add_artist(by)
                self.ax.add_artist(bz)

                bx_prev = bx
                by_prev = by
                bz_prev = bz

                if i == 0:
                    xt, yt, zt = 0, 0, 0            
                if (((i+1) % AVG_MAX) != 0):
                    xt += xp[i]
                    yt += yp[i]
                    zt += zp[i]
                else:
                    xt = (xt + xp[i])/AVG_MAX
                    yt = (yt + yp[i])/AVG_MAX
                    zt = (zt + zp[i])/AVG_MAX
                    xp_new=np.append(xp_new, np.array([[xt]]), axis=1)
                    yp_new=np.append(yp_new, np.array([[yt]]), axis=1)
                    zp_new=np.append(zp_new, np.array([[zt]]), axis=1)

                    self.ax.plot([xt], [yt], [zt], 'm.')

                    xt, yt, zt = 0, 0, 0

                i += 1
                plt.pause(t[i]-t[i-1])
            else:
                try:
                    tck, _ = interpolate.splprep([xp_new, yp_new, zp_new], s=1)
                    # x_knots, y_knots, z_knots = interpolate.splev(tck[0], tck)
                    u_fine = np.linspace(0,1,tvec.shape[0])
                    x_fine, y_fine, z_fine = interpolate.splev(u_fine, tck)
                    # ax.plot(x_knots, y_knots, z_knots, 'go')
                    self.ax.plot(x_fine, y_fine, z_fine, 'g')
                except:
                    print("no spline")
                
                plt.pause(20)
                break  

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
            self.ax.add_artist(bx)
            self.ax.add_artist(by)
            self.ax.add_artist(bz)

        return [min(xo), max(xo), min(yo), max(yo), min(zo), max(zo)]

    def RotateXYZ(self, pitch, roll, yaw):
        pitch, roll, yaw = [pitch*math.pi/180, roll*math.pi/180, yaw*math.pi/180]
        RotX=np.array([[1, 0, 0],[0, math.cos(pitch), -math.sin(pitch)],[0, math.sin(pitch), math.cos(pitch)]])
        RotY=np.array([[math.cos(roll), 0, math.sin(roll)],[0, 1, 0],[-math.sin(roll), 0, math.cos(roll)]])
        RotZ=np.array([[math.cos(yaw), -math.sin(yaw), 0],[math.sin(yaw), math.cos(yaw), 0],[0, 0, 1]])
        Rot = RotX.dot(RotY.dot(RotZ))

        return Rot

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
plotter.plotout('results/Mocap/movement_20191111_115459.npz', True, False)
#plotter.plotRT('results/Mocap/movement_20191111_112626.npz')