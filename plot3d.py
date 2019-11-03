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
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

AVG_MAX = 12

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
        # self.fig = Figure()
        # self.canvas = FigureCanvas(self.fig)
        
        #plt.show()
        #self.fig.show()

        # self.bx_prev, self.by_prev, self.bz_prev = self.plotCoordSys(np.array([[0, 0, 0]]), np.array([[0.,0.,0.]]), True)

    # def RotateXYZ(self, pitch, roll, yaw, x, y, z):
    #     rotx=np.array([[1, 0, 0],[0, math.cos(pitch), -math.sin(pitch)],[0, math.sin(pitch), math.cos(pitch)]])
    #     roty=np.array([[math.cos(roll), 0, math.sin(roll)],[0, 1, 0],[-math.sin(roll), 0, math.cos(roll)]])
    #     rotz=np.array([[math.cos(pitch), -math.sin(pitch), 0],[math.sin(pitch), math.cos(pitch), 0],[0, 0, 1]])
    #     rot=np.matmul(roty,rotz)
    #     rot=np.matmul(rotx,rot)
    #     temp = np.matmul(np.column_stack((x,y,z)),rot)
    #     xr, yr, zr = temp[:,0], temp[:,1], temp[:,2]

    #     return xr, yr, zr

    def Average(self, x, y, z, i):
        pass

    def plotout(self, file, use_avg):
        with np.load(file) as X:
            tvec = X['tvecs']
            rvec = X['rvecs']
            t_origin = X['t_origin']
            r_origin = X['t_origin']
            orientation = X['orientation']

        fig = plt.figure()
        self.ax = fig.add_subplot(111, projection='3d')

        # add the marker origins
        for i in range(t_origin.shape[0]):
            xp = orientation[0][0]*t_origin[i][0][orientation[1][0]]
            yp = orientation[0][1]*t_origin[i][0][orientation[1][1]]
            zp = orientation[0][2]*t_origin[i][0][orientation[1][2]]
            rxp = orientation[0][0]*r_origin[i][0][orientation[1][0]]
            ryp = orientation[0][1]*r_origin[i][0][orientation[1][1]]
            rzp = orientation[0][2]*r_origin[i][0][orientation[1][2]]
            if i == 0:
                # print(t_origin[i])
                bx, by, bz = self.plotCoordSys(np.array([[xp,yp,zp]]), np.array([[0.,0.,0.]]), False)
            else:
                bx, by, bz = self.plotCoordSys(np.array([[xp,yp,zp]]), np.array([[rxp,ryp,rzp]]), False)
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

        maxval=max((max(xp),max(yp),max(zp)))
        minval=min((min(xp),min(yp),min(zp)))

        self.ax.set_xlim([minval,maxval])
        self.ax.set_ylim([minval,maxval])
        self.ax.set_zlim([minval,maxval])

        if use_avg:
            xp_new = np.array([[0.]])
            yp_new = np.array([[0.]])
            zp_new = np.array([[0.]])
            for i in range(len(xp)):
                if i == 0:
                    xt = 0
                    yt = 0
                    zt = 0
                
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

                    xt = 0
                    yt = 0
                    zt = 0
            
            xp = xp_new[0][1:-1]
            yp = yp_new[0][1:-1]
            zp = zp_new[0][1:-1]

        try:
            tck, _ = interpolate.splprep([xp, yp, zp], s=0.001)
            # x_knots, y_knots, z_knots = interpolate.splev(tck[0], tck)
            u_fine = np.linspace(0,1,tvec.shape[0])
            x_fine, y_fine, z_fine = interpolate.splev(u_fine, tck)
            # ax.plot(x_knots, y_knots, z_knots, 'go')
            self.ax.plot(x_fine, y_fine, z_fine, 'g')
        except:
            print("no spline")

        self.ax.set_xlim([minval,maxval])
        self.ax.set_ylim([minval,maxval])
        self.ax.set_zlim([minval,maxval])
        self.ax.set_xlabel("X [m]")
        self.ax.set_ylabel("Y [m]")
        self.ax.set_zlabel("Z [m]")

        self.ax.plot(xp, yp, zp, 'r*')
        plt.show()

        #bx_prev, by_prev, bz_prev = self.plotCoordSys(np.array([[0, 0, 0]]), np.array([[0.,0.,0.]]))
        '''i = 0
        while(True):
            # if not self.pos_queue.empty():
            #     q=self.pos_queue.get()

            try:
                bx_prev.remove()
                by_prev.remove()
                bz_prev.remove()
            except:
                pass

            # maxval=max((max(xp),max(yp),max(zp)))
            # minval=min((min(xp),min(yp),min(zp)))

            # try:
            #     tck, _ = interpolate.splprep([xp, yp, zp], s=1)
            #     # x_knots, y_knots, z_knots = interpolate.splev(tck[0], tck)
            #     u_fine = np.linspace(0,1,tvec.shape[0])
            #     x_fine, y_fine, z_fine = interpolate.splev(u_fine, tck)
            #     # ax.plot(x_knots, y_knots, z_knots, 'go')
            #     self.ax.plot(x_fine, y_fine, z_fine, 'g')
            # except:
            #     pass

            # self.ax.set_xlim([minval,maxval])
            # self.ax.set_ylim([minval,maxval])
            # self.ax.set_zlim([minval,maxval])
            # self.ax.set_xlabel("X [m]")
            # self.ax.set_ylabel("Y [m]")
            # self.ax.set_zlabel("Z [m]")

            # self.ax.plot(xp, yp, zp, 'r*')
            # # self.ax.plot([minval,maxval],[0,0],[0,0],'k',linewidth=0.5)
            # # self.ax.plot([0,0],[minval,maxval],[0,0],'k',linewidth=0.5)
            # # self.ax.plot([0,0],[0,0],[minval,maxval],'k',linewidth=0.5)
            if i < len(xp):
                # self.ax.set_xlim([xp[i]-0.2, xp[i]+0.2])
                # self.ax.set_ylim([yp[i]-0.2, yp[i]+0.2])
                # self.ax.set_zlim([zp[i]-0.2, zp[i]+0.2])

                bx, by, bz = self.plotCoordSys(np.array([[xp[i],yp[i],zp[i]]]), np.array([[0.,0.,0.]]))
                self.ax.add_artist(bx)
                self.ax.add_artist(by)
                self.ax.add_artist(bz)

                bx_prev = bx
                by_prev = by
                bz_prev = bz

                # self.canvas.draw()
                # image = np.frombuffer(self.canvas.tostring_rgb(), dtype='uint8')
                self.fig.savefig('to.png')

                image = cv2.imread('to.png')

                cv2.imshow('img', image)
                if cv2.waitKey(1) == 27:
                    break
            else:
                break

            i += 1

            plt.pause(0.04)
            
            # self.fig.canvas.flush_events()   # update the plot and take care of window events (like resizing etc.)
            # time.sleep(0.04)               # wait for next loop iteration'''
    
    def plotRT(self, file):
        with np.load(file) as X:
            tvec = X['tvecs']
            rvec = X['rvecs']
            t_origin = X['t_origin']
            r_origin = X['t_origin']
            orientation = X['orientation']

        plt.ion()
        fig = plt.figure()
        self.ax = fig.add_subplot(111, projection='3d')

        # add the marker origins
        for i in range(t_origin.shape[0]):
            xp = orientation[0][0]*t_origin[i][0][orientation[1][0]]
            yp = orientation[0][1]*t_origin[i][0][orientation[1][1]]
            zp = orientation[0][2]*t_origin[i][0][orientation[1][2]]
            rxp = orientation[0][0]*r_origin[i][0][orientation[1][0]]
            ryp = orientation[0][1]*r_origin[i][0][orientation[1][1]]
            rzp = orientation[0][2]*r_origin[i][0][orientation[1][2]]
            if i == 0:
                # print(t_origin[i])
                bx, by, bz = self.plotCoordSys(np.array([[xp,yp,zp]]), np.array([[0.,0.,0.]]), False)
            else:
                bx, by, bz = self.plotCoordSys(np.array([[xp,yp,zp]]), np.array([[rxp,ryp,rzp]]), False)
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

        maxval=max((max(xp),max(yp),max(zp)))
        minval=min((min(xp),min(yp),min(zp)))

        self.ax.set_xlim([minval,maxval])
        self.ax.set_ylim([minval,maxval])
        self.ax.set_zlim([minval,maxval])

        self.markerEdge = self.markerEdge*10

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
            else:
                try:
                    tck, _ = interpolate.splprep([xp, yp, zp], s=0.001)
                    # x_knots, y_knots, z_knots = interpolate.splev(tck[0], tck)
                    u_fine = np.linspace(0,1,tvec.shape[0])
                    x_fine, y_fine, z_fine = interpolate.splev(u_fine, tck)
                    # ax.plot(x_knots, y_knots, z_knots, 'go')
                    self.ax.plot(x_fine, y_fine, z_fine, 'g')
                except:
                    print("no spline")

            i += 1
            plt.pause(0.04)

    def RotateXYZ(self, pitch, roll, yaw):
        pitch, roll, yaw = [pitch*math.pi/180, roll*math.pi/180, yaw*math.pi/180]
        RotX=np.array([[1, 0, 0],[0, math.cos(pitch), -math.sin(pitch)],[0, math.sin(pitch), math.cos(pitch)]])
        RotY=np.array([[math.cos(roll), 0, math.sin(roll)],[0, 1, 0],[-math.sin(roll), 0, math.cos(roll)]])
        RotZ=np.array([[math.cos(yaw), -math.sin(yaw), 0],[math.sin(yaw), math.cos(yaw), 0],[0, 0, 1]])
        Rot = RotX.dot(RotY.dot(RotZ))

        return Rot

    def plotCoordSys(self, origin, rot, euler):
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
        
        coord_arrow_X = Arrow3D((ox,ox+bases[0][0]),(oy,oy+bases[1][0]),(oz,oz+bases[2][0]), mutation_scale=10, lw=1, arrowstyle="-|>", color="r")
        coord_arrow_Y = Arrow3D((ox,ox+bases[0][1]),(oy,oy+bases[1][1]),(oz,oz+bases[2][1]), mutation_scale=10, lw=1, arrowstyle="-|>", color="g")
        coord_arrow_Z = Arrow3D((ox,ox+bases[0][2]),(oy,oy+bases[1][2]),(oz,oz+bases[2][2]), mutation_scale=10, lw=1, arrowstyle="-|>", color="b")

        return coord_arrow_X, coord_arrow_Y, coord_arrow_Z


plotter = Plotting(.11)
#plotter.plotout('results/movement_20191031_191207.npz', True)
plotter.plotRT('results/useful/movement_20191103_115350.npz')