import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import interpolate
import math
import cv2
import transformations as tf

class Plotting():

    def RotateX(self,angle, x, y, z):
        rad = angle*math.pi/180
        rotx=np.array([[1, 0, 0],[0, math.cos(rad), -math.sin(rad)],[0, math.sin(rad), math.cos(rad)]])
        temp = np.matmul(np.column_stack((x,y,z)),rotx)
        xr, yr, zr = temp[:,0], temp[:,1], temp[:,2]

        return xr, yr, zr

    def plotout(self, file):
        with np.load(file) as X:
            tvec = X['tvecs']
            rvec = X['rvecs']
            orientation = X['orientation']

        # print(rvec*180/math.pi)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        xp=orientation[0][0]*tvec[:,orientation[1][0]]
        yp=orientation[0][1]*tvec[:,orientation[1][1]]
        zp=orientation[0][2]*tvec[:,orientation[1][2]]
        # xp2=np.copy(xp)
        # for i in range(len(rvec)):
        #     rvec[i]=tf.rotationVectorToEulerAngles(rvec[i])
        #     xp2[i]=xp[i]*math.cos(rvec[i,1])-yp[i]*math.sin(rvec[i,1])
        #     yp[i]=yp[i]*math.cos(rvec[i,1])+xp[i]*math.sin(rvec[i,1])
        #     xp[i]=xp2[i]
        okay = np.where(np.abs(np.diff(xp)) + np.abs(np.diff(yp)) + np.abs(np.diff(zp)) > 0)
        xp = np.r_[xp[okay], xp[-1]]
        yp = np.r_[yp[okay], yp[-1]]
        zp = np.r_[zp[okay], zp[-1]]

        okay=np.array([0])
        for i in range(len(xp)):
            if xp[i]==0 and yp[i] == 0 and zp[i] == 0:
                continue
            else:
                okay = np.append(okay,[i])

        xp = np.r_[xp[okay]]
        yp = np.r_[yp[okay]]
        zp = np.r_[zp[okay]]

        xp, yp, zp = self.RotateX(10.5, xp, yp, zp)

        maxval=max((max(xp),max(yp),max(zp)))
        minval=min((min(xp),min(yp),min(zp)))

        # tck, _ = interpolate.splprep([xp, yp, zp], s=1)
        # # x_knots, y_knots, z_knots = interpolate.splev(tck[0], tck)
        # u_fine = np.linspace(0,1,tvec.shape[0])
        # x_fine, y_fine, z_fine = interpolate.splev(u_fine, tck)

        # ax.set_xlim([minval,maxval])
        # ax.set_ylim([minval,maxval])
        # ax.set_zlim([minval,maxval])
        ax.set_xlabel("X [m]")
        ax.set_ylabel("Y [m]")
        ax.set_zlabel("Z [m]")

        ax.plot(xp, yp, zp, 'r*')
        # ax.plot(x_knots, y_knots, z_knots, 'go')
        #ax.plot(x_fine, y_fine, z_fine, 'g')
        ax.plot([minval,maxval],[0,0],[0,0],'k',linewidth=0.5)
        ax.plot([0,0],[minval,maxval],[0,0],'k',linewidth=0.5)
        ax.plot([0,0],[0,0],[minval,maxval],'k',linewidth=0.5)

        plt.show()
        plt.pause(0.01)

# plotter = Plotting()
# plotter.plotout('results/movement_20191006_111620.npz')