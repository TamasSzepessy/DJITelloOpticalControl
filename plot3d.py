import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import interpolate
import math

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

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        xp=-tvec[:,0]
        yp=-tvec[:,2]
        zp=tvec[:,1]
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

        tck, _ = interpolate.splprep([xp, yp, zp], s=0.01)
        x_knots, y_knots, z_knots = interpolate.splev(tck[0], tck)
        u_fine = np.linspace(0,1,tvec.shape[0])
        x_fine, y_fine, z_fine = interpolate.splev(u_fine, tck)

        # ax.set_xlim([minval,maxval])
        # ax.set_ylim([minval,maxval])
        # ax.set_zlim([minval,maxval])
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

        ax.plot(xp, yp, zp, 'r*')
        ax.plot(x_knots, y_knots, z_knots, 'go')
        ax.plot(x_fine, y_fine, z_fine, 'g')
        ax.plot([minval,maxval],[0,0],[0,0],'k',linewidth=0.5)
        ax.plot([0,0],[minval,maxval],[0,0],'k',linewidth=0.5)
        ax.plot([0,0],[0,0],[minval,maxval],'k',linewidth=0.5)

        plt.show()

plotter = Plotting()
plotter.plotout('results/movement_20190921_161439.npz')