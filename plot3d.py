import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import interpolate

with np.load('results/movement_20190921_162400.npz') as X:
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

maxval=max((max(xp),max(yp),max(zp)))
minval=min((min(xp),min(yp),min(zp)))

tck, u = interpolate.splprep([xp, yp, zp], s=0.01)
x_knots, y_knots, z_knots = interpolate.splev(tck[0], tck)
u_fine = np.linspace(0,1,tvec.shape[0])
x_fine, y_fine, z_fine = interpolate.splev(u_fine, tck)

ax.set_xlim([minval,maxval])
ax.set_ylim([minval,maxval])
ax.set_zlim([minval,maxval])

ax.plot(xp, yp, zp, 'r*')
ax.plot(x_knots, y_knots, z_knots, 'go')
ax.plot(x_fine, y_fine, z_fine, 'g')


plt.show()