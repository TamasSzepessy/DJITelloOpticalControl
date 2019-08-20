import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        FancyArrowPatch.draw(self, renderer)

class Navigator():

    t = np.linspace(0,2*np.pi, 40)

    # Position Equation
    def rx(self, t):
        return t * np.cos(t)
    def ry(self, t):
        return t * np.sin(t)

    # Velocity Vectors
    def vx(self, t):
        return np.cos(t) - t*np.sin(t)
    def vy(self, t):
        return np.sin(t) + t*np.cos(t)

    # Acceleration Vectors
    def ax(self, t):
        return -2*np.sin(t) - t*np.cos(t)

    def ay(self, t):
        return 2*np.cos(t) - t*np.sin(t)

    # fig = plt.figure()
    # ax1 = fig.gca(projection='3d')

    def velos(self):
        t_step = 3
        vel_val_x=[]
        vel_val_y=[]
        vel_val_z=[]
        for t_pos in range(0, len(self.t)-1, t_step):
            t_val_start = self.t[t_pos]
            t_val_end = self.t[t_pos+1]

            # print(t_val_start)
            # print(t_val_end)

            vel_start = [self.rx(t_val_start), self.ry(t_val_start), t_val_start]
            vel_end = [self.rx(t_val_start)+self.vx(t_val_start), self.ry(t_val_start)+self.vy(t_val_start), t_val_end]
            # vel_vecs = list(zip(vel_start, vel_end))
            vel_val_x.append(vel_end[0]-vel_start[0])
            vel_val_y.append(vel_end[1]-vel_start[1])
            vel_val_z.append((vel_end[2]-vel_start[2]))
            # vel_arrow = Arrow3D(vel_vecs[0],vel_vecs[1],vel_vecs[2], mutation_scale=20, lw=1, arrowstyle="-|>", color="g")
            # ax1.add_artist(vel_arrow)

            # acc_start = [rx(t_val_start), ry(t_val_start), t_val_start]
            # acc_end = [rx(t_val_start)+ax(t_val_start), ry(t_val_start)+ay(t_val_start), t_val_start]
            # acc_vecs = list(zip(acc_start, acc_end))
            # acc_arrow = Arrow3D(acc_vecs[0],acc_vecs[1],acc_vecs[2], mutation_scale=20, lw=1, arrowstyle="-|>", color="m")
            # ax1.add_artist(acc_arrow)

        return vel_val_x, vel_val_y, vel_val_z

    # print(vel_val_x)
    # print(vel_val_y)
    # print(vel_val_z)

    # z = t
    # ax1.plot(rx(z), ry(z), z)
    # plt.xlim(-2*np.pi,2*np.pi)
    # plt.ylim(-6,6)
    # plt.show()

'''import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

#matplotlib inline
import random
# set seed to reproducible
random.seed(1)
data_size = 51
max_value_range = 132651
x = np.array([random.random()*max_value_range for p in range(0,data_size)])
y = np.array([random.random()*max_value_range for p in range(0,data_size)])
z = 2*x*x*x + np.sqrt(y)*y + random.random()
fig = plt.figure(figsize=(10,6))
ax = axes3d.Axes3D(fig)
ax.scatter3D(x,y,z, c='r')'''