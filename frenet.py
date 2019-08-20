import numpy as np
from scipy.integrate import solve_ivp

from scipy.integrate import cumtrapz
import matplotlib.pylab as plt

# Define the parameters as regular Python function:
def k(s):
    return 1

def t(s):
    return 0

# The equations: dz/dt = model(s, z):
def model(s, z):
    T = z[:3]   # z is a (9, ) shaped array, the concatenation of T, N and B 
    N = z[3:6]
    B = z[6:]

    dTds =            k(s) * N  
    dNds = -k(s) * T          + t(s) * B
    dBds =           -t(s)* N

    return np.hstack([dTds, dNds, dBds])


T0, N0, B0 = [1, 0, 0], [0, 1, 0], [0, 0, 1] 

z0 = np.hstack([T0, N0, B0])

s_span = (0, 6) # start and final "time"
t_eval = np.linspace(*s_span, 100)  # define the number of point wanted in-between,
                                    # It is not necessary as the solver automatically
                                    # define the number of points.
                                    # It is used here to obtain a relatively correct 
                                    # integration of the coordinates, see the graph

# Solve:
sol = solve_ivp(model, s_span, z0, t_eval=t_eval, method='RK45')
print(sol.message)
# >> The solver successfully reached the end of the integration interval.

# Unpack the solution:
T, N, B = np.split(sol.y, 3)  # another way to unpack the z array
s = sol.t

# Bonus: integration of the normal vector in order to get the coordinates
#        to plot the curve  (there is certainly better way to do this)
coords = cumtrapz(T, x=s)

plt.plot(coords[0, :], coords[1, :])
plt.axis('equal'); plt.xlabel('x'); plt.xlabel('y')