#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 14 17:18:17 2021

@author: ericchen
"""
import matplotlib.pyplot as plt
import numpy as np 
fig = plt.figure()
plt.rcParams['figure.figsize'] = (10,10)

l = 4.5
def iterate(X, Y, Z, l):
    mu = 2.2
    alpha = 5; gamma = 1; beta = 8; delta=1
    Xdot = alpha*Y*Z - gamma*X
    Ydot = mu*(Y+Z) - beta*(X*Z)
    Zdot = delta*Y - l*Z
    return Xdot,Ydot,Zdot

dt = 0.01
num_steps = 100000

xs = np.empty(num_steps + 1)
ys = np.empty(num_steps + 1)
zs = np.empty(num_steps + 1)

xs[0], ys[0], zs[0] = (0.1, 0.1, 0.1)
for i in range(num_steps):
    x_dot, y_dot, z_dot = iterate(xs[i], ys[i], zs[i], l)
    xs[i + 1] = xs[i] + (x_dot * dt)
    ys[i + 1] = ys[i] + (y_dot * dt)
    zs[i + 1] = zs[i] + (z_dot * dt)

ax = plt.figure().add_subplot(projection='3d')

ax.plot(xs, ys, zs, lw=0.5)
ax.set_xlabel("Employees (x(t))")
ax.set_ylabel("Working Capital (y(t))")
ax.set_zlabel("Loan Amount (z(t)")
ax.set_title("Model of Firm (gamma="+str(l)+")")

plt.show()

#%%
fig = plt.figure()
plt.rcParams['figure.figsize'] = (10,10)

l = 4.0
mu = 2.3
def iterate(X, Y, Z, l, mu):

    alpha = 5; gamma = 1; beta = 8; delta=1
    Xdot = alpha*Y*Z - gamma*X
    Ydot = mu*(Y+Z) - beta*(X*Z)
    Zdot = delta*Y - l*Z
    return Xdot,Ydot,Zdot

dt = 0.01
num_steps = 100000

xs = np.empty(num_steps + 1)
ys = np.empty(num_steps + 1)
zs = np.empty(num_steps + 1)

xs[0], ys[0], zs[0] = (0.1, 0.1, 0.1)
for i in range(num_steps):
    x_dot, y_dot, z_dot = iterate(xs[i], ys[i], zs[i], l, mu)
    xs[i + 1] = xs[i] + (x_dot * dt)
    ys[i + 1] = ys[i] + (y_dot * dt)
    zs[i + 1] = zs[i] + (z_dot * dt)

ax = plt.figure().add_subplot(projection='3d')

ax.plot(xs, ys, zs, lw=0.5)
ax.set_xlabel("Employees (x(t))")
ax.set_ylabel("Working Capital (y(t))")
ax.set_zlabel("Loan Amount (z(t)")
ax.set_title("Model of Firm (gamma="+str(l)+" mu="+str(mu)+")")

plt.show()


#%%
import numpy as np
import matplotlib.pyplot as plt


def lorenz(x, y, z, s=10, r=28, b=2.667):
    """
    Given:
       x, y, z: a point of interest in three dimensional space
       s, r, b: parameters defining the lorenz attractor
    Returns:
       x_dot, y_dot, z_dot: values of the lorenz attractor's partial
           derivatives at the point x, y, z
    """
    x_dot = s*(y - x)
    y_dot = r*x - y - x*z
    z_dot = x*y - b*z
    return x_dot, y_dot, z_dot


dt = 0.01
num_steps = 10000

# Need one more for the initial values
xs = np.empty(num_steps + 1)
ys = np.empty(num_steps + 1)
zs = np.empty(num_steps + 1)

# Set initial values
xs[0], ys[0], zs[0] = (0., 1., 1.05)

# Step through "time", calculating the partial derivatives at the current point
# and using them to estimate the next point
for i in range(num_steps):
    x_dot, y_dot, z_dot = lorenz(xs[i], ys[i], zs[i])
    xs[i + 1] = xs[i] + (x_dot * dt)
    ys[i + 1] = ys[i] + (y_dot * dt)
    zs[i + 1] = zs[i] + (z_dot * dt)


# Plot
ax = plt.figure().add_subplot(projection='3d')

ax.plot(xs, ys, zs, lw=0.5)
ax.set_xlabel("X Axis")
ax.set_ylabel("Y Axis")
ax.set_zlabel("Z Axis")
ax.set_title("Lorenz Attractor")

plt.show()

#%%

from scipy import integrate
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import cnames
from matplotlib import animation
import seaborn as sns
from IPython.display import HTML
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib.colors import ListedColormap, BoundaryNorm


def iterate(current, t0):
    X = current[0]
    Y = current[1]
    Z = current[2]
    l = 4.0
    mu = 2.2
    alpha = 5; gamma = 1; beta = 8; delta=1
    Xdot = alpha*Y*Z - gamma*X
    Ydot = mu*(Y+Z) - beta*(X*Z)
    Zdot = delta*Y - l*Z
    return [Xdot,Ydot,Zdot]


fig = plt.figure()
plt.rcParams['figure.figsize'] = (10,10)

ax = fig.add_axes([0, 0, 1, 1], projection='3d')
ax.axis('off')

# prepare the axes limits
ax.set_xlim((0,3))
ax.set_ylim((-3, 3))
ax.set_zlim((-2, 2))

# set point-of-view: specified by (altitude degrees, azimuth degrees)
ax.view_init(50, 0)

N_trajectories = 10
size_queue = 500
T = 8
N = 1000
x0 = np.array([[0.1, 0.1, 0]]*N_trajectories)
x0[:,2] = np.linspace(0, 0.5, N_trajectories)

# Solve for the trajectories
t = np.linspace(0, T, N)
x_t = np.asarray([integrate.odeint(iterate, x0i, t)
                  for x0i in x0])

print('Trajectories simulated.')


pts = [ax.plot([], [], [], 'o', c=c, alpha = 0.8)[0] for c in ['black']*N_trajectories]

# Create a continuous norm to map from data points to colors
norm = plt.Normalize(0, 1)
lines = [Line3DCollection([], cmap='cool', norm=norm, lw = 2, alpha = 0.6) for n in range(N_trajectories)]
for line in lines:
    line.set_array(np.linspace(1,0.,size_queue))
    ax.add_collection(line)
    
def animate(i):
    for line, pt, xi in zip(lines, pts, x_t):
        x, y, z = xi[:i].T
        #pad with nan to properly handle colorization
        if len(x)<size_queue:
            xt = np.concatenate([(size_queue-len(x))*[np.nan],x])
            yt = np.concatenate([(size_queue-len(x))*[np.nan],y])
            zt = np.concatenate([(size_queue-len(x))*[np.nan],z])
        else:
            xt = x
            yt = y
            zt = z
        points = np.array([xt[-size_queue:], yt[-size_queue:], zt[-size_queue:]]).T.reshape(-1, 1, 3)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        line.set_segments(segments)
        pt.set_data(x[-1:], y[-1:])
        pt.set_3d_properties(z[-1:])
    ax.view_init(50, 0.3 * i)
    fig.canvas.draw()
    return lines + pts

anim = animation.FuncAnimation(fig, animate, frames=N, interval=30, blit=True)
anim.save('/Users/ericchen/comp_phys/lorenz1.gif', writer='imagemagick', fps=30)

#%%

import numpy as np
from scipy import integrate
import matplotlib
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import cnames
from matplotlib import animation
import seaborn as sns

def lorentz_deriv( l_coor , t0, sigma=10., beta=8./3, rho=28.0):
    """Compute the time-derivative of a Lorentz system."""
    [x, y, z] = l_coor
    return [sigma * (y - x), x * (rho - z) - y, x * y - beta * z]

#parameters
N_trajectories = 20
size_queue = 500
T = 80
N = 1000

#initial condition
x0 = np.array([[0.1, 0.1, 0]]*N_trajectories)
x0[:,2] = np.linspace(0, 30, N_trajectories)

# Solve for the trajectories
t = np.linspace(0, T, N)
x_t = np.asarray([integrate.odeint(lorentz_deriv, x0i, t) for x0i in x0])

# Define the 2D array of colors
array_colors = [sns.hls_palette(N, l=.6, s=x) for x in np.linspace(0.4,0.9, N_trajectories)]

# Set up figure & 3D axis for animation
fig = plt.figure()
ax = fig.add_axes([0, 0, 1, 1], projection='3d')
ax.axis('off')

# set up lines and points
lines = [ax.plot([], [], [], '-', c=c[0], alpha = 0.7)[0] for c in array_colors]
pts = [ax.plot([], [], [], 'o', c=c[0])[0] for c in array_colors]

# prepare the axes limits
ax.set_xlim((-25, 25))
ax.set_ylim((-35, 35))
ax.set_zlim((-10, 40))

# set point-of-view: specified by (altitude degrees, azimuth degrees)
ax.view_init(30, 0)

# animation function.  This will be called sequentially with the frame number
def animate(i):
    t = int(i/N*T)
    for idx, (line, pt, xi) in enumerate(zip(lines, pts, x_t)):
        x, y, z = xi[:i].T

        line.set_data(x[-size_queue:], y[-size_queue:])
        line.set_3d_properties(z[-size_queue:])
        line.set_color(array_colors[idx][i])

        pt.set_data(x[-1:], y[-1:])
        pt.set_3d_properties(z[-1:])
        pt.set_color(array_colors[idx][i])

    ax.view_init(30, 0.3 * i)
    fig.canvas.draw()
    return lines + pts

# instantiate the animator.
anim = animation.FuncAnimation(fig, animate, frames=N, interval=30, blit=True)
anim.save('/Users/ericchen/comp_phys/lorenz2.gif', writer='imagemagick', fps=30)

#Save as mp4. This requires mplayer or ffmpeg to be installed
#anim.save('lorenz_rainbow.mp4', writer = 'ffmpeg', fps=3, extra_args=['-vcodec','libx264'], dpi = 250)

#Or directly show the result (doesn't work with Jupyter)
#plt.show()