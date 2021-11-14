#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 12 10:07:28 2021

@author: ericchen
"""
#%%

import matplotlib.pyplot as plt
import numpy as np 
from celluloid import Camera


plt.rcParams['figure.figsize'] = (10,10)
camera = Camera(plt.figure())

def iterate(num, camera):
    N = 50
    n =1000
    
    x_ref = np.linspace(0,1,n)
    
    x_long = np.linspace(0,1,n)
    y_long = np.linspace(0, 1,n)
    
    y = np.zeros(n)
    
    r = np.linspace(1,4,num)
    for j in range (0,num):
        x_old = 0.2
        x_new = 0

        
        for i in range (1,n):
            y[i] = r[j]*x_ref[i]*(1-(x_ref[i]))
            
        plt.plot(x_ref,y)
        plt.plot(x_long, y_long)
        plt.vlines(x_old, 0, x_old,linestyles='solid', alpha = 0.4, color = 'green')
        
        for i in range(1,N):
            x_new = r[j]*x_old*(1-(x_old))
            plt.plot(x_old, x_new, marker="o", markersize=5, markeredgecolor="red", markerfacecolor="red", alpha = (i/N))
            plt.hlines(x_new, x_old, x_new,linestyles='solid', alpha = 0.3, color = 'green')
            plt.vlines(x_old, x_old, x_new,linestyles='solid', alpha = 0.3, color = 'green')
            x_old = x_new
           
        plt.show()
    return camera

cam = iterate(30,camera)
anim = cam.animate(blit=False)
anim.save('anim.gif')

#%%
def logistic(r, x):
    """Implementation of the logistic map
    
    Args:
        x (float): previous value from 0 to 1
        r (float): the R paremter
    """
    return r * x * (1 - x)

fig = plt.figure(figsize=(20, 10))
ax = plt.axes()

r = np.linspace(2.5, 4.0, 10000)
x = 0.2*np.ones(len(r))


iterations = 1000
for i in range(iterations):
    x = logistic(r,x)
    if(i>950):
        ax.plot(r, x, ',b', alpha=.5)
plt.title("Bifurcation Diagram")
plt.show()

#%%

import matplotlib.animation as animation

fig = plt.figure(figsize=(20, 10))
ax = plt.axes()
n = 100000
r = np.linspace(2.5, 4.0, n)



def animate(i):
    # clear axes object
    plt.cla()
    ax.set_title(f"Bifurcation Diagram for $x^{{(r)}}_{{n + 1}} = rx^{{(r)}}_{{n}}(1 - x^{{(r)}}_{{n}})$ for $n=${i + 1}", fontsize=24)
    ax.set_xlabel('r', fontsize=24)
    ax.set_ylabel('$x^{(r)}_{n}$', fontsize=24)
    ax.tick_params(axis='both', which='major', labelsize=18)
    ax.set_xlim(2.5, 4.0)
    ax.set_ylim(0.0, 1.0)
    
    x = np.random.uniform(0.0, 1.0, n)  # initial value
    for _ in range(i + 1):  # animated iterations
        x = logistic(r, x)
    
    l = ax.plot(r, x, ',b', alpha=.5)
    return l

# call the animator	 
anim = animation.FuncAnimation(fig, animate, frames=iterations, interval=225, blit=True)
# save the animation as a gif file 
anim.save('biffurcation_rand_init_value.gif',writer='imagemagick')

#%%
from matplotlib.pyplot import cm

plt.rcParams['figure.figsize'] = (10,10)
plt.xlim((0.25, 0.75))   # set the xlim to left, right
plt.ylim(0.8,1.0)
def plot_one(r, c):
    x = np.random.uniform(0.0, 1.0, 1000)  # initial value
    for i in range (0,500):
        x = logistic(r,x)
    x_1 = logistic(r,x)
    plt.plot(x,x_1, marker = 'o',markersize=1, linestyle = "None", c=c)
    
plot_one(3.9, 'blue')
plt.xlabel("x(t)", fontsize = 20)
plt.ylabel("x(t+1)", fontsize = 20)
plt.title("Logistic Map, r = 3.9", fontsize = 20)



#%%
plt.rcParams['figure.figsize'] = (10,10)
plt.xlim((0.25, 0.75))   # set the xlim to left, right
plt.ylim(0.8,1.0)
n = 50
rates = np.linspace(3.6,4.0,n)
color = iter(cm.rainbow(np.linspace(0, 0.4, n)))
np.random.seed(15)

for r in rates:
    c = next(color)
    plot_one(r,c)
    
plt.xlabel("x(t)", fontsize = 20)
plt.ylabel("x(t+1)", fontsize = 20)
plt.title("Logistic Map, 50 curves from r=3.6 to r=4.0", fontsize = 20)

#%%

plt.rcParams['figure.figsize'] = (10,10)
plt.xlim((0.0, 1.0))   # set the xlim to left, right
plt.ylim(0.0,1.0)
plot_one(3.99,"blue")
plt.xlabel("x(t)", fontsize = 20)
plt.ylabel("x(t+1)", fontsize = 20)
plt.title("Logistic Map r=3.99", fontsize = 20)

#%%
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
plt.rcParams['figure.figsize'] = (10,10)

def plot_3d(r, c,ax):
    #ax = fig.add_subplot(111, projection='3d')
    x = np.random.uniform(0.0, 1.0, 1000)  # initial value
    for i in range (0,500):
        x = logistic(r,x)
    x_1 = logistic(r,x)
    x_2 = logistic(r,x_1)
    ax.scatter(x,x_1, zs=x_2, marker = 'o', c=c, s=1)
    ax.set_xlabel("x(t)", fontsize=12)
    ax.set_ylabel("x(t+1)", fontsize=12)
    ax.set_zlabel("x(t+2)", fontsize=12)
    #ax.set_title("Logistic attractor r=3.99", fontsize = 20)
plot_3d(3.99, 'blue')

#%%

from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
plt.rcParams['figure.figsize'] = (10,10)

n = 50
rates = np.linspace(3.6,4.0,n)
color = iter(cm.rainbow(np.linspace(0, 0.4, n)))
np.random.seed(15)
ax = fig.add_subplot(111, projection='3d')

for r in rates:
    c = next(color)
    plot_3d(r,c,ax)

ax.set_title("Logistic Attractor from r=3.6 to r=3.9", fontsize=20)

for angle in range(0, 360):
    ax.view_init(30, angle)
    plt.draw()
    plt.pause(.001)
    
    
ax.view_init(30, 90)
plt.draw()
p = plt.show()

#%%


        



