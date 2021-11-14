#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 29 09:56:58 2021

@author: ericchen
"""
import numpy as np
import matplotlib.pyplot as plt
def shm_drag_forced(omega, omega0, r, y0, vy0, f, plot):
    t = np.linspace(0,100/omega, 10000)
    y = np.zeros(len(t))
    v = np.zeros(len(t))
    dt = t[1]-t[0]
    y[0] = y0
    v[0] = vy0
    
    for i in range (len(t)-1):
        a = -omega**2*y[i] - 2*r*v[i] + f*np.cos(omega*t[i])
        v[i+1] = v[i]+a*dt
        y[i+1] += y[i]+v[i+1]*dt
        
    if plot == True:
        plt.plot(t,y)
        
    return max(y)
    
omega0 = 2
r = 1
optimal_omega = (omega0**2 - (r**2)/2)**(1/2)
print("Optimal omega:" + str(optimal_omega))
optimal_omega = (omega0**2 - (r**2)/2)**(1/2)
shm_drag_forced(optimal_omega, omega0, r, 0, 10, 0, True)

#%%
