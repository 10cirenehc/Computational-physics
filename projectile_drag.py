#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 13 10:03:00 2021

@author: ericchen
"""
import numpy as np
import matplotlib.pyplot as plt

def quadratic_drag(v0, theta, b, m, plot):
   # v0 =3
   # theta = 30
    #b = 0.05
   # m = 10
    
    theta = np.deg2rad(theta)
    
    dt = 0.01
    g = 9.81
    
    r0 = np.array([0.0,0.0])
    v0 = np.array([v0*np.cos(theta),v0*np.sin(theta)])
    
    v_t = np.copy(v0); v = v0
    r_t = np.copy(r0); r = r0
    t0 = np.array([0.0])
    t_values = np.copy(t0); t = t0
    
    while r[1]>=0:
        a = -b*np.linalg.norm(v)*v/m+np.array([0,-g])
        t += dt
        v += a*dt
        r += v*dt
        r_t = np.vstack((r_t, r))
        v_t = np.vstack((v_t,v))
        t = np.vstack((t_values, t))
    
    if(plot == 1):
        plt.plot(t, ((v_t[:,0])**2+(v_t[:,1])**2)**(1/2)); 
        plt.xlabel("Time(s)"); plt.ylabel("Velocity(m/s)")
        plt.title("Velocity")
        plt.show()
        
        plt.plot(t, (1/2)*m*(((v_t[:,0])**2+(v_t[:,1])**2)**(1/2))**2); 
        plt.xlabel("Time(s)"); plt.ylabel("Kinertic Energy(joules)")
        plt.title("Kinetic Energy")
        plt.show()
        
        plt.plot(r_t[:,0], r_t[:,1]);
        plt.xlabel("x(m)"); plt.ylabel("y(m)")
        plt.title("Position")
        plt.show()
        
    if(plot==2):
        plt.plot(r_t[:,0], r_t[:,1]);
        plt.xlabel("x(m)"); plt.ylabel("y(m)")
        plt.title("Position")
        
    return r_t
    
quadratic_drag(100, 90, 0.3, 10.0, True)

#%%
def envelope(v0):
    
    angles = np.linspace(45,90, 1000)
    v0 = 100
    b = 0
    m = 10
    maxima = np.array([0,0])
    angle = 50
    for angle in angles:
        r_t = quadratic_drag(v0,angle,b,m,0)
        r = (r_t[:,0]**2+r_t[:,1]**2)**(1/2)
        index = np.argmax(r)
        maxima = np.vstack((maxima,r_t[index]))
    
    plt.plot(maxima[:,0], maxima[:,1])
    
envelope(100)

#%%
def quadratic_drag_wind(v0, theta, b, m, w, plot):
   # v0 =3
   # theta = 30
    #b = 0.05
   # m = 10
    
    theta = np.deg2rad(theta)
    
    dt = 0.01
    g = 9.81
    
    r0 = np.array([0.0,0.0])
    v0 = np.array([v0*np.cos(theta),v0*np.sin(theta)])
    
    v_t = np.copy(v0); v = v0
    r_t = np.copy(r0); r = r0
    t0 = np.array([0.0])
    t_values = np.copy(t0); t = t0
    
    while r[1]>=0:
        a = -(b/m)*np.linalg.norm(v-w)*(v-w)/m+np.array([0,-g])
        t += dt
        v += a*dt
        r += v*dt
        r_t = np.vstack((r_t, r))
        v_t = np.vstack((v_t,v))
        t = np.vstack((t_values, t))
    
    if(plot == 1):
        plt.plot(t, ((v_t[:,0])**2+(v_t[:,1])**2)**(1/2)); 
        plt.xlabel("Time(s)"); plt.ylabel("Velocity(m/s)")
        plt.title("Velocity")
        plt.show()
        
        plt.plot(t, (1/2)*m*(((v_t[:,0])**2+(v_t[:,1])**2)**(1/2))**2); 
        plt.xlabel("Time(s)"); plt.ylabel("Kinertic Energy(joules)")
        plt.title("Kinetic Energy")
        plt.show()
        
        plt.plot(r_t[:,0], r_t[:,1]);
        plt.xlabel("x(m)"); plt.ylabel("y(m)")
        plt.title("Position")
        plt.show()
        
    if(plot==2):
        plt.plot(r_t[:,0], r_t[:,1]);
        plt.xlabel("x(m)"); plt.ylabel("y(m)")
        plt.title("Position")
        
        
    return r_t
        
wind = np.array([-4,100])
quadratic_drag_wind(100, 45, 0.3, 10.0, wind, True)
    
        
    
        