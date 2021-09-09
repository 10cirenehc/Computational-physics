#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  9 09:53:45 2021

@author: ericchen
"""
import numpy as np 
import matplotlib.pyplot as plt

def friction(v0,a,b):
    t=np.linspace(0,10,1000)
    dt=t[1]-t[0]
    N=np.zeros(len(t));N[0]= v0
    #Euler's method
    for i in range(1,len(N)):
        N[i]=N[i-1]+(a-b*N[i-1])*dt
       
    plt.plot(t,N)
    plt.axhline(color='black');plt.axvline(color='black');plt.grid()
    plt.xlabel("time");plt.ylabel("Velocity(m/s)")
    plt.title("Velocity with Drag")
    
friction(0,10,0.5)

def population_growth_a(N0, a, b):
    t=np.linspace(0,10,1000)
    dt=t[1]-t[0]
    N=np.zeros(len(t));N[0]= N0
    #Euler's method
    for i in range(1,len(N)):
        N[i]=N[i-1]+(a*N[i-1])*dt
       
    plt.plot(t,N)
    plt.axhline(color='black');plt.axvline(color='black');plt.grid()
    plt.xlabel("time");plt.ylabel("Population")
    plt.title("Population growth no limit")
    
    plt.plot(t, N0*np.exp(a*t), 'b.')
    
population_growth_a(3, 2, 3)

def population_growth_b(N0, a, b):

    t=np.linspace(0,10,1000)
    dt=t[1]-t[0]
    N=np.zeros(len(t));N[0]= N0
    #Euler's method
    for i in range(1,len(N)):
        N[i]=N[i-1]+(a*N[i-1]-b*(N[i-1])**2)*dt
       
    plt.plot(t,N)
    plt.axhline(color='black');plt.axvline(color='black');plt.grid()
    plt.xlabel("time");plt.ylabel("Population")
    plt.title("Population growth no limit")
    
    #plt.plot(t, N0*np.exp(a*t), 'b.')
    
population_growth_b(6,10,0.003)