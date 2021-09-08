#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  7 10:00:31 2021

@author: ericchen
"""
import numpy as np
import matplotlib.pyplot as plt 


def radioactivity(N0,r):
    t=np.linspace(0,5/r,1000)
    dt=t[1]-t[0]
    N=np.zeros(len(t));N[0]=N0
    #Euler's method
    for i in range(1,len(N)):
        N[i]=N[i-1]*(1-r*dt)
       
    plt.plot(t,N)
    plt.axhline(color='black');plt.axvline(color='black');plt.grid()
    plt.xlabel("time");plt.ylabel("Number of undecayed atoms")
    plt.title("Radioactive Decay")
   

def two_types(Na0, Nb0, Ra, Rb):
    t=np.linspace(0,5/min(Ra,Rb),100)
    dt=t[1]-t[0]
    Na=np.zeros(len(t));Na[0]=Na0
    #Euler's method
    for i in range(1,len(Na)):
        Na[i]=Na[i-1]*(1-Ra*dt)
       
    Nb=np.zeros(len(t));Nb[0]=Nb0
    for i in range(1,len(Nb)):
        Nb[i]=Nb[i-1]+dt*(Ra*Na[i-1]-Rb*Nb[i-1])
        
    #Nb_a = np.zeros(len(t));
    Nb_a = (Ra/(Rb-Ra))*Na0*(np.exp(-Ra*t)-np.exp(-Rb*t))
    
        
    plt.plot(t,Na, label = "A")
    plt.plot(t,Nb, label = "B")
    plt.plot(t,Nb_a, 'b.',label="Analytical", color='g')
    plt.legend()
    plt.axhline(color='black');plt.axvline(color='black');plt.grid()
    plt.xlabel("time");plt.ylabel("Number of undecayed atoms")
    plt.title("Radioactive Decay")
    
    
two_types(1000,0,4,3)