#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 31 10:28:50 2021

@author: ericchen
"""
import numpy as np 
import matplotlib.pyplot as plt

#%%
#Exercise one 

def e1(M,c,rho,K,T0,Tw, Ty):
    t = (((M**(2/3))*c*(rho**(1/3))))/(K*(np.pi**2)*((4*np.pi)/3)**(2/3))*np.log(0.76*((T0-Tw)/(Ty-Tw)))
    print (t)
    
e1(67,3.7,1.038,5.4*10**(-3),20,100,70)

#%% 
#Exercise 2


def sin(x):
    from math import factorial
    terms = np.zeros([1,8])
    true = np.sin(x)
    a = 1
    for i in range (0,8):
        term = (x**a)/(factorial(a))
        if i%2 ==1:
            term = term*(-1)
        
        terms[0,i] = term
        error = ((true-np.sum(terms))/true)*100
        print("current error: " + str(error))
        a = a+2
        
    print("answer:" + str(np.sum(terms)))
        
sin(0.9)
        

#%%
#exercise 3

def charge (plot):
    L = 9
    R = 60
    q0 = 10
    C = 0.00005
    t = np.linspace(0,0.8,1000000)
    
    charge = np.zeros([50,])
    charge =  q0*np.exp((-R*t)/(2*L))*np.cos(((1/(L*C))-(R/(2*L))**2)**(1/2)*t)
    
    if (plot==True):
        plt.plot(t,charge)
        
charge(True)

#%%
#Exercise 4

def water_density():
    
    T = np.linspace(0,100,10000)
    density = (5.5289*10**-8)*T**3-8.5016*10**(-6)*T**2+6.5622*10**(-5)+0.99987
    
    plt.plot(T,density)
    plt.title("Water density as function of temperature")
    plt.xlabel("Temperature (C)")
    plt.ylabel("Density(g/cm^3)")
    
water_density()

#%%
#Exeercise 5

def cos():
    from math import factorial
    x = np.linspace(0,3*np.pi/2,10000)
    taylor = 1-(x**2/factorial(2))+(x**4/factorial(4))-(x**6/factorial(6))+(x**8/factorial(8))
    true = np.cos(x)
    
    plt.plot(x,taylor, 'g', label="fit")
    plt.plot(x,true, 'r')
    plt.legend()
    
cos()
#%%

#Exercise 6

def gaussian (sigma):
    x = np.linspace(-6,6,10000)
    g = (1/(sigma*(2*np.pi)))*np.exp(-(x**2/(2*sigma**2)))
    plt.plot(x,g, label=str(sigma))

gaussian(1)
gaussian(1.5)
gaussian(2)
plt.legend()

#%%
#Exercise 7

def LJ_potential(r):
    epsilon = 1.65*10**(-21)
    sigma = 3.4*10**(-10)
    U = 4*epsilon*((sigma/r)**12-(sigma/r)**6)
    
    plt.plot(r,U)
    plt.title("Lennard Jones potential")
    plt.xlim(2.7*10**-10,6*10**-10)
    plt.ylim(-3*10**-21,10*10**-21)
    
r = np.linspace(1*10**-10,10*10**-10,10000)
LJ_potential(r)

#%%
#Derivative

def LJ_force(r):
    epsilon = 1.65*10**(-21)
    sigma = 3.4*10**(-10)
    U = -(4 * epsilon * ((-12*sigma**12/r**13)-(-6*sigma**6/r**7)))
    
    plt.plot(r,U)
    plt.xlim(2.7*10**-10,6*10**-10)
    plt.ylim(-3*10**-11,10*10**-11)

r = np.linspace(1*10**-10,10*10**-10,10000)
LJ_force(r)

#%%
#Small displacements
def LJ_oscillate(r):
    ep = 1.65*10**(-21)
    sigma = 3.4*10**(-10)
    V0 = -ep
    r0 = 2**(1/6)*sigma
    k = 36*ep/(2**(2/3)*sigma**2)
    V = (0.5)*k*(r-r0)**2+V0
    
    plt.plot(r,V)
    plt.xlim(2.7*10**-5,1)
    
r = np.linspace(1*10**-5,1,10000)
LJ_oscillate(r)

#%%
#8: Sunflower

def sunflower():
    s = np.linspace(1,600,600)
    r = s**(1/2)
    phi = (1+5**(1/2))/2
    theta = (2*np.pi*s)/phi
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    ax.plot(theta, r, linestyle = "None", marker = "o")
    

sunflower()

#%%
#9: fitting graphs to data

def fit():
    L = np.linspace(0.1,1.0,10)
    T = np.array([0.6,0.9,1.1,1.3,1.4,1.6,1.7,1.8,1.9,2.0])
    plt.plot(L,T, linestyle = "None", marker="o")
    
    for i in range (1,4):
        deg = i
        coeff = np.polyfit(L,T,deg)
        p = np.poly1d(coeff)
        fitted = p(L)
        plt.plot(L, fitted, label = "Deg = " + str(deg))
    
    plt.legend()
    
fit()

#%%
#10: wave packet

def wave(x,t):
    y = np.exp(-(x-3*t)**2)*np.sin(3*np.pi*(x-t))
    
    plt.plot(x,y, color = "g")
    
x = np.linspace(-4,4,10000)
t = 0
wave(x,t)
        
        

