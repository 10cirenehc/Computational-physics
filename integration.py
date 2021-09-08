#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 30 10:18:28 2021

@author: ericchen
"""
def trapezoidal(f, a, b, n):
    h = (b-a)/n
    f_sum = 0
    for i in range(1, n, 1):
        x = a + i*h
        f_sum = f_sum + f(x)
    return h*(0.5*f(a) + f_sum + 0.5*f(b))

from math import exp

v = lambda t: 3*(t**2)*exp(t**3)
n = 4
numerical = trapezoidal(v, 0, 1, n)

#%%
#exact

V = lambda t: exp(t**3)
exact = V(1) - V(0)
print(abs(exact - numerical))

#%%

numerical = trapezoidal(v, 0, 1, n=400)

#%%
#vectorizing

from numpy import linspace, sum
def midpoint(f, a, b, n):
    h = (b-a)/n
    x = linspace(a + h/2, b - h/2, n)
    return h*sum(f(x))


def trapezoidal(f, a, b, n):
    h = (b-a)/n
    x = linspace(a, b, n+1)
    s = sum(f(x)) - 0.5*f(a) - 0.5*f(b)
    return h*s


