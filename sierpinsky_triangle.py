#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 19 09:59:45 2021

@author: ericchen
"""
import numpy as np
import random
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(20, 20), dpi = 200)
ax = plt.axes()

def draw_triangle(ax):
    ax.plot([-1,1],[0,0], color = 'green')
    ax.plot([-1,0],[0,1*np.sin(np.pi/3)], color = 'green')
    ax.plot([0,1], [1*np.sin(np.pi/3),0], color = 'green')
    
draw_triangle(ax)

def point_on_triangle(ax):
    """
    Random point on the triangle with vertices pt1, pt2 and pt3.
    """
    pt1 = [-1,0]
    pt2 = [1,0]
    pt3 = [0, 1*np.sin(np.pi/3)]
    x, y = sorted([random.random(), random.random()])
    s, t, u = x, y - x, 1 - y
    point = [s * pt1[0] + t * pt2[0] + u * pt3[0],
            s * pt1[1] + t * pt2[1] + u * pt3[1]]
    plt.plot(point[0], point[1], marker = 'o', markersize = 25)
    return point

point_on_triangle(ax)

def chaos_game(ax, iterations):
    fig = plt.figure(figsize=(20, 20), dpi = 200)
    ax = plt.axes()

    draw_triangle(ax)
    point = point_on_triangle(ax)
    
    for i in range(iterations):
        points = [[-1,0],[1,0], [0, 1*np.sin(np.pi/3)]]
        vertex = random.choice(points)
        ax.plot((point[0]+vertex[0])/2, (point[1]+vertex[1])/2, marker = 'o', markersize = 2, color = 'red')
        point = [(point[0]+vertex[0])/2, (point[1]+vertex[1])/2]
        
chaos_game(ax, 50000)
    
    
    
    
    
    