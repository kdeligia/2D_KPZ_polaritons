#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 14:16:40 2020

@author: delis
"""

import numpy as np
import matplotlib.pyplot as pl
from matplotlib import rc
from matplotlib.texmanager import TexManager
import matplotlib.ticker as plticker
import os 

pl.rc('font', family='sans-serif')
pl.rc('text', usetex=True)

dr = np.loadtxt('/Users/delis/Desktop/data/dr_2_7.dat')
dt = np.loadtxt('/Users/delis/Desktop/data/dt.dat')
g1_500_0_1908 = np.load('/Users/delis/Desktop/g1_x_500_0_1908.npy')
D1_500_0_1908 = np.load('/Users/delis/Desktop/D1_x_500_0_1908.npy')
g1_1000_1_908 = np.load('/Users/delis/Desktop/g1_x_1000_1_908.npy')
D1_1000_1_908 = np.load('/Users/delis/Desktop/D1_x_1000_1_908.npy')
g1_1000_1_908 = np.load('/Users/delis/Desktop/g1_x_1000_0_1908.npy')
D1_1000_1_908 = np.load('/Users/delis/Desktop/D1_x_1000_0_1908.npy')

pl.loglog(dr, np.abs(g1_500_0_1908)/D1_500_0_1908)


'''
g1_x = np.load('/Users/delis/Desktop/data/g1_x_p_1pt89.npy')
g1_t = np.load('/Users/delis/Desktop/data/g1_t_p_1pt89.npy')
D1_x = np.load('/Users/delis/Desktop/data/D1_x_p_1pt89.npy')
D1_t = np.load('/Users/delis/Desktop/data/D1_t_p_1pt89.npy')

fig, ax = pl.subplots(1,1, figsize=(8,8))
ax.set_xscale('log')
ax.set_yscale('log')
ax.plot(dt, -2*np.log(g1_t/D1_t))
ax.plot(dt, 0.0005*dt**0.5)
pl.show()

fig, ax = pl.subplots(1,1, figsize=(8,8))
ax.set_xscale('log')
ax.set_yscale('log')
ax.plot(dr, -2*np.log(g1_x/D1_x))
ax.plot(dr, 0.0005*dr**0.5)
pl.show()
'''