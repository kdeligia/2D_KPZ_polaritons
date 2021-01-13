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
g1_2_7 = np.load('/Users/delis/Desktop/g1/g1_x_27.npy')
g1_2_8 = np.load('/Users/delis/Desktop/g1/g1_x_28.npy')
D1_2_7 = np.load('/Users/delis/Desktop/g1/D1_x_27.npy')
D1_2_8 = np.load('/Users/delis/Desktop/g1/D1_x_28.npy')

fig, ax = pl.subplots(1,1, figsize=(8,8))
ax.set_xscale('log')
ax.set_yscale('log')
ax.plot(np.abs(g1_2_7)/D1_2_7, label=r'$N=2^7, g=6.4522, g_r=0.1455$')
ax.plot(np.abs(g1_2_8)/D1_2_8, label=r'$N=2^8, g=6.4522, g_r=0,1455$')
ax.tick_params(axis='both', which='both', direction='in', labelsize=20, pad=12, length=12)
pl.tight_layout()
pl.legend()
pl.show()

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