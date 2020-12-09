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

g0 = np.loadtxt('/Users/delis/Desktop/2D/g_0.dat')
th = np.loadtxt('/Users/delis/Desktop/2D/64x64.txt')
dx = np.loadtxt('/Users/delis/Desktop/2D/dx.dat')

fig,ax = pl.subplots(1,1, figsize=(8,6))
ax.set_xscale('log')
ax.set_yscale('log')
ax.plot(dx[1:], -2*np.log(g0)[1:], 'o', label='Mine')
ax.plot(th[int(len(th)/2)+1:, 0], -2*np.log(th[int(len(th)/2)+1:, 1]), '^', label=r'Kai Ji')
ax.plot(dx[1:], 0.07*dx[1:]**0.8)
ax.legend()
ax.tick_params(axis='both', which='both', labelsize=16)
pl.subplots_adjust(left=0.2, right=0.95)