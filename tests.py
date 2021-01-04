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
dt = np.loadtxt('/Users/delis/Desktop/data/dt_50k_200_200k.dat')
g1_v1 = np.loadtxt('/Users/delis/Desktop/data/g1_v1_50k_200_200k.dat', dtype=np.complex_)
g1_v2 = np.loadtxt('/Users/delis/Desktop/data/g1_v2_50k_200_200k.dat', dtype=np.complex_)
avg_sqrt_n = np.loadtxt('/Users/delis/Desktop/data/avg_sqrt_density_50k_200_200k.dat', dtype=np.complex_)
denom = np.loadtxt('/Users/delis/Desktop/data/denominator_50k_200_200k.dat', dtype=np.complex_)

fig,ax = pl.subplots(1,1, figsize=(10,10))
ax.set_xscale('log')
ax.set_yscale('log')
ax.plot(dt[1:], g1_v1[1:, 0])
ax.plot(dt[1:], 0.0002*dt[1:]**(2*0.28))
ax.tick_params(axis='x', which='both', direction='in', labelsize=30, pad=12, length=12)
ax.tick_params(axis='y', which='both', direction='in', labelsize=30, pad=8, length=12)
pl.show()

fig,ax = pl.subplots(1,1, figsize=(10,10))
ax.set_xscale('log')
ax.set_yscale('log')
ax.plot(dt[1:], g1_v1[1:, 0])
ax.plot(dt[1:], 0.0002*dt[1:]**(2*0.25))
ax.tick_params(axis='x', which='both', direction='in', labelsize=30, pad=12, length=12)
ax.tick_params(axis='y', which='both', direction='in', labelsize=30, pad=8, length=12)
pl.show()

