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

'''
g1 = np.loadtxt('/Users/delis/Desktop/2^7/final_g1.dat', dtype=np.complex_)
dx = np.loadtxt('/Users/delis/Desktop/2^7/dx_N2_N_2**7.dat')
dt = np.loadtxt('/Users/delis/Desktop/2^7/dt_50k_250_300k.dat')

fig,ax = pl.subplots(2, 1, figsize=(6,8))
ax[0].set_xscale('log')
ax[0].set_yscale('log')
ax[1].set_xscale('log')
ax[1].set_yscale('log')
ax[0].plot(dt[1:], g1[1:,0], label=r'$-2 \ln g_1(\Delta t, \Delta x=0)$')
ax[0].plot(dt[1:], 0.01*dt[1:]**(2*0.23))
ax[0].plot(dt[1:], 0.001*dt[1:]**(1))
ax[1].plot(dx[1:], g1[0, 1:], label=r'$-2 \ln g_1(\Delta t=0, \Delta x)$')
ax[1].plot(dx[1:], 0.01*dx[1:]**0.8)
ax[0].tick_params(which='both', axis='both', labelsize=16, length=4)
ax[1].tick_params(which='both', axis='both', labelsize=16, length=4)
ax[0].legend(prop=dict(size=16))
ax[1].legend(prop=dict(size=16))
'''

t = np.loadtxt('/Users/delis/Desktop/figures/t.dat')

fig1 = np.loadtxt('/Users/delis/Desktop/Figures/n1.dat')
fig1_det = np.loadtxt('/Users/delis/Desktop/Figures/n1_0.dat')
fig,ax = pl.subplots(1, 1, figsize=(6,6))
ax.plot(t, fig1, label=r'$n(0,0)$')
ax.plot(t, fig1_det, label=r'$n(0,0)_{SS}$')
ax.tick_params(which='both', axis='both', labelsize=16, length=4)
ax.tick_params(which='both', axis='both', labelsize=16, length=4)
ax.hlines(y=249.99999999999997, xmin=t[0], xmax=t[-1])
ax.legend(prop=dict(size=16))
ax.legend(prop=dict(size=16))
pl.title(r'$p=1.2, m=-4 \times 10^{-4} m_e, g_r=0.02 \mu eV \mu m^2, g=0.96 \mu eV \mu m^2$')
pl.show()

fig1 = np.loadtxt('/Users/delis/Desktop/Figures/n2.dat')
fig,ax = pl.subplots(1, 1, figsize=(6,6))
ax.plot(t, fig1, label=r'$n(0,0)$')
ax.tick_params(which='both', axis='both', labelsize=16, length=4)
ax.tick_params(which='both', axis='both', labelsize=16, length=4)
ax.hlines(y=249.99999999999997, xmin=t[0], xmax=t[-1])
ax.legend(prop=dict(size=16))
ax.legend(prop=dict(size=16))
pl.title(r'$p=1.15, m=4 \times 10^{-4} m_e, g_r=0.06 \mu eV \mu m^2, g=4.6 \mu eV \mu m^2$')
pl.show()

fig1 = np.loadtxt('/Users/delis/Desktop/Figures/n3.dat')
fig,ax = pl.subplots(1, 1, figsize=(6,6))
ax.plot(t, fig1, label=r'$n(0,0)$')
ax.tick_params(which='both', axis='both', labelsize=16, length=4)
ax.tick_params(which='both', axis='both', labelsize=16, length=4)
ax.hlines(y=249.99999999999997, xmin=t[0], xmax=t[-1])
ax.legend(prop=dict(size=16))
ax.legend(prop=dict(size=16))
pl.title(r'$p=1.15, m=4 \times 10^{-5} m_e, g_r=0.06 \mu eV \mu m^2, g=4.6 \mu eV \mu m^2$')
pl.show()