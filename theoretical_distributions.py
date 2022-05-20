#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 27 00:03:48 2021

@author: delis
"""

from functions import stats
import numpy as np
import matplotlib.pyplot as pl
from matplotlib import rc
from matplotlib.texmanager import TexManager
pl.rc('font', family='sans-serif')
pl.rc('text', usetex=True)
path_phase = r'/Users/delis/Documents/Data 2D/Phase'

convention = 2 ** (-2/3)
gue = np.genfromtxt('/Users/delis/Python Workspace/Paper Figures/Figure 2 data/GUEps.txt')
goe = np.genfromtxt('/Users/delis/Python Workspace/Paper Figures/Figure 2 data/GOEps.txt')
F0 = np.genfromtxt('/Users/delis/Python Workspace/Paper Figures/Figure 2 data/FO.txt')

Pq_gue = np.exp(gue[:, 2])
q_gue = gue[:, 0]

Pq_goe = np.exp(goe[:, 2]) / convention
q_goe = goe[:, 0] * convention

Pq_F0 = np.exp(F0[:, 2])
q_F0 = F0[:, 0]

fig,ax = pl.subplots()
ax.set_yscale('log')
ax.plot(q_F0, Pq_F0, c='red', label=r'BR', linewidth=2)
ax.plot(q_gue, Pq_gue, c='blue', label=r'TW-GUE', linewidth=2)
ax.plot(q_goe, Pq_goe, c='green', label=r'TW-GOE', linewidth=2)
ax.set_xlabel(r'$X$', fontsize=16)
ax.set_ylabel(r'$P(X)$', fontsize=16)
ax.tick_params(axis='both', which='both',labelsize = 16, length=4, direction='in')
ax.set_xlim(-6, 7)
ax.set_ylim(1e-5, 1)
ax.legend(loc='upper right', frameon=False, prop=dict(size=14), markerscale=2)
pl.savefig('/Users/delis/Desktop/fig4.eps', format='eps')

fig.show()

print('--- Statistical params TW-GUE ---')
stats(q_gue, Pq_gue)
print('--- Statistical params TW-GOE ---')
stats(q_goe, Pq_goe)
print('--- Statistical params F0 ---')
stats(q_F0, Pq_F0)