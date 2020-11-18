#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 14:16:40 2020

@author: delis
"""

import numpy as np
import matplotlib.pyplot as pl

deltax = np.loadtxt('/Users/delis/Desktop/dx.dat')
g1 = np.loadtxt('/Users/delis/desktop/g_2.dat')

fig,ax = pl.subplots(1,1, figsize=(8,6))
ax.set_xscale('log')
ax.set_yscale('log')
ax.plot(deltax, g1)