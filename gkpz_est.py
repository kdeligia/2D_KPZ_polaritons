#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 15 17:36:10 2021

@author: delis
"""

import itertools
import matplotlib.pyplot as pl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from matplotlib.texmanager import TexManager
from matplotlib import rcParams as rcP

fig_width_pt = 246.0
inches_per_pt = 1.0 / 72.27
golden_mean = (np.sqrt(5) - 1.0) / 2.0
fig_width = fig_width_pt * inches_per_pt
fig_height = fig_width * golden_mean

fig_size = [fig_width, fig_height]
params = {
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'axes.formatter.limits' : [-4, 4],
        'legend.columnspacing' : 1,
        'legend.fontsize' : 9,
        'legend.frameon': False,
        'axes.labelsize': 12,
        'font.size': 12,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'lines.linewidth': 1,
        'lines.markersize': 3,
        'ytick.major.pad' : 4,
        'xtick.major.pad' : 4,
        'text.usetex': True,
        'font.family' : 'sans-serif',
        'font.weight' : 'light',
        'figure.figsize': fig_size,
        'savefig.format': 'pdf',
        'savefig.bbox': 'tight'
}
rcP.update(params)

c = 3e2 #μm/ps
hbar = 6.582119569 * 1e2 # μeV ps
melectron = 0.510998950 * 1e12 / c**2 # μeV/(μm^2/ps^2)

hatt = 32 # ps
hatx = 4 * np.sqrt(2)
hatpsi = 1 / hatx # μm^-1
hatrho = 1 / hatx ** 2 # μm^-2
hatepsilon = hbar / hatt # μeV

m_tilde = 8e-5
m_dim = m_tilde * melectron
Kc = hbar ** 2 / (2 * m_dim * hatepsilon * hatx ** 2)

p = 2
gamma0_tilde = 0.3125 * hatt
gammar_tilde = 0.1 * gamma0_tilde
gamma2_tilde = 0.1 * hatt / hatx ** 2
Kd = gamma2_tilde / 2

g = 0
gr = 0
ns_tilde = 3.75 / hatrho

def compute(Kc, Kd, p, g, gr, ns, gamma0, gammar, gamma2):
    n0 = ns * (p - 1)
    P = p * gamma0 * ns
    nrth = P / gammar
    sigma = gamma0 * (p + 1) / 4
    if g == 0:
        a = 0
    else:
        a = (2 * p * g * ns / (hbar * gamma0_tilde)) * (1 - 2 * gr * nrth / (p ** 2 * g * ns))
    lambdakpz = - 2 * (Kc - a * Kd)
    nu = Kd + a * Kc
    D = sigma / (2 * n0) * (1 + a ** 2)
    gkpz = np.abs(lambdakpz) ** 2 * D / nu ** 3
    print(lambdakpz, nu, D)
    return gkpz

gkpz = compute(Kc, Kd, p, g, gr, ns_tilde, gamma0_tilde, gammar_tilde, gamma2_tilde)
print(gkpz)
