#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 11:03:50 2019

@author: delis
"""
import os
import numpy as np
import matplotlib.pyplot as pl
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
        'legend.fontsize' : 10,
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
melectron = 0.510998950 * 1e12 / c ** 2 # μeV/(μm^2/ps^2)

hatt = 32 # ps
hatx = 4 * np.sqrt(2)
hatpsi = 1 / hatx # μm^-1
hatrho = 1 / hatx ** 2 # μm^-2
hatepsilon = hbar / hatt # μeV

def confining(array, V_0, l):
    V = (V_0/l) * (np.exp(-(array - len(array)/2) ** 2 / (l**2)) + np.exp(-(array + len(array)/2) ** 2 / (l**2)))
    return V

def bogoliubov(**args):
    N = args.get('N')
    dx = args.get('dx')
    k = (2 * np.pi) / dx * np.fft.fftfreq(N, d = 1)
    k_phys = np.fft.fftshift(k)

    gamma2_tilde = args.get('gamma2')  * hatt / hatx ** 2
    gamma0_tilde = args.get('gamma0') * hatt
    Kc = hbar ** 2 / (2 * args.get('m') * melectron * hatepsilon * hatx ** 2)
    Kd = gamma2_tilde / 2
    n0_tilde = args.get('ns') * (args.get('p') - 1)
    mu_tilde = args.get('g') * n0_tilde
    p = args.get('p')
    Gamma = gamma0_tilde * (p - 1) / (2 * p)

    Im_plus = np.zeros_like(k)
    Im_minus = np.zeros_like(k)
    for i in range(len(k_phys)):
        if (- Gamma ** 2 + Kc ** 2 * k_phys[i] ** 4 + 2 * Kc * k_phys[i] ** 2 * mu_tilde) >= 0:
            Im_plus[i] = - Kd * k_phys[i] ** 2  - Gamma
            Im_minus[i] = - Kd * k_phys[i] ** 2  - Gamma
        else:
            Im_plus[i] = - Kd * k_phys[i] ** 2 - Gamma + np.sqrt(np.abs(- Gamma ** 2 + Kc ** 2 * k_phys[i] ** 4 + 2 * Kc * k_phys[i] ** 2 * mu_tilde))
            Im_minus[i] = - Kd * k_phys[i] ** 2 - Gamma - np.sqrt(np.abs(- Gamma ** 2 + Kc ** 2 * k_phys[i] ** 4 + 2 * Kc * k_phys[i] ** 2 * mu_tilde))
    return k, Im_plus, Im_minus

'''
def dimensional_units(**args):
    L_dim = L_tilde * hatx                                # result in μm
    P_dim = P_tilde * (1 / (hatx ** 2 * hatt))            # result in μm^-2 ps^-1
    R_dim = R_tilde * (hatx ** 2 / hatt)                  # result in μm^2 ps^-1
    gamma0_dim = gamma0_tilde * (1 / hatt)                # result in ps^-1
    gammar_dim = gammar_tilde * (1 / hatt)                # result in ps^-1
    ns_dim = ns_tilde * hatrho                            # result in μm^-2
    n0_dim = n0_tilde * hatrho                            # result in μm^-2
    nr_dim = nres_tilde * hatrho                          # result in μm^-2
    return L_dim, P_dim, R_dim, gamma0_dim, gammar_dim, gamma2_dim, ns_dim, n0_dim, nr_dim
'''

def space_grid(N, dx_tilde):
    x_0 = - N * dx_tilde / 2
    x = x_0 + dx_tilde * np.arange(N)
    y = x
    return x, y

def time(dt, N_input, i_start, di):
    length = int((N_input + di) / di)
    t = np.zeros(length)
    for i in range(N_input + i_start + di):
        if i >= i_start and i <= N_input + i_start + di and i % di == 0:
            t[(i - i_start) // di] = i * dt
    return t

def vortex_detect(theta, N, dx, x, y):
    positions = np.zeros((N, N))
    partialx, partialy = np.gradient(theta, dx)
    integral = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if np.abs(partialx[i, j]) > 0.6 * np.pi / dx:
                partialx[i, j] -= np.sign(partialx[i, j]) * np.pi / dx
            if np.abs(partialy[i, j]) > 0.6 * np.pi / dx:
                partialy[i, j] -= np.sign(partialy[i, j]) * np.pi / dx
    for i in range(2, N, 2):
        for j in range(2, N, 2):
            i0 = i
            j0 = j
            I1 = 1/6 * (partialy[i0 - 1, j0 - 1] + 4 * partialy[i0 - 1, j0] + partialy[i0 - 1, j0 + 1])
            I3 = - 1/6 * (partialy[i0 + 1, j0 - 1] + 4 * partialy[i0 + 1, j0] + partialy[i0 + 1, j0 + 1])
            I2 = 1/6 * (partialx[i0 - 1, j0 + 1] + 4 * partialx[i0, j0 + 1] + partialx[i0 + 1, j0 + 1])
            I4 = - 1/6 * (partialx[i0 - 1, j0 - 1] + 4 * partialx[i0, j0 - 1] + partialx[i0 + 1, j0 - 1])
            integral[i0, j0] = - (I1 + I2 + I3 + I4)
            if np.abs(integral[i, j] / (np.pi)) > 1:
                positions[i0, j0] = np.sign(integral[i0, j0])
            
            '''
            if i0 == N - 2 and j0 == N - 2:
                pl.plot(y[j0], x[i0], 'bo')
                pl.plot(y[j0 - 1], x[i0 - 1], 'go-')
                pl.plot(y[j0], x[i0 - 1], 'go-')
                pl.plot(y[j0 + 1], x[i0 - 1], 'go-')
                
                pl.plot(y[j0 - 1], x[i0 + 1], 'go-')
                pl.plot(y[j0], x[i0 + 1], 'go-')
                pl.plot(y[j0 + 1], x[i0 + 1], 'go-')
                
                pl.plot(y[j0 + 1], x[i0 - 1], 'go-')
                pl.plot(y[j0 + 1], x[i0], 'go-')
                pl.plot(y[j0 + 1], x[i0 + 1], 'go-')
                
                pl.plot(y[j0 - 1], x[i0 - 1], 'go-')
                pl.plot(y[j0 - 1], x[i0], 'go-')
                pl.plot(y[j0 - 1], x[i0 + 1], 'go-')
            '''
    return positions

class vortex_plots_class:
    def __init__(self):
        self.count = 0
    def __call__(self, folder, x, ti, vortex_positions, phase, density):
        if self.count == 0:
            x *= hatx
        X, Y = np.meshgrid(x, x)
        fig1, ax1 = pl.subplots()
        im1 = ax1.pcolormesh(X, Y, phase, cmap='twilight')
        ax1.plot(x[np.where(vortex_positions == 1)[1]], x[np.where(vortex_positions == 1)[0]], 'go', markersize = 4)
        ax1.plot(x[np.where(vortex_positions == -1)[1]], x[np.where(vortex_positions == -1)[0]], 'bo', markersize = 4)
        ax1.set_xlabel('$y[\mu m]$')
        ax1.set_ylabel('$x[\mu m]$')
        cbar1 = pl.colorbar(im1, ax = ax1)
        cbar1.ax.tick_params(labelsize = 12)
        cbar1.ax.set_ylabel(r'$\theta(x, y)$', fontsize = 12)
        #cbar1.set_ticks([-np.pi, np.pi])
        #cbar1.set_ticklabels([r'-$\pi$', r'$\pi$'])
        pl.savefig(folder + os.sep + 'fig_theta' + str(self.count) + '.pdf')
        pl.close()
        fig2, ax2 = pl.subplots()
        im2 = ax2.pcolormesh(X, Y, density / hatx ** 2, cmap='RdBu_r')
        ax2.plot(x[np.where(vortex_positions == 1)[1]], x[np.where(vortex_positions == 1)[0]], 'go', markersize = 4)
        ax2.plot(x[np.where(vortex_positions == -1)[1]], x[np.where(vortex_positions == -1)[0]], 'bo', markersize = 4)
        ax2.set_xlabel(r'$y$')
        ax2.set_ylabel(r'$x$')
        cbar2 = pl.colorbar(im2, ax=ax2)
        cbar2.ax.tick_params(labelsize = 12)
        cbar2.ax.set_ylabel(r'$n(x,y)$', fontsize = 12)
        pl.savefig(folder + os.sep + 'fig_density' + str(self.count) + '.pdf')
        pl.close()
        self.count += 1
        return None

def get_indices(x):
    indices = {}
    N = len(x)
    dx_tilde = x[1] - x[0]
    for rad_count in range(N//2):
        l = []
        for i in range(N):
            for j in range(N):
                if np.sqrt(x[i]**2 + x[j]**2) <= rad_count * dx_tilde and np.sqrt(x[i]**2 + x[j]**2) > (rad_count-1) * dx_tilde:
                    l.append([i, j])
                    indices['r = ' + str(rad_count)] = l
    return indices

def isotropic_avg(keyword, matrix, central_element, **args):
    N = len(matrix[0])
    if keyword == 'psi correlation' or keyword == 'n correlation' or keyword == 'n average':
        avg = np.zeros(N//2, dtype = complex)
    else:
        avg = np.zeros(N//2)
    for rad in range(N//2):
        indices = args.get('r = ' + str(rad))
        if keyword == 'psi correlation' or keyword == 'n correlation':
            for i in range(len(indices)):
                avg[rad] += central_element * matrix[indices[i][0], indices[i][1]] / len(indices)
        elif keyword == 'n average':
            for i in range(len(indices)):
                avg[rad] += matrix[indices[i][0], indices[i][1]] / len(indices)
        elif keyword == 'deltatheta':
            for i in range(len(indices)):
                avg[rad] += (central_element - matrix[indices[i][0], indices[i][1]]) / len(indices)
    return avg

def unwinding(deltatheta, cutoff, keyword):
    if keyword == 'distinct points':
        for i in range(len(deltatheta)):
            if abs(deltatheta[i]) > cutoff * 2 * np.pi:
                deltatheta[i] -= np.sign(deltatheta[i]) * 2 * np.pi
        return deltatheta
    if keyword == 'whole profile':
        howmany = len(deltatheta[deltatheta > cutoff * 2 * np.pi])
        if howmany == 0:
            pass
        else:
            deltatheta[deltatheta > cutoff * 2 * np.pi] -= np.sign(deltatheta[deltatheta > cutoff * 2 * np.pi]) * 2 * np.pi
        return deltatheta