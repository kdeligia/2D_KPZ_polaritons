#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 11:03:50 2019

@author: delis
"""
import os
import numpy as np
import matplotlib.pyplot as pl

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

def vortex_detect(a, theta, x, y):
    # The integral should be calculated counter clock wise. 
    #   D------I3------C
    #   |              |
    #   |              |
    #   |              |
    #   I4  (xc, yc)  I2,  I = I1 + I2 + I3 + I4
    #   |              |
    #   |              |
    #   |              |
    #   A------I1------B
    #   <-----2a------->

    positions = np.zeros((len(x), len(y)))
    for i in range(a, len(x) - 2, a + 1):
        for j in range(a, len(y) - 2, a + 1):
            i0 = i
            j0 = j
            '''
            j1 = j0 - a
            i1 = i0 - a
            
            j2 = j0 + a
            i2 = i0 - a
            
            j3 = j0 + a
            i3 = i0 + a
            
            j4 = j0 - a
            i4 = i0 + a
            pl.plot(x[j0], y[i0], 'ro')
            pl.plot(x[j1], y[i1], 'go', markersize = 0.4)
            pl.plot(x[j2], y[i2], 'go', markersize = 0.4)
            pl.plot(x[j3], y[i3], 'go', markersize = 0.4)
            pl.plot(x[j4], y[i4], 'go', markersize = 0.4)

            for q in range(len(x)):
                for w in range(len(y)):
                    pl.plot(x[q], y[w], 'go', markersize = 0.2)
            pl.show()
            '''
            '''
            pl.plot(x[j1 + 1], y[i1], 'bo')
            pl.plot(x[j2 - 1], y[i2], 'bo')
    
            pl.plot(x[j2], y[i2 + 1], 'bo')
            pl.plot(x[j3], y[i3 - 1], 'bo')
    
            pl.plot(x[j3 - 1], y[i3], 'bo')
            pl.plot(x[j4 + 1], y[i4], 'bo')
    
            pl.plot(x[j4], y[i4 - 1], 'bo')
            pl.plot(x[j1], y[i1 + 1], 'bo')
            pl.show()
            '''
            thetaplus_down = theta[i0 + a, j0 + a - 1]
            thetaminus_down = theta[i0 + a, j0 - (a - 1)]
            
            thetaplus_right = theta[i0 - (a - 1), j0 + a]
            thetaminus_right = theta[i0 + (a - 1) , j0 + a]
            
            thetaplus_up = theta[i0 - a, j0 - (a - 1)]
            thetaminus_up = theta[i0 - a, j0 + (a - 1)]
            
            thetaplus_left = theta[i0 + (a - 1), j0 - a]
            thetaminus_left = theta[i0 - (a - 1), j0 - a]
            
            if abs(thetaplus_down - thetaminus_down) >= 2 * np.pi * 0.5:
                thetaplus_down -= 2 * np.pi * np.sign(thetaplus_down - thetaminus_down)
            I1 = 0.5 * (thetaplus_down - thetaminus_down)
            if abs(thetaplus_right - thetaminus_right) >= 2 * np.pi * 0.5:
                thetaplus_right -= 2 * np.pi * np.sign(thetaplus_right - thetaminus_right)
            I2 = 0.5 * (thetaplus_right - thetaminus_right)
            if abs(thetaplus_up - thetaminus_up) >= 2 * np.pi * 0.5:
                thetaplus_up -= 2 * np.pi * np.sign(thetaplus_up - thetaminus_up)
            I3 = 0.5 * (thetaplus_up - thetaminus_up)
            if abs(thetaplus_left - thetaminus_left) >= 2 * np.pi * 0.5:
                thetaplus_left -= 2 * np.pi * np.sign(thetaplus_left - thetaminus_left)
            I4 = 0.5 * (thetaplus_left - thetaminus_left) 
            I = (I1 + I2 + I3 + I4) / (2 * np.pi)
            if np.abs(I) > 0.1:
                positions[i0, j0] = np.sign(I)
    return positions

def vortex_plots(folder, x, t, index, vortex_positions, phase, density):
    X, Y = np.meshgrid(x, x, indexing='xy')
    fig, ax = pl.subplots(2, 1, sharex=True, sharey=True, figsize=(10, 12))
    im1 = ax[0].pcolormesh(X, Y, phase, vmin = -np.pi, vmax = np.pi, cmap='twilight')
    ax[0].plot(x[np.where(vortex_positions == 1)[1]], x[np.where(vortex_positions == 1)[0]], 'go', markersize=12)
    ax[0].plot(x[np.where(vortex_positions == -1)[1]], x[np.where(vortex_positions == -1)[0]], 'bo', markersize=12)
    ax[0].set_xlabel('$x$', fontsize=  20)
    ax[0].set_ylabel('$y$', fontsize = 20)
    ax[0].tick_params(axis='both', which='both', direction='in', labelsize=16, pad=12, length=12)
    cbar1 = pl.colorbar(im1, ax = ax[0])
    cbar1.ax.tick_params(labelsize=16)
    cbar1.ax.set_ylabel(r'$\theta(x,y)$', fontsize = 20)

    #im2 = ax[1].pcolormesh(X, Y, density, vmin = 0.1 * np.mean(density), vmax = 10 * np.mean(density), cmap='RdBu_r')
    im2 = ax[1].pcolormesh(X, Y, density, cmap='RdBu_r')
    ax[1].plot(x[np.where(vortex_positions == 1)[1]], x[np.where(vortex_positions == 1)[0]], 'go', markersize=12)
    ax[1].plot(x[np.where(vortex_positions == -1)[1]], x[np.where(vortex_positions == -1)[0]], 'bo', markersize=12)
    ax[1].set_xlabel(r'$x$', fontsize = 20)
    ax[1].set_ylabel(r'$y$', fontsize = 20)
    ax[1].tick_params(axis='both', which='both', direction='in', labelsize=16, pad=12, length=12)
    cbar2 = pl.colorbar(im2, ax=ax[1])
    cbar2.ax.tick_params(labelsize=16)
    cbar2.ax.set_ylabel(r'$n(x,y)$', fontsize = 20)

    fig.suptitle('t = %.1f' % t[index], fontsize=16)
    pl.savefig(folder + os.sep + 'fig' + str(index) + '.jpg', format='jpg')
    pl.close()
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
    avg = np.zeros(N//2, dtype=complex)
    for rad in range(N//2):
        indices = args.get('r = ' + str(rad))
        if keyword == 'psi':
            for i in range(len(indices)):
                avg[rad] += central_element * matrix[indices[i][0], indices[i][1]] / len(indices)
        elif keyword == 'density':
            for i in range(len(indices)):
                avg[rad] += matrix[indices[i][0], indices[i][1]] / len(indices)
        elif keyword == 'deltatheta':
            for i in range(len(indices)):
                avg[rad] += (central_element - matrix[indices[i][0], indices[i][1]]) / len(indices)
    return avg

def unwinding(deltatheta, cutoff):
    for i in range(len(deltatheta)):
        if abs(deltatheta[i]) > cutoff * 2 * np.pi:
            deltatheta[i] -= np.sign(deltatheta[i]) * 2 * np.pi
    return deltatheta