#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 11:03:50 2019

@author: delis
"""
from scipy.interpolate import griddata
import os
import numpy as np
import matplotlib.pyplot as pl

def confining(array, V_0, l):
    V = (V_0/l) * (np.exp(-(array - len(array)/2) ** 2 / (l**2)) + np.exp(-(array + len(array)/2) ** 2 / (l**2)))
    return V

def dimensional_units(**args):
    #L_dim = L_tilde * hatx                                                      # result in μm
    #P_dim = P_tilde * (1/(hatx**2 * hatt))                                      # result in μm^-2 ps^-1
    #R_dim = R_tilde * (hatx**2/hatt)                                            # result in μm^2 ps^-1
    #gamma0_dim = gamma0_tilde * (1/hatt)                                        # result in ps^-1
    #gammar_dim = gammar_tilde * (1/hatt)                                        # result in ps^-1
    #ns_dim = ns_tilde * hatrho                                                  # result in μm^-2
    #n0_dim = n0_tilde * hatrho                                                  # result in μm^-2
    #nr_dim = nres_tilde * hatrho                                                # result in μm^-2
    #return L_dim, P_dim, R_dim, gamma0_dim, gammar_dim, gamma2_dim, ns_dim, n0_dim, nr_dim
    return None

def space_momentum(N, dx_tilde):
    dkx_tilde = 2 * np.pi / (N * dx_tilde)
    x_0 = - N * dx_tilde / 2
    kx0 = - np.pi / dx_tilde
    x = x_0 + dx_tilde * np.arange(N)
    kx = kx0 + dkx_tilde * np.arange(N)
    return x, kx

def time(dt, N_steps, i1, i2, secondarystep):
    lengthindex = i2-i1
    length = lengthindex//secondarystep
    t = np.zeros(length)
    for i in range(N_steps):
        if i>=i1 and i<=i2 and i%secondarystep==0:
            t[(i-i1)//secondarystep] = i*dt
    return t

def names_subfolders(create, path, sigma_array, p_array, om_array, g_dim, gr_dim, gamma0_tilde, ns_tilde):
    init = path + os.sep + 'ns' + str(int(ns_tilde)) + '_' + 'g' + str(g_dim)+ '_' + 'gr' + str(gr_dim) + '_' + 'gamma' + str(gamma0_tilde)
    if create == True:
        os.mkdir(init)
    subfolders = {}
    for sigma in sigma_array:
        for p in p_array:
            for om in om_array:
                subfolders['p=' + str(p), 'sigma=' + str(sigma)] = init + os.sep + 'sigma' + str(sigma) + '_' + 'p' + str(p) + '_' + 'om' + str(int(om))
    return subfolders

def vortices(a, theta, x, y):
    # The integral should be calculated counter clock wise. 
    #   D------I3------C
    #   |              |
    #   |              |
    #   |              |
    #   I4  (xc, yc)  I2,       I = I1 + I2 + I3 + I4
    #   |              |
    #   |              |
    #   |              |
    #   A------I1------B
    #   <-----2a------->
    dx = x[1] - x[0]
    N = len(x)
    loops = np.zeros((N, N))
    positions = np.zeros((N, N))
    for i in range(0, N//2, 2 * int(a/dx) + 1):
        col = N//2 + i
        for j in range(0, N//2, 2 * int(a/dx) + 1):
            for row in [N//2 + j, N//2 - j]:
                x0 = x[col]
                y0 = y[row]
                thetaplus  = theta[np.where(abs(y - y0) <= a)[0][0],  np.where(abs(x - x0) <= a)[0][-2]]
                thetaminus = theta[np.where(abs(y - y0) <= a)[0][0],  np.where(abs(x - x0) <= a)[0][1]]
                if abs(thetaplus - thetaminus) >= 2 * np.pi * 0.7:
                    thetaplus -= 2 * np.pi * np.sign(thetaplus - thetaminus)
                I1 = thetaplus - thetaminus
                thetaplus  = theta[np.where(abs(y - y0) <= a)[0][-2],  np.where(abs(x - x0) <= a)[0][-1]]
                thetaminus = theta[np.where(abs(y - y0) <= a)[0][1],  np.where(abs(x - x0) <= a)[0][-1]]
                if abs(thetaplus - thetaminus) >= 2 * np.pi * 0.7:
                    thetaplus -= 2 * np.pi * np.sign(thetaplus - thetaminus)
                I2 = thetaplus - thetaminus
                thetaplus  = theta[np.where(abs(y - y0) <= a)[0][-1],  np.where(abs(x - x0) <= a)[0][1]]
                thetaminus = theta[np.where(abs(y - y0) <= a)[0][-1],  np.where(abs(x - x0) <= a)[0][-2]]
                if abs(thetaplus - thetaminus) >= 2 * np.pi * 0.7:
                    thetaplus -= 2 * np.pi * np.sign(thetaplus - thetaminus)
                I3 = thetaplus - thetaminus
                thetaplus  = theta[np.where(abs(y - y0) <= a)[0][1],  np.where(abs(x - x0) <= a)[0][0]]
                thetaminus = theta[np.where(abs(y - y0) <= a)[0][-2],  np.where(abs(x - x0) <= a)[0][0]]
                if abs(thetaplus - thetaminus) >= 2 * np.pi * 0.7:
                    thetaplus -= 2 * np.pi * np.sign(thetaplus - thetaminus)
                I4 = thetaplus - thetaminus
                I = (I1 + I2 + I3 + I4) / (2 * np.pi)
                loops[row ,col] = I 
                if np.abs(I) > 0.25 and np.abs(I) < 1.:
                    positions[row, col] = np.sign(I)
    for i in range(0, N//2, 2 * int(a/dx) + 1):
        if i!= 0 :
            col = N//2 - i
            for j in range(0, N//2, 2 * int(a/dx) + 1):
                for row in [N//2 + j, N//2 - j]:
                    x0 = x[col]
                    y0 = y[row]
                    thetaplus  = theta[np.where(abs(y - y0) <= a)[0][0],  np.where(abs(x - x0) <= a)[0][-2]]
                    thetaminus = theta[np.where(abs(y - y0) <= a)[0][0],  np.where(abs(x - x0) <= a)[0][1]]
                    if abs(thetaplus - thetaminus) >= 2 * np.pi * 0.7:
                        thetaplus -= 2 * np.pi * np.sign(thetaplus - thetaminus)
                    I1 = thetaplus - thetaminus
                    thetaplus  = theta[np.where(abs(y - y0) <= a)[0][-2],  np.where(abs(x - x0) <= a)[0][-1]]
                    thetaminus = theta[np.where(abs(y - y0) <= a)[0][1],  np.where(abs(x - x0) <= a)[0][-1]]
                    if abs(thetaplus - thetaminus) >= 2 * np.pi * 0.7:
                        thetaplus -= 2 * np.pi * np.sign(thetaplus - thetaminus)
                    I2 = thetaplus - thetaminus
                    thetaplus  = theta[np.where(abs(y - y0) <= a)[0][-1],  np.where(abs(x - x0) <= a)[0][1]]
                    thetaminus = theta[np.where(abs(y - y0) <= a)[0][-1],  np.where(abs(x - x0) <= a)[0][-2]]
                    if abs(thetaplus - thetaminus) >= 2 * np.pi * 0.7:
                        thetaplus -= 2 * np.pi * np.sign(thetaplus - thetaminus)
                    I3 = thetaplus - thetaminus
                    thetaplus  = theta[np.where(abs(y - y0) <= a)[0][1],  np.where(abs(x - x0) <= a)[0][0]]
                    thetaminus = theta[np.where(abs(y - y0) <= a)[0][-2],  np.where(abs(x - x0) <= a)[0][0]]
                    if abs(thetaplus - thetaminus) >= 2 * np.pi * 0.7:
                        thetaplus -= 2 * np.pi * np.sign(thetaplus - thetaminus)
                    I4 = thetaplus - thetaminus
                    I = (I1 + I2 + I3 + I4) / (2 * np.pi)
                    loops[row, col] = I 
                    if np.abs(I) > 0.25 and np.abs(I) < 1.:
                        positions[row, col] = np.sign(I)
    return positions, loops

'''
            delta = 0
            fig, ax = pl.subplots(1,1, figsize=(10, 10))
            for n in range(2 * int(a / dx) + 1):
                ax.plot(delta, theta[np.where(y0 - y == a)[0][0],  np.where(abs(x0 - x) <= a)[0][n]], 'ro', markersize=5)
                delta += dx
                print(theta[np.where(y0 - y == a)[0][0],  np.where(abs(x0 - x) <= a)[0][n]])
            print('Next side')
            for n in range(1, 2 * int(a / dx) + 1):
                ax.plot(delta, theta[np.where(abs(y0 - y) <= a)[0][n],  np.where(x0 - x == - a)[0][0]], 'bo', markersize=5)
                delta += dx
                print(theta[np.where(abs(y0 - y) <= a)[0][n],  np.where(x0 - x == - a)[0][0]])
            print('Next side')
            for n in range(1, 2 * int(a / dx) + 1):
                ax.plot(delta, theta[np.where(y0 - y == - a)[0][0],  np.where(abs(x - x0) <= a)[0][2 * int(a / dx) - n]], 'go', markersize=5)
                delta += dx
                print(theta[np.where(y0 - y == - a)[0][0],  np.where(abs(x - x0) <= a)[0][2 * int(a / dx) - n]])
            print('Next side')
            for n in range(1, 2 * int(a / dx) + 1):
                ax.plot(delta, theta[np.where(abs(y0 - y) <= a)[0][2 * int(a / dx) - n], np.where(x0 - x == a)[0][0]], 'yo', markersize=5)
                delta += dx
                print(theta[np.where(abs(y0 - y) <= a)[0][2 * int(a / dx) - n], np.where(x0 - x == a)[0][0]])
            ax.tick_params(axis='both', which='both', direction='in', labelsize=16, pad=12, length=12)
            pl.show()
'''

def vortex_plots(x, t, index, vortex_positions, phase, density):
    path = '/Users/delis/Desktop/vortices'
    X, Y = np.meshgrid(x, x, indexing='xy')
    fig, ax = pl.subplots(2, 1, sharex=True, sharey=True, figsize=(10, 12))
    im1 = ax[0].pcolormesh(X, Y, phase, vmin = -np.pi, vmax = np.pi, cmap='twilight')
    ax[0].plot(x[np.where(vortex_positions == 1)[1]], x[np.where(vortex_positions == 1)[0]], 'go', markersize=12)
    ax[0].plot(x[np.where(vortex_positions == -1)[1]], x[np.where(vortex_positions == -1)[0]], 'bo', markersize=12)
    ax[0].set_xlabel(r'$x$', fontsize=  20)
    ax[0].set_ylabel(r'$y$', fontsize = 20)
    ax[0].tick_params(axis='both', which='both', direction='in', labelsize=16, pad=12, length=12)
    cbar1 = pl.colorbar(im1, ax = ax[0])
    cbar1.ax.tick_params(labelsize=16)
    cbar1.ax.set_ylabel(r'$\theta(x,y)$', fontsize = 20)
    im2 = ax[1].pcolormesh(X, Y, density, vmin = 0.2 * np.mean(density), vmax = 2 * np.mean(density), cmap='RdBu_r')
    ax[1].plot(x[np.where(vortex_positions == 1)[1]], x[np.where(vortex_positions == 1)[0]], 'go', markersize=12)
    ax[1].plot(x[np.where(vortex_positions == -1)[1]], x[np.where(vortex_positions == -1)[0]], 'bo', markersize=12)
    ax[1].set_xlabel(r'$x$', fontsize = 20)
    ax[1].set_ylabel(r'$y$', fontsize = 20)
    ax[1].tick_params(axis='both', which='both', direction='in', labelsize=16, pad=12, length=12)
    cbar2 = pl.colorbar(im2, ax=ax[1])
    cbar2.ax.tick_params(labelsize=16)
    cbar2.ax.set_ylabel(r'n(x,y)', fontsize = 20)
    fig.suptitle(r't = %.1f' % t[index], fontsize=16)
    pl.savefig(path + os.sep + 'fig' + str(index) + '.jpg', format='jpg')
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

def isotropic_avg(key, matrix, center, **args):
    N = len(matrix[0])
    avg = np.zeros(N//2, dtype=complex)
    for rad in range(N//2):
        indices = args.get('r = ' + str(rad))
        if key == 'psi correlation':
            for i in range(len(indices)):
                avg[rad] += np.conjugate(center) * matrix[indices[i][0], indices[i][1]] / len(indices)
        elif key == 'density average':
            for i in range(len(indices)):
                avg[rad] += matrix[indices[i][0], indices[i][1]] / len(indices)
    return avg

def unwinding(theta_wound_new, theta_wound_old, theta_unwound_old, cutoff):
    length = len(theta_wound_new)
    theta_unwound_new = np.zeros(length)
    for i in range(length):
        deltatheta = theta_wound_new [i]- theta_wound_old[i]
        if abs(deltatheta) > cutoff * 2 * np.pi:
            theta_unwound_new[i] = theta_unwound_old[i] + deltatheta - np.sign(deltatheta) * 2 * np.pi
        else:
            theta_unwound_new[i] = theta_unwound_old[i] + deltatheta
    return theta_unwound_new