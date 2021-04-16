#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 11:03:50 2019

@author: delis
"""

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

def vortices(index, x, t, phase):
    size = len(x)
    dx = x[1] - x[0]
    X, Y = np.meshgrid(x, x)
    count_v = 0
    count_av = 0
    grad = np.gradient(phase, dx)
    v_pos = np.zeros((size, size))
    av_pos = np.zeros((size, size))
    for i in range(1, size-1):
        for j in range(1, size-1):
            loop = (2 * dx * (grad[0][j+1, i+1] - grad[1][j+1, i+1]) +
                    2 * dx * (grad[0][j+1, i-1] + grad[1][j+1, i-1]) +
                    2 * dx * (-grad[0][j-1, i-1] + grad[1][j-1, i-1]) +
                    2 * dx * (-grad[0][j-1, i+1] - grad[1][j-1, i+1]) +
                    2 * dx * (grad[0][j+1, i] + grad[1][j, i-1] - grad[0][j-1, i] - grad[1][j, i+1]))
            if loop >= 2 * np.pi:
                count_v += 1
                v_pos[i,j] = 1
            elif loop <= - 2 * np.pi:
                count_av +=1
                av_pos[i,j] = 1
    xv = np.array([x[i] for i in range(size) for j in range(size) if v_pos[i,j]==1])
    yv = np.array([x[j] for i in range(size) for j in range(size) if v_pos[i,j]==1])
    xav = np.array([x[i] for i in range(size) for j in range(size) if av_pos[i,j]==1])
    yav = np.array([x[j] for i in range(size) for j in range(size) if av_pos[i,j]==1])
    fig,ax = pl.subplots(1,1, figsize=(10, 8))
    ax.plot(xv, yv, 'go', markersize=6)
    ax.plot(xav, yav, 'ro', markersize=6)
    ax.set_xlim(x[0], x[-1])
    ax.set_ylim(x[0], x[-1])
    im = ax.pcolormesh(X, Y, phase, vmin = -np.pi, vmax = np.pi, cmap='Greys')
    pl.colorbar(im)
    pl.title(r't = %.2f' % t[index])
    pl.savefig('/Users/delis/Desktop/vortices' + os.sep + 'fig' + str(index) + '.jpg', format='jpg')
    pl.show()
    total = count_v + count_av
    return total

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

def isotropic_avg(matrix, center, obs, **args):
    N = len(matrix[0])
    avg = np.zeros(N//2, dtype=complex)
    for rad in range(N//2):
        indices = args.get('r = ' + str(rad))
        if obs == 'psi correlation':
            for i in range(len(indices)):
                avg[rad] += np.conjugate(center) * matrix[indices[i][0], indices[i][1]] / len(indices)
        elif obs == 'density average':
            for i in range(len(indices)):
                avg[rad] += matrix[indices[i][0], indices[i][1]] / len(indices)
    return avg