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

def vortices(x, phase):
    size = len(x)
    dx = x[1] - x[0]
    count_v = 0
    count_av = 0
    grad_y, grad_x = np.gradient(phase, dx, dx)
    positions = np.zeros((size, size))
    for y0 in range(1, size-1):
        for x0 in range(1, size-1):
            Cd = 2 * dx * (grad_x[y0 + 1, x0 - 1] + grad_x[y0 + 1, x0] + grad_x[y0 + 1, x0 + 1])
            Cu = 2 * dx * (- grad_x[y0 - 1, x0 - 1] - grad_x[y0 - 1, x0] - grad_x[y0 - 1, x0 - 1])
            Cl = 2 * dx * (grad_y[y0 - 1, x0 - 1] + grad_y[y0, x0 - 1] + grad_y[y0 + 1, x0 - 1])
            Cr = 2 * dx * (- grad_y[y0 + 1, x0 + 1] - grad_y[y0, x0 + 1] - grad_y[y0 - 1, x0 + 1])
            loop = Cd + Cu + Cl + Cr
            if (np.abs(loop) % (2 * np.pi) >= 0.95) and (np.abs(loop) % (2 * np.pi) <= 1.05) and loop > 0 :
                count_v += 1
                positions[y0, x0] = 1
            if (np.abs(loop) % (2 * np.pi) >= 0.95) and (np.abs(loop) % (2 * np.pi) <= 1.05) and loop < 0 :
                count_av += 1
                positions[y0, x0] = -1
    total_number = count_v + count_av
    return total_number, positions

def vortex_plots(x, t, index, positions, phase):
    size = len(x)
    X, Y = np.meshgrid(x, x, indexing='ij')
    fig,ax = pl.subplots(1,1, figsize=(10, 8))
    for y0 in range(size):
        for x0 in range(size):
            if positions[y0, x0] == 1:
                ax.plot(x[y0], x[x0], 'go')
            elif positions[y0, x0] == -1:
                ax.plot(x[y0], x[x0], 'bo')
    ax.plot(x[size//2], x[size//2], 'ro', markersize=4)
    im = pl.pcolormesh(X, Y, phase, vmin = -np.pi, vmax = np.pi, cmap = 'twilight')
    pl.colorbar(im)
    pl.title(r't = %.1f' % t[index])
    pl.savefig('/Users/delis/Desktop/vortices' + os.sep + 'fig' + str(index) + '.jpg', format='jpg')
    pl.close();
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
    if type(theta_wound_new) is np.float64:
        length = 1
        theta_unwound_new = np.zeros(length)
        deltatheta = theta_wound_new - theta_wound_old
        if abs(deltatheta) > cutoff * 2 * np.pi:
            theta_unwound_new = theta_unwound_old + deltatheta - np.sign(deltatheta) * 2 * np.pi
        else:
            theta_unwound_new = theta_unwound_old + deltatheta
    else:
        length = len(theta_wound_new)
        theta_unwound_new = np.zeros(length)
        for i in range(length):
            deltatheta = theta_wound_new [i]- theta_wound_old[i]
            if abs(deltatheta) > cutoff * 2 * np.pi:
                theta_unwound_new[i] = theta_unwound_old[i] + deltatheta - np.sign(deltatheta) * 2 * np.pi
            else:
                theta_unwound_new[i] = theta_unwound_old[i] + deltatheta
    return theta_unwound_new