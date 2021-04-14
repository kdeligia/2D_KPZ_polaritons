#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 11:03:50 2019

@author: delis
"""

import os
import numpy as np

def confining(array, V_0, l):
    V = (V_0/l) * (np.exp(-(array - len(array)/2) ** 2 / (l**2)) + np.exp(-(array + len(array)/2) ** 2 / (l**2)))
    return V

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

def vortices(index, dx_tilde, N, phase):
    count_v = 0
    count_av = 0
    grad = np.gradient(phase, dx_tilde)
    #v_pos = np.zeros((N, N))
    #av_pos = np.zeros((N, N))
    for i in range(1, N-1):
        for j in range(1, N-1):
            loop = (2*dx_tilde*(grad[0][j+1, i+1] - grad[1][j+1, i+1]) +
                    2*dx_tilde*(grad[0][j+1, i-1] + grad[1][j+1, i-1]) +
                    2*dx_tilde*(-grad[0][j-1, i-1] + grad[1][j-1, i-1]) +
                    2*dx_tilde*(-grad[0][j-1, i+1] - grad[1][j-1, i+1]) +
                    2*dx_tilde*(grad[0][j+1, i] + grad[1][j, i-1] - grad[0][j-1, i] - grad[1][j, i+1]))
            if loop >= 2 * np.pi:
                count_v += 1
                #v_pos[i,j] = 1
            elif loop <= - 2 * np.pi:
                count_av +=1
                #av_pos[i,j] = 1
    '''
    xv = np.array([x[i] for i in range(N) for j in range(N) if v_pos[i,j]==1])
    yv = np.array([x[j] for i in range(N) for j in range(N) if v_pos[i,j]==1])
    xav = np.array([x[i] for i in range(N) for j in range(N) if av_pos[i,j]==1])
    yav = np.array([x[j] for i in range(N) for j in range(N) if av_pos[i,j]==1])
    fig,ax = pl.subplots(1,1, figsize=(8,6))
    ax.plot(xv, yv, 'go', markersize=2)
    ax.plot(xav, yav, 'ro', markersize=2)
    ax.set_xlim(x[0], x[-1])
    ax.set_ylim(x[0], x[-1])
    im = ax.pcolormesh(X, Y, phase, vmin = -np.pi, vmax = np.pi, cmap='Greys')
    pl.colorbar(im)
    pl.title(r't = %.2f' % t[index])
    #pl.savefig('/Users/delis/Desktop/vortices' + os.sep + 'fig' + str(count) + '.jpg', format='jpg')
    pl.show()
    '''
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

def isotropic_avg(matrix, obs, **args):
    N = len(matrix[0])
    avg = np.zeros(N//2, dtype=complex)
    center_indices = args.get('r = ' + str(0))
    for rad in range(N//2):
        indices = args.get('r = ' + str(rad))
        if obs == 'psi correlation':
            for i in range(len(indices)):
                avg[rad] += np.conjugate(matrix[center_indices[0][0], center_indices[0][1]]) * matrix[indices[i][0], indices[i][1]] / len(indices)
        elif obs == 'density average':
            for i in range(len(indices)):
                avg[rad] += matrix[indices[i][0], indices[i][1]] / len(indices)
    return avg