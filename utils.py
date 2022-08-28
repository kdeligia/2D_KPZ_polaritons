#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 11:03:50 2019

@author: delis
"""

import os
import numpy as np

c = 3e2  # μm/ps
hbar = 6.582119569 * 1e2  # μeV ps
melectron = 0.510998950 * 1e12 / c ** 2  # μeV/(μm^2/ps^2)

hatt = 32  # ps
hatx = 4 * np.sqrt(2)
hatpsi = 1 / hatx  # μm^-1
hatrho = 1 / hatx ** 2  # μm^-2
hatepsilon = hbar / hatt  # μeV


def confining(array, V_0, ell):
    V = (V_0 / ell) * (np.exp(-(array - len(array)/2) ** 2 / (ell**2)) + np.exp(-(array + len(array)/2) ** 2 / (ell**2)))
    return V


def bogoliubov(**args):
    N = args.get('N')
    dx = args.get('dx')
    k = (2 * np.pi) / dx * np.fft.fftfreq(N, d=1)
    k_phys = np.fft.fftshift(k)

    gamma2_tilde = args.get('gamma2') * hatt / hatx ** 2
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
            Im_plus[i] = - Kd * k_phys[i] ** 2 - Gamma
            Im_minus[i] = - Kd * k_phys[i] ** 2 - Gamma
        else:
            Im_plus[i] = - Kd * k_phys[i] ** 2 - Gamma + np.sqrt(np.abs(- Gamma ** 2 + Kc ** 2 * k_phys[i] ** 4 + 2 * Kc * k_phys[i] ** 2 * mu_tilde))
            Im_minus[i] = - Kd * k_phys[i] ** 2 - Gamma - np.sqrt(np.abs(- Gamma ** 2 + Kc ** 2 * k_phys[i] ** 4 + 2 * Kc * k_phys[i] ** 2 * mu_tilde))
    return k, Im_plus, Im_minus


def space_grid(N, dx_tilde):
    x_0 = - N * dx_tilde / 2
    x = x_0 + dx_tilde * np.arange(N)
    y = np.copy(x)
    return x, y


def full_id(**params):
    p = params.get('p')
    gamma0 = params.get('gamma0')
    gamma2 = params.get('gamma2')
    g = params.get('g')
    gr = params.get('gr')
    ns = params.get('ns')
    m = params.get('m')
    return os.path.join('m', str(m), '_',
                       'p', str(p), '_',
                       'gamma', str(gamma0), '_',
                       'gammak', str(gamma2), '_',
                       'g', str(g), '_',
                       'gr', str(gr), '_',
                       'ns', str(ns)).replace('/', '')


def mksubfolder(string):
    try:
        os.mkdir(string)
    except FileExistsError:
        pass


def get_radial_indices(x):
    indices = {}
    N = len(x)
    dx_tilde = x[1] - x[0]
    for rad_count in range(N//2):
        mylist = []
        for i in range(N):
            for j in range(N):
                if np.sqrt(x[i]**2 + x[j]**2) <= rad_count * dx_tilde and np.sqrt(x[i]**2 + x[j]**2) > (rad_count-1) * dx_tilde:
                    mylist.append([i, j])
                    indices['r = ' + str(rad_count)] = mylist
    return indices


def isotropic_avg(keyword, matrix, central_element, **args):
    N = len(matrix[0])
    if keyword == 'psi correlation' or keyword == 'n correlation' or keyword == 'n average':
        avg = np.zeros(N//2, dtype=complex)
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


def unwinding(deltatheta):
    howmany = len(deltatheta[abs(deltatheta) > np.pi])
    if howmany == 0:
        return deltatheta
    else:
        deltatheta[abs(deltatheta) > np.pi] -= np.sign(deltatheta[abs(deltatheta) > np.pi]) * 2 * np.pi
        return deltatheta


def ensemble_average(folder):
    result = []
    for file in os.listdir(folder):
        if not file.startswith('.'):
            result.append(np.load(folder + os.sep + file))                      # Take care here, depending on how you save maybe you want to use loadtxt instead. I use .npy to conserve space, but if you use .dat or .txt you need loadtxt!
    return sum(result) / len(result)


def append_theta_trajectories(folder):
    result = []
    for file in os.listdir(folder):
        if not file.startswith('.'):
            result.append(np.load(folder + os.sep + file))
    return np.vstack(result)
