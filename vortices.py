#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 19 13:53:23 2021

@author: delis
"""

c = 3e2 #μm/ps
hbar = 6.582119569 * 1e2 # μeV ps

from scipy.ndimage import gaussian_filter
import os
import numpy as np
import external as ext
from scipy.fftpack import fft2, ifft2

'''
import matplotlib.pyplot as pl
from matplotlib import rc
from matplotlib.texmanager import TexManager
import re
pl.rc('font', family='sans-serif')
pl.rc('text', usetex=True)
pl.close('all')
'''

hatt = 1 # ps
hatx = 1 # μm
hatpsi = 1 / hatx # μm^-1
hatrho = 1 / hatx ** 2 # μm^-2
hatepsilon = hbar / hatt # μeV
melectron = 0.510998950 * 1e12 / c ** 2 # μeV/(μm^2/ps^2)

# =============================================================================
# 
# =============================================================================
N = 2 ** 7
dx_tilde = 0.5

sampling_begin = 0
sampling_step = 5000
N_steps = 10000000 + sampling_begin + sampling_step
dt_tilde = 1e-3
sampling_end = N_steps
sampling_window = sampling_end - sampling_begin
t = ext.time(dt_tilde, N_steps, sampling_begin, sampling_end, sampling_step)

x, y = ext.space_momentum(N, dx_tilde)
isotropic_indices = ext.get_indices(x)

kx = (2 * np.pi) / dx_tilde * np.fft.fftfreq(N, d = 1)
ky = (2 * np.pi) / dx_tilde * np.fft.fftfreq(N, d = 1)

class model:
    def __init__(self, p, sigma, gamma0, gamma2, g, gr, ns, m):
        self.KX, self.KY = np.meshgrid(kx, ky, sparse = True)
        self.gamma2_tilde = gamma2  * hatt / hatx ** 2
        self.gamma0_tilde = gamma0 * hatt
        self.gammar_tilde = 0.1 * self.gamma0_tilde
        
        self.Kc = hbar ** 2 / (2 * m * melectron * hatepsilon * hatx ** 2)
        self.Kd = self.gamma2_tilde / 2
        self.g_tilde = g * hatrho / hatepsilon
        self.gr_tilde = gr * hatrho / hatepsilon
        
        self.R_tilde = self.gammar_tilde / ns
        self.ns_tilde = self.gammar_tilde / self.R_tilde
        self.P_tilde = p * self.gamma0_tilde * self.ns_tilde
        self.p = self.P_tilde * self. R_tilde / (self.gamma0_tilde * self.gammar_tilde)
        self.sigma = sigma

        self.initcond = np.full((N,N), 5)
        rot = np.ones((N, N), dtype = complex)
        self.psi_x = rot * self.initcond
        self.psi_x /= hatpsi

# =============================================================================
# Definition of the split steps
# =============================================================================
    def n(self, psi):
        return (psi * np.conjugate(psi)).real

    def exp_x(self, dt, n):
        if self.g_tilde == 0:
            self.uc_tilde = 0
        else:
            self.uc_tilde = self.g_tilde * (n + 2 * self.p * (self.gr_tilde / self.g_tilde) * (self.gamma0_tilde / self.R_tilde) * (1 / (1 + n / self.ns_tilde)))
        self.rd_tilde = (self.gamma0_tilde / 2) * (self.p / (1 + n / self.ns_tilde) - 1)
        return np.exp(-1j * dt * (self.uc_tilde + 1j * self.rd_tilde))

    def exp_k(self, dt):
        return np.exp(-1j * dt * (self.KX ** 2 + self.KY ** 2) * (self.Kc - 1j * self.Kd))

# =============================================================================
# Time evolution
# =============================================================================
    def time_evolution(self, folder):
        np.random.seed()
        a_vort = 2 * dx_tilde
        vortex_number = np.zeros(len(t))
        density = np.zeros(len(t))
        for i in range(N_steps):
            self.psi_x *= self.exp_x(0.5 * dt_tilde, self.n(self.psi_x))
            psi_k = fft2(self.psi_x)
            psi_k *= self.exp_k(dt_tilde)
            self.psi_x = ifft2(psi_k)
            self.psi_x *= self.exp_x(0.5 * dt_tilde, self.n(self.psi_x))
            self.psi_x += np.sqrt(dt_tilde) * np.sqrt(self.sigma / dx_tilde ** 2) * (np.random.normal(0, 1, (N,N)) + 1j * np.random.normal(0, 1, (N,N)))
            if i >= sampling_begin and i <= sampling_end and i % sampling_step == 0:
                time_index = (i - sampling_begin) // sampling_step
                print(time_index)
                vortex_positions, ignore = ext.vortex_positions(a_vort, np.angle(self.psi_x), x, y)
                ext.vortex_plots(folder, x, t, time_index, vortex_positions, np.angle(self.psi_x), self.n(self.psi_x))
                vortex_number[time_index] = len(np.where(vortex_positions == 1)[0]) + len(np.where(vortex_positions == -1)[0])
                density[time_index] = np.mean(self.n(self.psi_x))
                if t[time_index] < 5000 or t[time_index] > 7000:
                    self.sigma = 0
        return vortex_number, density

# =============================================================================
# 
# =============================================================================
from qutip import *
n_batch = 1
qutip.settings.num_cpus = n_batch

p_array = np.array([2])
gamma2_array = np.array([0.1])
gamma0_array = np.array([10])
sigma_array = np.array([0.02])
g_array = np.array([0])
m_array = np.array([1e-4])
gr = 0
ns = 1.

print('------- Kinetic term (real) ------- : %.5f' % (hbar ** 2 / (2 * m_array[0] * melectron * hatepsilon * hatx ** 2)))
path_local = r'/scratch/konstantinos'
save_local = r'/home6/konstantinos'

init = path_local + os.sep + 'N' + str(N) + '_' + 'ns' + str(int(ns)) + '_' + 'm' + str(m_array[0])
if os.path.isdir(init) == False:
    os.mkdir(init)
ids = ext.ids(p_array, sigma_array, gamma0_array, gamma2_array, g_array)

def vortices(p, gamma0, gamma2, g):
    sigma = sigma_array[np.where(p_array == p)[0][0]]
    print('--- Secondary simulation parameters: p = %.1f, sigma = %.2f, g = %.1f, ns = %.i' % (p, sigma, g, ns))
    id_string = ids.get(('p=' + str(p), 'sigma=' + str(sigma), 'gamma0=' + str(gamma0), 'gamma2=' + str(gamma2), 'g=' + str(g)))
    save_folder = init + os.sep + id_string
    if os.path.isdir(save_folder) == False:
        os.mkdir(save_folder)
    gpe = model(p, sigma, gamma0, gamma2, g, gr = gr, ns = ns, m = m_array[0])
    nvort, dens = gpe.time_evolution(save_folder)
    np.savetxt(save_local + os.sep + id_string + '_' + 'nv' + '.dat', nvort)
    np.savetxt(save_local + os.sep + id_string + '_' + 'dens' + '.dat', dens)
    os.system(
        'ffmpeg -framerate 10 -i ' + 
        save_folder + os.sep + 
        'fig%d.jpg ' + 
        save_local + os.sep + 
        id_string + '.mp4')
    return None

def parallel(p):
    g = g_array[0]
    for gamma2 in gamma2_array:
        gamma0 = gamma0_array[np.where(gamma2_array == gamma2)[0][0]]
        print('--- Primary simulation parameters: gamma0 = %.f, gamma2 = %.2f' % (gamma0, gamma2))
        parallel_map(vortices, p, task_kwargs=dict(gamma0 = gamma0, gamma2 = gamma2, g = g))
    return None

parallel(p_array)