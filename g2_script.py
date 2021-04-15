#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 13 16:21:57 2021

@author: delis
"""

c = 3e2 #μm/ps
hbar = 6.582119569 * 1e2 # μeV ps

#from scipy.ndimage import gaussian_filter
import os
from scipy.fftpack import fft2, ifft2
import numpy as np
import external as ext

hatt = 1 # ps
hatx = 1 # μm
hatpsi = 1/hatx # μm^-1
hatrho = 1/hatx**2 # μm^-2
hatepsilon = hbar/hatt # μeV
melectron = 0.510998950 * 1e12 / c**2 # μeV/(μm^2/ps^2)

m_tilde = 3.8e-5 * 3
m_dim = m_tilde * melectron
gamma0_tilde = 0.2 * 100
gammar_tilde = gamma0_tilde * 0.1
P_tilde = gamma0_tilde * 1
R_tilde = gammar_tilde / 1
ns_tilde = gammar_tilde / R_tilde
Kc = hbar**2 / (2 * m_dim * hatepsilon * hatx**2)

print('ns = %.i' % (ns_tilde * hatrho))
print('Kc = %.4f' % Kc)

# =============================================================================
# 
# =============================================================================
N = 2 ** 6
L_tilde = 2 ** 6
dx_tilde = 0.5

N_steps = 500000
dt_tilde = 1e-2
every = 500
i1 = 20000
i2 = N_steps
lengthwindow = i2-i1

t = ext.time(dt_tilde, N_steps, i1, i2, every)
x, kx =  ext.space_momentum(N, dx_tilde)
isotropic_indices = ext.get_indices(x)
X, Y = np.meshgrid(x, x)
KX, KY = np.meshgrid(kx, kx)

class model:
    def __init__(self, p, sigma, om_tilde, g_dim, gr_dim, psi_x=0):
        self.dkx_tilde = kx[1] - kx[0]
        self.sigma = sigma
        self.om_tilde = om_tilde
        self.p = p
        self.g_tilde = g_dim * hatrho / hatepsilon
        self.gr_tilde = gr_dim * hatrho / hatepsilon
        self.damp = 1 + 2 * self.p * self.gr_tilde / (R_tilde * self.om_tilde) + self.p / (2 * self.om_tilde) * 1j 
        self.psi_x = psi_x
        self.psi_x = np.full((N, N), 0.01**(1/2))
        self.psi_x /= hatpsi
        self.psi_mod_k = fft2(self.psi_mod_x)

    def _set_fourier_psi_x(self, psi_x):
        self.psi_mod_x = psi_x * np.exp(-1j * KX[0,0] * X - 1j * KY[0,0] * Y) * dx_tilde * dx_tilde / (2 * np.pi)

    def _get_psi_x(self):
        return self.psi_mod_x * np.exp(1j * KX[0,0] * X + 1j * KY[0,0] * Y) * 2 * np.pi / (dx_tilde * dx_tilde)

    def _set_fourier_psi_k(self, psi_k):
        self.psi_mod_k = psi_k * np.exp(1j * X[0,0] * self.dkx_tilde * np.arange(N) + 1j * Y[0,0] * self.dkx_tilde * np.arange(N))

    def _get_psi_k(self):
        return self.psi_mod_k * np.exp(-1j * X[0,0] * self.dkx_tilde * np.arange(N) - 1j * Y[0,0] * self.dkx_tilde * np.arange(N))

    psi_x = property(_get_psi_x, _set_fourier_psi_x)
    psi_k = property(_get_psi_k, _set_fourier_psi_k)
# =============================================================================
# Definition of the split steps
# =============================================================================
    def n(self, psi):
        return (psi * np.conjugate(psi)).real

    def prefactor_x(self, n):
        if self.g_tilde == 0:
            self.uc_tilde = 0
        else:
            self.uc_tilde = self.g_tilde * (n + 2 * self.p * (self.gr_tilde / self.g_tilde) * (gamma0_tilde / R_tilde) * (1 / (1 + n / ns_tilde)))
        self.rd_tilde = (gamma0_tilde / 2) * (self.p / (1 + n / ns_tilde) - 1)
        return np.exp(-1j * 0.5 * dt_tilde * (self.uc_tilde + 1j * self.rd_tilde) / self.damp)

    def prefactor_k(self):
        return np.exp(-1j * dt_tilde * (KX ** 2 + KY ** 2) * Kc / self.damp)

# =============================================================================
# Time evolution
# =============================================================================
    def time_evolution(self):
        np.random.seed()
        n_sqrt_avg_t = np.zeros(len(t), dtype = complex)
        n_correlation_t = np.zeros(len(t), dtype=complex)
        for i in range(N_steps):
            self.psi_x *= self.prefactor_x(self.n(self.psi_x))
            self.psi_mod_k = fft2(self.psi_mod_x)
            self.psi_k *= self.prefactor_k()
            self.psi_mod_x = ifft2(self.psi_mod_k)
            self.psi_x *= self.prefactor_x(self.n(self.psi_x))
            self.psi_x += np.sqrt(dt_tilde) * np.sqrt(self.sigma / dx_tilde ** 2) * (np.random.normal(0, 1, (N,N)) + 1j * np.random.normal(0, 1, (N,N))) / self.damp
            if i>=i1 and i<=i2 and i%every==0:
                time_array_index = (i-i1)//every
                if i == i1:
                    n_sampling_begin = self.n(self.psi_x)[N//2, N//2]
                n_sqrt_avg_t[time_array_index] = np.sqrt(self.n(self.psi_x)[N//2, N//2])
                n_correlation_t[time_array_index] = np.sqrt(n_sampling_begin) * np.sqrt(self.n(self.psi_x)[N//2, N//2])
        return n_correlation_t, n_sqrt_avg_t

# =============================================================================
# Parallel tests
# =============================================================================
from qutip import *
parallel_tasks = 128
n_batch = 64
n_internal = parallel_tasks//n_batch
qutip.settings.num_cpus = n_batch

sigma_array = np.array([1e-2])
p_knob_array = np.array([1.5, 2., 3., 5.])
om_knob_array = np.array([1e9])
p_array = p_knob_array * P_tilde * R_tilde / (gamma0_tilde * gammar_tilde)
gr_dim = 0
g_dim = 0

path = r'/scratch/konstantinos'
final_save = r'/home6/konstantinos'

def names_subfolders(sigma_array, p_array):
    save_folder = path + os.sep + 'ns' + str(int(ns_tilde)) + '_' + 'g' + str(g_dim)+ '_' + 'gr' + str(gr_dim) + '_' + 'gamma' + str(gamma0_tilde)
    if save_folder in os.listdir(path):
        print(f'Folder "{save_folder}" exists.')
    else:
        os.mkdir(save_folder)
        print(f'Folder "{save_folder}" succesfully created.')
    subfolders = {}
    for sigma in sigma_array:
        for p in p_array:
            subfolders['p=' + str(p), 'sigma=' + str(sigma)] = save_folder + os.sep + 'g2' + '_' + 'sigma' + str(sigma) + '_' + 'p' + str(p) + '_' + 'om' + str(int(om_knob_array[0]))
    return subfolders

def g2(i_batch, p, sigma, om_tilde, g_dim, gr_dim):
        n_correlation_t_batch = np.zeros(len(t), dtype = complex)
        n_sqrt_avg_t_batch = np.zeros(len(t), dtype = complex)
        for i_n in range(n_internal):
            gpe = model(p, sigma, om_tilde, g_dim, gr_dim)
            n_correlation_t_run, n_sqrt_avg_t_run = gpe.time_evolution()
            n_correlation_t_batch += n_correlation_t_run / n_internal
            n_sqrt_avg_t_batch += n_sqrt_avg_t_run / n_internal
        np.save(subfolders['p=' + str(p), 'sigma=' + str(sigma)] + os.sep + 'density_correlation' + '_' +'core' + str(i_batch + 1) + '.npy', n_correlation_t_batch)
        np.save(subfolders['p=' + str(p), 'sigma=' + str(sigma)] + os.sep + 'density_average' + '_' + 'core' + str(i_batch + 1) + '.npy', n_sqrt_avg_t_batch)

subfolders = names_subfolders(sigma_array, p_array)

def call_avg():
    for sigma in sigma_array:
        for p in p_array:
            n_correlation_t = np.zeros(len(t), dtype = complex)
            n_sqrt_avg_t = np.zeros(len(t), dtype = complex)
            os.mkdir(subfolders['p=' + str(p), 'sigma=' + str(sigma)])
            print('Starting g2 simulations for sigma = %.2f, p = %.1f' % (sigma, p))
            parallel_map(g2, range(n_batch), task_kwargs=dict(p=p, sigma=sigma, om_tilde=om_knob_array[0], g_dim=g_dim, gr_dim=gr_dim))
            for file in os.listdir(subfolders['p=' + str(p), 'sigma=' + str(sigma)]):
                if 'density_correlation' in file:
                    n_correlation_t += np.load(subfolders['p=' + str(p), 'sigma=' + str(sigma)] + os.sep + file) / n_batch
                elif 'density_average' in file:
                    n_sqrt_avg_t += np.load(subfolders['p=' + str(p), 'sigma=' + str(sigma)] + os.sep + file) / n_batch
            np.save(final_save + os.sep + 'g2' + 
                '_' + 'sigma' + str(sigma) + 
                '_' + 'p' + str(p) + 
                '_' + 'om' + str(int(om_knob_array[0])) + 
                '_' + 'g' + str(g_dim) + '_' + 'gr' + str(gr_dim) + '_' + 'gamma' + str(gamma0_tilde) +'.npy', (np.abs(n_correlation_t)/(n_sqrt_avg_t[0] * n_sqrt_avg_t).real))

call_avg()