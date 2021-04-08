#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 15:09:50 2020

@author: delis
"""

c = 3E2 #μm/ps
hbar = 6.582119569 * 1E2 # μeV ps

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

# =============================================================================
# 
# =============================================================================

N = 2**6
L_tilde = 2**6
dx_tilde = 0.5
dkx_tilde = 2 * np.pi / (N * dx_tilde)

def dimensional_units():
    L_dim = L_tilde * hatx                                                      # result in μm
    P_dim = P_tilde * (1/(hatx**2 * hatt))                                      # result in μm^-2 ps^-1
    R_dim = R_tilde * (hatx**2/hatt)                                            # result in μm^2 ps^-1
    gamma0_dim = gamma0_tilde * (1/hatt)                                        # result in ps^-1
    gammar_dim = gammar_tilde * (1/hatt)                                        # result in ps^-1
    ns_dim = ns_tilde * hatrho                                                  # result in μm^-2
    n0_dim = n0_tilde * hatrho                                                  # result in μm^-2
    nr_dim = nres_tilde * hatrho                                                # result in μm^-2
    return L_dim, P_dim, R_dim, gamma0_dim, gammar_dim, gamma2_dim, ns_dim, n0_dim, nr_dim

def arrays():
    x_0 = - N * dx_tilde / 2
    kx0 = - np.pi / dx_tilde
    x = x_0 + dx_tilde * np.arange(N)
    kx = kx0 + dkx_tilde * np.arange(N)
    return x, kx

x, kx =  arrays()
X, Y = np.meshgrid(x, x)
KX, KY = np.meshgrid(kx, kx)

time_steps = 400000
dt_tilde = 1e-3
every = 20000
i1 = 20000
i2 = time_steps
lengthwindow = i2-i1
t = ext.time(dt_tilde, time_steps, i1, i2, every)

class model:
    def __init__(self, sigma, p, om_tilde, g_dim, gr_dim, psi_x=0):
        self.sigma = sigma
        self.p = p
        self.g_tilde = g_dim * hatrho / hatepsilon
        self.gr_tilde = gr_dim * hatrho / hatepsilon
        self.damp = 1 + 2 * self.p * self.gr_tilde / (R_tilde * om_tilde) + self.p / (2 * om_tilde) * 1j 
        self.psi_x = psi_x
        self.psi_x = np.full((N, N), 0.01**(1/2))
        self.psi_x /= hatpsi
        self.psi_mod_k = fft2(self.psi_mod_x)

    def _set_fourier_psi_x(self, psi_x):
        self.psi_mod_x = psi_x * np.exp(-1j * KX[0,0] * X - 1j * KY[0,0] * Y) * dx_tilde * dx_tilde / (2 * np.pi)

    def _get_psi_x(self):
        return self.psi_mod_x * np.exp(1j * KX[0,0] * X + 1j * KY[0,0] * Y) * 2 * np.pi / (dx_tilde * dx_tilde)

    def _set_fourier_psi_k(self, psi_k):
        self.psi_mod_k = psi_k * np.exp(1j * X[0,0] * dkx_tilde * np.arange(N) + 1j * Y[0,0] * dkx_tilde * np.arange(N))

    def _get_psi_k(self):
        return self.psi_mod_k * np.exp(-1j * X[0,0] * dkx_tilde * np.arange(N) - 1j * Y[0,0] * dkx_tilde * np.arange(N))

    psi_x = property(_get_psi_x, _set_fourier_psi_x)
    psi_k = property(_get_psi_k, _set_fourier_psi_k)
# =============================================================================
# Definition of the split steps
# =============================================================================
    def n(self):
        return (self.psi_x * np.conjugate(self.psi_x)).real

    def prefactor_x(self):
        if self.g_tilde == 0:
            self.uc_tilde = 0
        else:
            self.uc_tilde = self.g_tilde * (self.n() + 2 * self.p * (self.gr_tilde / self.g_tilde) * (gamma0_tilde / R_tilde) * (1 / (1 + self.n() / ns_tilde)))
        self.I_tilde = (gamma0_tilde / 2) * (self.p / (1 + self.n() / ns_tilde) - 1)
        return np.exp(-1j * 0.5 * dt_tilde * (self.uc_tilde + 1j * self.I_tilde) / self.damp)

    def prefactor_k(self):
        return np.exp(-1j * dt_tilde * (KX ** 2 + KY ** 2) * Kc / self.damp)

# =============================================================================
# Time evolution
# =============================================================================
    def time_evolution(self):
        np.random.seed()
        '''
        g1_x = np.zeros(int(N/2), dtype = complex)
        d1_x = np.zeros(int(N/2), dtype = complex)
        '''
        g1_x = np.zeros((len(t), int(N/2)), dtype = complex)
        d1_x = np.zeros((len(t), int(N/2)))
        for i in range(time_steps+1):
            #self.sigma = gamma0_tilde * (self.p / (1 + self.n() / ns_tilde) + 1) / 4
            self.psi_x *= self.prefactor_x()
            self.psi_mod_k = fft2(self.psi_mod_x)
            self.psi_k *= self.prefactor_k()
            self.psi_mod_x = ifft2(self.psi_mod_k)
            self.psi_x *= self.prefactor_x()
            self.psi_x += np.sqrt(dt_tilde) * np.sqrt(self.sigma / dx_tilde ** 2) * (np.random.normal(0, 1, (N,N)) + 1j * np.random.normal(0, 1, (N,N))) / self.damp
            if i>=i1 and i<=i2 and i%every==0:
                time_array_index = (i-i1)//every
                for n in range(0, N, int(N/8)):
                    g1_x[time_array_index] += np.conjugate(self.psi_x[n, int(N/2)]) * self.psi_x[n, int(N/2):] / 8
                    d1_x[time_array_index] += np.conjugate(self.psi_x[n, int(N/2):]) * self.psi_x[n, int(N/2):] / 8
        '''
        for i in range(0, N, int(N/8)):
            g1_x += np.conjugate(self.psi_x[i, int(N/2)]) * self.psi_x[i, int(N/2):] / 8
            d1_x += np.conjugate(self.psi_x[i, int(N/2):]) * self.psi_x[i, int(N/2):] / 8
        '''
        return g1_x, d1_x

# =============================================================================
# Parallel tests
# =============================================================================
from qutip import *
parallel_tasks = 128
n_batch = 64
n_internal = parallel_tasks//n_batch
qutip.settings.num_cpus = n_batch

sigma_array = np.array([2.1e-2])
p_knob_array = np.array([2])
om_knob_array = np.array([1e9])
p_array = p_knob_array * P_tilde * R_tilde / (gamma0_tilde * gammar_tilde)
gr_dim = 0
g_dim = 0

path_init_cluster = r'/scratch/konstantinos'
final_save_cluster = r'/home6/konstantinos'

def names_subfolders(sigma_array, p_array):
    save_folder = path_init_cluster + os.sep + 'spat' + '_' + 'g' + str(g_dim)+ '_' + 'gr' + str(gr_dim) + '_' + 'gamma' + str(gamma0_tilde)
    os.mkdir(save_folder)
    subfolders = {}
    for sigma in sigma_array:
        for p in p_array:
            subfolders[str(p), str(sigma)] = save_folder + os.sep + 'sigma' + str(sigma) + '_' + 'p' + str(p) + '_' + 'om' + str(int(om_knob_array[0]))
    return subfolders

def x_g1_parallel(i_batch, sigma, p, om, g_dim, gr_dim):
        g1_x_batch = np.zeros((len(t), int(N/2)), dtype=complex)
        d1_x_batch = np.zeros((len(t),int(N/2)))
        for i_n in range(n_internal):
            gpe = model(sigma, p, om, g_dim, gr_dim)
            g1_x_run, d1_x_run = gpe.time_evolution()
            g1_x_batch += g1_x_run / n_internal
            d1_x_batch += d1_x_run / n_internal
        np.save(subfolders[str(p), str(sigma)] + os.sep + 'numerator' +'core' + str(i_batch + 1) + '.npy', g1_x_batch)
        np.save(subfolders[str(p), str(sigma)] + os.sep + 'denominator' + 'core' + str(i_batch + 1) + '.npy', d1_x_batch)

subfolders = names_subfolders(sigma_array, p_array)

def call_avg():
    subfolders = names_subfolders(sigma_array, p_array)
    for sigma in sigma_array:
        for p in p_array:
            numerator = np.array((len(t), int(N/2)), dtype=complex)
            denominator = np.array((len(t), int(N/2)))
            if subfolders[str(p), str(sigma)] in os.listdir(path_init_cluster):
                print(f'Folder "{subfolders[str(p), str(sigma)]}" exists.')
            else:
                os.mkdir(subfolders[str(p), str(sigma)])
                print(f'Folder "{subfolders[str(p), str(sigma)]}" succesfully created!')
            print('Starting simulations for sigma = %.2f, p = %.1f' % (sigma, p))
            parallel_map(x_g1_parallel, range(n_batch), task_kwargs=dict(sigma=sigma, p=p, om=om_knob_array[0], g_dim=g_dim, gr_dim=gr_dim))
            for file in os.listdir(subfolders[str(p), str(sigma)]):
                if 'numerator' and '.npy' in file:
                    numerator += np.load(subfolders[str(p), str(sigma)] + os.sep + file)/ n_batch
                elif 'denominator' and 'npy' in file:
                    denominator += np.load(subfolders[str(p), str(sigma)] + os.sep + file)/ n_batch
            for i in range(len(t)):
                denominator[i] *= denominator[i, 0]
            np.save(final_save_cluster + os.sep + 'spat_g1' + 
                '_' + 'sigma' + str(sigma) + 
                '_' + 'p' + str(p) + 
                '_' + 'om' + str(int(om_knob_array[0])) + 
                '_' + 'g' + str(g_dim) + '_' + 'gr' + str(gr_dim) + '_' + 'gamma' + str(gamma0_tilde) +'.npy', np.abs(numerator)/np.sqrt(denominator))

