#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 15:09:50 2020

@author: delis
"""

c = 3e2 #μm/ps
hbar = 6.582119569 * 1e2 # μeV ps

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
        g1_x = np.zeros(N//2, dtype = complex)
        d1_x = np.zeros(N//2, dtype = complex)
        g1_t = np.zeros(len(t), dtype = complex)
        d1_t = np.zeros(len(t), dtype = complex)
        for i in range(N_steps):
            self.psi_x *= self.prefactor_x()
            self.psi_mod_k = fft2(self.psi_mod_x)
            self.psi_k *= self.prefactor_k()
            self.psi_mod_x = ifft2(self.psi_mod_k)
            self.psi_x *= self.prefactor_x()
            self.psi_x += np.sqrt(dt_tilde) * np.sqrt(self.sigma / dx_tilde ** 2) * (np.random.normal(0, 1, (N,N)) + 1j * np.random.normal(0, 1, (N,N))) / self.damp
            if i>=i1 and i<=i2 and i%every==0:
                time_array_index = (i-i1)//every
                if i == i1:
                    psi_sampling_begin = self.psi_x[N//2, N//2]
                g1_t[time_array_index] = np.conjugate(psi_sampling_begin) * self.psi_x[N//2, N//2] 
                d1_t[time_array_index] = self.n()[N//2, N//2]
        g1_x = ext.isotropic_avg(self.psi_x, 'psi', **isotropic_indices)
        d1_x = ext.isotropic_avg(self.psi_x, 'dens', **isotropic_indices)
        return g1_x, d1_x.real, g1_t, d1_t.real

# =============================================================================
# Parallel tests
# =============================================================================
from qutip import *
parallel_tasks = 64
n_batch = 64
n_internal = parallel_tasks//n_batch
qutip.settings.num_cpus = n_batch

sigma_array = np.array([1e-2])
p_knob_array = np.array([2])
om_knob_array = np.array([1e9])
p_array = p_knob_array * P_tilde * R_tilde / (gamma0_tilde * gammar_tilde)
gr_dim = 0
g_dim = 0

path = r'/scratch/konstantinos'
final_save = r'/home6/konstantinos'

def names_subfolders(sigma_array, p_array):
    save_folder_spatial = path+ os.sep + 'spatial' + '_' + 'ns' + str(ns_tilde) + '_' + 'g' + str(g_dim)+ '_' + 'gr' + str(gr_dim) + '_' + 'gamma' + str(gamma0_tilde)
    save_folder_temporal = path+ os.sep + 'temporal' + '_' + 'ns' + str(ns_tilde) + '_' + 'g' + str(g_dim)+ '_' + 'gr' + str(gr_dim) + '_' + 'gamma' + str(gamma0_tilde)
    if save_folder_spatial in os.listdir(path):
        print(f'Folder "{save_folder_spatial}" exists.')
    else:
        os.mkdir(save_folder_spatial)
        print(f'Folder "{save_folder_spatial}" succesfully created.')
    if save_folder_temporal in os.listdir(path):
        print(f'Folder "{save_folder_temporal}" exists.')
    else:
        os.mkdir(save_folder_temporal)
        print(f'Folder "{save_folder_temporal}" succesfully created.')
    subfolders_spatial = {}
    subfolders_temporal = {}
    for sigma in sigma_array:
        for p in p_array:
            subfolders_spatial[str(p), str(sigma)] = save_folder_spatial + os.sep + 'sigma' + str(sigma) + '_' + 'p' + str(p) + '_' + 'om' + str(int(om_knob_array[0]))
            subfolders_temporal[str(p), str(sigma)] = save_folder_temporal + os.sep + 'sigma' + str(sigma) + '_' + 'p' + str(p) + '_' + 'om' + str(int(om_knob_array[0]))
    return subfolders_spatial, subfolders_temporal

def g1(i_batch, sigma, p, om, g_dim, gr_dim):
        g1_x_batch = np.zeros(N//2, dtype = complex)
        d1_x_batch = np.zeros(N//2, dtype = complex)
        g1_t_batch = np.zeros(len(t), dtype = complex)
        d1_t_batch = np.zeros(len(t), dtype = complex)
        for i_n in range(n_internal):
            gpe = model(sigma, p, om, g_dim, gr_dim)
            g1_x_run, d1_x_run, g1_t_run, d1_t_run = gpe.time_evolution()
            g1_x_batch += g1_x_run / n_internal
            d1_x_batch += d1_x_run / n_internal
            g1_t_run += g1_t_run / n_internal
            d1_t_run += d1_t_run / n_internal
        np.save(subfolders_spatial[str(p), str(sigma)] + os.sep + 'numerator' + '_' +'core' + str(i_batch + 1) + '.npy', g1_x_batch)
        np.save(subfolders_spatial[str(p), str(sigma)] + os.sep + 'denominator' + '_' + 'core' + str(i_batch + 1) + '.npy', d1_x_batch)
        np.save(subfolders_temporal[str(p), str(sigma)] + os.sep + 'numerator' + '_' +'core' + str(i_batch + 1) + '.npy', g1_t_batch)
        np.save(subfolders_temporal[str(p), str(sigma)] + os.sep + 'denominator' + '_' + 'core' + str(i_batch + 1) + '.npy', d1_t_batch)

subfolders_spatial, subfolders_temporal = names_subfolders(sigma_array, p_array)

def call_avg():
    for sigma in sigma_array:
        for p in p_array:
            numerator_spatial = np.zeros((N//2), dtype = complex)
            denominator_spatial = np.zeros((N//2), dtype = complex)
            numerator_temporal = np.zeros(len(t), dtype = complex)
            denominator_temporal = np.zeros(len(t), dtype = complex)
            if subfolders_spatial[str(p), str(sigma)] in os.listdir(path):
                print(f'Subfolder "{subfolders_spatial[str(p), str(sigma)]}" exists.')
            else:
                os.mkdir(subfolders_spatial[str(p), str(sigma)])
                print(f'Subfolder "{subfolders_temporal[str(p), str(sigma)]}" succesfully created.')
            if subfolders_temporal[str(p), str(sigma)] in os.listdir(path):
                print(f'Subfolder "{subfolders_temporal[str(p), str(sigma)]}" exists.')
            else:
                os.mkdir(subfolders_temporal[str(p), str(sigma)])
                print(f'Subfolder "{subfolders_temporal[str(p), str(sigma)]}" succesfully created.')
            print('Starting simulations for sigma = %.2f, p = %.1i' % (sigma, p))
            parallel_map(g1, range(n_batch), task_kwargs=dict(sigma=sigma, p=p, om=om_knob_array[0], g_dim=g_dim, gr_dim=gr_dim))
            for file in os.listdir(subfolders_spatial[str(p), str(sigma)]):
                if 'numerator' in file:
                    numerator_spatial += np.load(subfolders_spatial[str(p), str(sigma)] + os.sep + file) / n_batch
                elif 'denominator' in file:
                    denominator_spatial += np.load(subfolders_spatial[str(p), str(sigma)] + os.sep + file) / n_batch
            for file in os.listdir(subfolders_temporal[str(p), str(sigma)]):
                if 'numerator' in file:
                    numerator_temporal += np.load(subfolders_temporal[str(p), str(sigma)] + os.sep + file) / n_batch
                elif 'denominator' in file:
                    denominator_temporal += np.load(subfolders_temporal[str(p), str(sigma)] + os.sep + file) / n_batch
            np.save(final_save + os.sep + 'spatial' + 
                '_' + 'sigma' + str(sigma) + 
                '_' + 'p' + str(p) + 
                '_' + 'om' + str(int(om_knob_array[0])) + 
                '_' + 'g' + str(g_dim) + '_' + 'gr' + str(gr_dim) + '_' + 'gamma' + str(gamma0_tilde) +'.npy', np.abs(numerator_spatial)/np.sqrt(denominator_spatial))
            np.save(final_save + os.sep + 'temporal' + 
                '_' + 'sigma' + str(sigma) + 
                '_' + 'p' + str(p) + 
                '_' + 'om' + str(int(om_knob_array[0])) + 
                '_' + 'g' + str(g_dim) + '_' + 'gr' + str(gr_dim) + '_' + 'gamma' + str(gamma0_tilde) +'.npy', np.abs(numerator_temporal)/np.sqrt(denominator_temporal))

call_avg()