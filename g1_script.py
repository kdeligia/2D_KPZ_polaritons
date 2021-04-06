#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 15:09:50 2020

@author: delis
"""

c = 3E2 #μm/ps
hbar = 6.582119569 * 1E2 # μeV ps

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

m_tilde = 2e-5
m_dim = m_tilde * melectron
gamma0_tilde = 0.2
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

time_steps = 100000
dt_tilde = 1e-2

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
        #print('sigma = %.2f, p = %.3f, tilde g = %.1f, tilde gr = %.3f, TWR = %.3f' % (self.sigma, self.p, g_dim, gr_dim, self.g_tilde / (gamma0_tilde * dx_tilde**2)))

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
        g1_x = np.zeros(int(N/2), dtype = complex)
        d1_x = np.zeros(int(N/2), dtype = complex)
        for i in range(time_steps+1):
            #self.sigma = gamma0_tilde * (self.p / (1 + self.n() / ns_tilde) + 1) / 4
            self.psi_x *= self.prefactor_x()
            self.psi_mod_k = fft2(self.psi_mod_x)
            self.psi_k *= self.prefactor_k()
            self.psi_mod_x = ifft2(self.psi_mod_k)
            self.psi_x *= self.prefactor_x()
            self.psi_x += np.sqrt(dt_tilde) * np.sqrt(self.sigma / dx_tilde ** 2) * (np.random.normal(0, 1, (N,N)) + 1j * np.random.normal(0, 1, (N,N))) / self.damp
        for i in range(N):
            g1_x += np.conjugate(self.psi_x[i, int(N/2)]) * self.psi_x[i, int(N/2):] / N
            d1_x += np.conjugate(self.psi_x[i, int(N/2):]) * self.psi_x[i, int(N/2):] / N
        return g1_x, d1_x

# =============================================================================
# Parallel tests
# =============================================================================
from qutip import *
parallel_tasks = 320
n_batch = 64
n_internal = parallel_tasks//n_batch
qutip.settings.num_cpus = n_batch

sigma_array = np.array([1e-2, 2e-2, 4e-2])
p_knob_array = np.array([1.8])
om_knob_array = np.array([10])
p_array = p_knob_array * P_tilde * R_tilde / (gamma0_tilde * gammar_tilde)
gr_dim = 0
g_dim = 0

path_init = r'/scratch/konstantinos'
save_folder = path_init + os.sep + 'x_cor' + '_' + 'g' + str(g_dim)+ '_' + 'gr' + str(gr_dim) + '_' + 'gamma' + str(gamma0_tilde)
os.mkdir(save_folder)

for sigma in sigma_array:
    print('Starting simulations for sigma = %.3f' % sigma)
    save_subfolder = save_folder + os.sep + 'p' + str(np.round(p_array[0], 3)) + '_' + 'om' + str(int(om_knob_array[0])) + '_' + 'sigma' + str(sigma)
    os.mkdir(save_subfolder)

    def x_g1_parallel(i_batch):
        correlation_batch = np.zeros((2, int(N/2)), dtype=complex)
        for i_n in range(n_internal):
            gpe = model(sigma, p_knob_array[0], om_knob_array[0], g_dim, gr_dim)
            g1_x_run, d1_x_run = gpe.time_evolution()
            correlation_batch += np.vstack((g1_x_run, d1_x_run)) / n_internal
            print('CORRELATION Core', i_batch, 'completed realisation number', i_n + 1)
        np.save(save_subfolder + os.sep + 'x_g1' + '_' + 'core' + str(i_batch + 1) + '.npy', correlation_batch)
    parallel_map(x_g1_parallel, range(n_batch))

for sigma in sigma_array:
    save_subfolder = save_folder + os.sep + 'p' + str(np.round(p_array[0], 3)) + '_' + 'om' + str(int(om_knob_array[0])) + '_' + 'sigma' + str(sigma)
    result = np.zeros((2, int(N/2)), dtype = complex)
    for file in os.listdir(save_subfolder):
        if '.npy' in file:
            item = np.load(save_subfolder + os.sep + file)
            result += item / n_batch
    np.save(r'/home6/konstantinos' + os.sep + 'final_x_g1' + 
            '_' + 'p' + str(np.round(p_array[0], 3)) + 
            '_' + 'om' + str(int(om_knob_array[0])) + 
            '_' + 'sigma' + str(sigma) + 
            '_' + 'g' + str(g_dim) + '_' + 'gr' + str(gr_dim) + '_' + 'gamma' + str(gamma0_tilde) +'.npy', result)