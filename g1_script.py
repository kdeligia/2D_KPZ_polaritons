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

m_tilde = 3.8e-5
m_dim = m_tilde * melectron
gamma0_tilde = 0.22
gammar_tilde = 0.1 * gamma0_tilde
P_tilde = 50 * gamma0_tilde 
R_tilde = gammar_tilde / 50
gamma2_tilde = 0.02
ns_tilde = gammar_tilde / R_tilde
Kc = hbar**2 / (2 * m_dim * hatepsilon * hatx**2)
Kd = gamma2_tilde / 2

# =============================================================================
# 
# =============================================================================

N = 2**7
L_tilde = 2**7
dx_tilde = L_tilde / N
dkx_tilde = 2 * np.pi / (N * dx_tilde)

'''
def dimensional_units():
    L_dim = L_tilde * hatx                                                      # result in μm
    P_dim = P_tilde * (1/(hatx**2 * hatt))                                      # result in μm^-2 ps^-1
    R_dim = R_tilde * (hatx**2/hatt)                                            # result in μm^2 ps^-1
    gamma0_dim = gamma0_tilde * (1/hatt)                                        # result in ps^-1
    gammar_dim = gammar_tilde * (1/hatt)                                        # result in ps^-1
    gamma2_dim = gamma2_tilde * (hatx**2 / hatt)                                # result in μm^2 ps^-1
    ns_dim = ns_tilde * hatrho                                                  # result in μm^-2
    n0_dim = n0_tilde * hatrho                                                  # result in μm^-2
    nr_dim = nres_tilde * hatrho                                                # result in μm^-2
    return L_dim, P_dim, R_dim, gamma0_dim, gammar_dim, gamma2_dim, ns_dim, n0_dim, nr_dim
'''

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
dt_tilde = 4.5e-2

class model:
    def __init__(self, p, g_dim, gr_dim, psi_x=0):
        m_dim = m_tilde * melectron
        self.p = p
        self.g_tilde = g_dim * hatrho / hatepsilon
        self.gr_tilde = gr_dim * hatrho / hatepsilon
        self.psi_x = psi_x
        self.psi_x = np.full((N, N), np.sqrt(1 / (2 * dx_tilde**2)) + 0.01)
        self.psi_x /= hatpsi
        self.psi_mod_k = fft2(self.psi_mod_x)
        self.Kc = hbar**2 / (2 * m_dim * hatepsilon * hatx**2)
        self.Kd = gamma2_tilde / 2
        if self.g_tilde == 0:
            self.uc = 0
        else:
            self.uc =  self.g_tilde * (1 - 2 * self.p * (self.gr_tilde / self.g_tilde) * (gamma0_tilde / gammar_tilde))

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
        return (self.psi_x * np.conjugate(self.psi_x)).real - 1/(2*dx_tilde**2)

    def prefactor_x(self):
        if self.g_tilde == 0:
            self.uc_tilde = 0
        else:
            self.uc_tilde = self.g_tilde * (self.n() + 2 * self.p * (self.gr_tilde / self.g_tilde) * (gamma0_tilde / R_tilde) * (1 / (1 + self.n() / ns_tilde)))
        self.I_tilde = (gamma0_tilde / 2) * (self.p * (1 / (1 + self.n() / ns_tilde)) - 1)
        return np.exp(-1j * 0.5 * dt_tilde * (self.uc_tilde + 1j * self.I_tilde))

    def prefactor_k(self):
        return np.exp(-1j * dt_tilde * (KX ** 2 + KY ** 2) * (self.Kc - 1j * self.Kd))

# =============================================================================
# Time evolution
# =============================================================================
    def time_evolution(self):
        np.random.seed()
        g1_x = np.zeros(int(N/2), dtype = complex)
        d1_x = np.zeros(int(N/2))
        for i in range(time_steps+1):
            self.sigma = gamma0_tilde * (self.p / (1 + self.n() / ns_tilde) + 1) / (4 * dx_tilde**2)
            self.psi_x *= self.prefactor_x()
            self.psi_mod_k = fft2(self.psi_mod_x)
            self.psi_k *= self.prefactor_k()
            self.psi_mod_x = ifft2(self.psi_mod_k)
            self.psi_x *= self.prefactor_x()
            self.psi_x += np.sqrt(dt_tilde) * np.sqrt(self.sigma) * (np.random.normal(0, 1, (N,N)) + 1j*np.random.normal(0, 1, (N,N)))
        for i in range(N):
            g1_x += np.conjugate(self.psi_x[i, int(N/2)]) * self.psi_x[i, int(N/2):] / N
            d1_x += self.n()[i, int(N/2):] / N
        g1_x[0] -= 1/(2*dx_tilde**2)
        return g1_x, d1_x

# =============================================================================
# Parallel tests
# =============================================================================
from qutip import *
parallel_tasks = 256
n_batch = 64
n_internal = parallel_tasks//n_batch
qutip.settings.num_cpus = n_batch

knob_array = np.array([1.2, 1.3, 1.4, 1.5, 1.6, 1.8, 1.9, 2])
p_array = knob_array * P_tilde * R_tilde / (gamma0_tilde * gammar_tilde)
gr_dim = 0
g_dim = 0

path_init = r'/Users/delis/Desktop'
#path_init = r'/scratch/konstantinos'
save_folder = path_init + os.sep + 'correlations' + '_' + 'Kd' + str(gamma2_tilde/2) + '_' + 'g' + str(g_dim) + '_' + 'gr' + str(gr_dim)
os.mkdir(save_folder)

print('Saturation in μm^-2 %.2f' % (ns_tilde * hatrho))
print('Kd = %.3f' % (gamma2_tilde/2))
print('Kc = %.4f' % Kc)
print('g = %.1f' % g_dim)
print('gr = %.1f' % gr_dim)

for p in p_array:
    print('Starting simulations for p = %.3f' % p)
    save_subfolder = save_folder + os.sep + 'pump' + '_' + str(np.round(p, 3))
    os.mkdir(save_subfolder)

    def x_g1_parallel(i_batch):
        correlation_batch = np.zeros((2, int(N/2)), dtype=complex)
        for i_n in range(n_internal):
            gpe = model(p, g_dim, gr_dim)
            g1_x_run, d1_x_run = gpe.time_evolution()
            correlation_batch += np.vstack((g1_x_run, d1_x_run)) / n_internal
            print('CORRELATION Core', i_batch, 'completed realisation number', i_n + 1)
        np.save(save_subfolder + os.sep + 'x_g1' + '_' + 'core' + str(i_batch + 1) + '.npy', correlation_batch)
    parallel_map(x_g1_parallel, range(n_batch))

for p in p_array:
    save_subfolder = save_folder + os.sep + 'pump' + '_' + str(np.round(p, 3))
    result = np.zeros((2, int(N/2)), dtype = complex)
    for file in os.listdir(save_subfolder):
        if '.npy' in file:
            item = np.load(save_subfolder + os.sep + file)
            result += item / n_batch
    np.save(r'/home6/konstantinos' + os.sep + 'final_x_g1' + str(np.round(p, 3)) + '_' + str(g_dim) + '_' + str(gr_dim) + '.npy', result)