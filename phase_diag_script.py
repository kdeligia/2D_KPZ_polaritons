#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 10:40:36 2021

@author: delis
"""

c = 3E2 #μm/ps
hbar = 6.582119569 * 1E2 # μeV ps

import os
from scipy.fftpack import fft2, ifft2
import numpy as np
import external as ext
from qutip import *

hatt = 1 # ps
hatx = 1 # μm
hatpsi = 1/hatx # μm^-1
hatrho = 1/hatx**2 # μm^-2
hatepsilon = hbar/hatt # μeV
melectron = 0.510998950 * 1E12 / c**2 # μeV/(μm^2/ps^2)

m_tilde = -6.2e-5
gamma0_tilde = 0.22
gammar_tilde = 0.1 * gamma0_tilde
gamma2_tilde = 0.04
P_tilde = 39.6 * 5
R_tilde = gammar_tilde / 500
p = P_tilde * R_tilde / (gamma0_tilde * gammar_tilde)

ns_tilde = gammar_tilde / R_tilde
n0_tilde = ns_tilde * (p - 1)
nres_tilde = P_tilde / (gammar_tilde * (1 + n0_tilde/ns_tilde))

N = 2**7
L_tilde = 2**7
dx_tilde = L_tilde / N
dkx_tilde = 2 * np.pi / (N * dx_tilde)

def dimensional_units():
    L_dim = L_tilde * hatx                                                      # result in μm
    P_dim = P_tilde * (1/(hatx**2 * hatt))                                      # result in μm^-2 ps^-1
    R_dim = R_tilde * (hatx**2/hatt)                                            # result in μm^2 ps^-1
    gamma0_dim = gamma0_tilde * (1/hatt)                                        # result in ps^-1
    gammar_dim = gammar_tilde * (1/hatt)                                        # result in ps^-1
    gamma2_dim = gamma2_tilde * (hatx**2 / hatt)                                # result in μm^2 ps^-1
    ns_dim = ns_tilde * hatrho                                                  # result in μm^-2
    m_dim = m_tilde * melectron
    n0_dim = n0_tilde * hatrho                                                  # result in μm^-2
    nr_dim = nres_tilde * hatrho                                                # result in μm^-2
    return L_dim, P_dim, R_dim, gamma0_dim, gammar_dim, gamma2_dim, ns_dim, m_dim, n0_dim, nr_dim

def arrays():
    x_0 = - N * dx_tilde / 2
    kx0 = - np.pi / dx_tilde
    x = x_0 + dx_tilde * np.arange(N)
    kx = kx0 + dkx_tilde * np.arange(N)
    return x, kx

L_dim, P_dim, R_dim, gamma0_dim, gammar_dim, gamma2_dim, ns_dim, m_dim, n0_dim, nr_dim = dimensional_units()
x, kx =  arrays()
X, Y = np.meshgrid(x, x)
KX, KY = np.meshgrid(kx, kx)

time_steps = 80000
dt_tilde = 1.5e-2
every = 1000
i1 = 0
i2 = time_steps
lengthwindow = i2-i1
t = ext.time(dt_tilde, time_steps, i1, i2, every)
xi = hbar / np.sqrt(2 * abs(m_dim) * n0_dim)

class model:
    def __init__(self, g, gr, psi_x=0):
        self.g = g 
        self.gr = gr
        self.sigma = gamma0_tilde * (p + 1) / (4 * dx_tilde**2)
        self.psi_x = psi_x
        self.psi_x = np.full((N, N), 0.05)
        self.psi_x /= hatpsi
        self.psi_mod_k = fft2(self.psi_mod_x)
        self.Kc = hbar**2 / (2 * m_dim * hatepsilon * hatx**2)
        self.Kd = gamma2_tilde / 2

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
        return (self.psi_x * np.conjugate(self.psi_x)).real - 1/(2 * dx_tilde**2)

    def prefactor_x(self):
        self.uc_tilde = self.g * (self.n() + 2 * (self.gr / self.g) * (P_tilde / gammar_tilde) * (1 / (1 + self.n() / ns_tilde)))
        self.I_tilde = (gamma0_tilde / 2) * (p * (1 / (1 + self.n() / ns_tilde)) - 1)
        return np.exp(-1j * 0.5 * dt_tilde * (self.uc_tilde + 1j * self.I_tilde))

    def prefactor_k(self):
        return np.exp(-1j * dt_tilde * ((KX ** 2 + KY ** 2)*(self.Kc - 1j * self.Kd)))

# =============================================================================
# Time evolution
# =============================================================================
    def vortices(self):
        count_v = 0
        count_av = 0
        theta = np.angle(self.psi_x)
        grad = np.gradient(theta, dx_tilde)
        for i in range(1, N-1):
            for j in range(1, N-1):
                loop = (2*dx_tilde*(grad[0][j+1, i+1] - grad[1][j+1, i+1]) +
                        2*dx_tilde*(grad[0][j+1, i-1] + grad[1][j+1, i-1]) +
                        2*dx_tilde*(-grad[0][j-1, i-1] + grad[1][j-1, i-1]) +
                        2*dx_tilde*(-grad[0][j-1, i+1] - grad[1][j-1, i+1]) +
                        2*dx_tilde*(grad[0][j+1, i] + grad[1][j, i-1] - grad[0][j-1, i] - grad[1][j, i+1]))
                if loop >= 2 * np.pi:
                    count_v += 1
                elif loop <= - 2 * np.pi:
                    count_av +=1
        total = count_v + count_av
        return total

    def time_evolution(self):
        np.random.seed()
        n_sum = np.zeros(len(t))
        v = np.zeros(len(t))
        for i in range(time_steps+1):
            self.psi_x *= self.prefactor_x()
            self.psi_mod_k = fft2(self.psi_mod_x)
            self.psi_k *= self.prefactor_k()
            self.psi_mod_x = ifft2(self.psi_mod_k)
            self.psi_x *= self.prefactor_x()
            self.psi_x += np.sqrt(dt_tilde) * np.sqrt(self.sigma) * (np.random.normal(0, 1, (N,N)) + 1j*np.random.normal(0, 1, (N,N)))
            if i>=i1 and i<=i2 and i%every==0:
                n = self.n()
                n_sum[(i-i1)//every] = np.mean(n)
                v[(i-i1)//every] = self.vortices()
        return  n_sum, v

# =============================================================================
# Phase diagram
# =============================================================================
name_remote = r'/scratch/konstantinos/'
save_remote = r'/home6/konstantinos/'

parallel_tasks = 240
n_batch = 80
n_internal = parallel_tasks//n_batch
qutip.settings.num_cpus = n_batch

mu_cond = 5
g_tilde = (mu_cond / hatepsilon) * (1 / n0_tilde)
mu_res_array = np.array([100, 200, 500, 1000, 2000, 5000, 7000, 10000])

for mu_res in mu_res_array:
    print('Starting for mu_res = ', mu_res)
    os.mkdir(name_remote+'phase_diagram_'+str(mu_res)+'_'+str(mu_cond))
    gr_tilde = (mu_res / hatepsilon) * (1 / (2 * nres_tilde))

    def parallel_phase_diagram(i_batch):
        quantities_batch = np.zeros(2)
        for i_n in range(n_internal):
            gpe = model(g_tilde, gr_tilde)
            nsum, v = gpe.time_evolution()
            quantities_batch += np.mean(nsum[53:]) / n_internal, (np.mean(v[53:])/N**2) / n_internal
            print('PHASE DIAGRAM Core', i_batch, 'completed realisation number', i_n+1)
        np.savetxt(name_remote+'phase_diagram_'+str(mu_res)+'_'+str(mu_cond)+os.sep+'file_core'+str(i_batch+1)+'.dat', quantities_batch)
    parallel_map(parallel_phase_diagram, range(n_batch))

for mu_res in mu_res_array:
    result = np.zeros(2)
    for file in os.listdir(name_remote+'phase_diagram_'+str(mu_res)+'_'+str(mu_cond)):
        if '.dat' in file:
            item = np.loadtxt(name_remote+'phase_diagram_'+str(mu_res)+'_'+str(mu_cond)+os.sep+file)
            result += item / n_batch
    np.savetxt(save_remote+'phase_diagram_'+str(mu_res)+'_'+str(mu_cond)+'_result.dat', result)

# =============================================================================
#  Treatment
# =============================================================================

final = np.zeros((2, len(mu_res_array)))
for i in range(len(mu_res_array)):
    final[:, i] = np.loadtxt(save_remote+'phase_diagram_'+str(mu_res_array[i])+'_' + str(mu_cond)+'_result.dat')
np.savetxt(save_remote+'phase_diagram_'+str(mu_cond)+'_FINAL.dat', final)