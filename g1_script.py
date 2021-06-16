#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 15:09:50 2020

@author: delis
"""

c = 3e2 #μm/ps
hbar = 6.582119569 * 1e2 # μeV ps

import os
import numpy as np
import external as ext
from scipy.fftpack import fft2, ifft2

hatt = 1 # ps
hatx = 1 # μm
hatpsi = 1/hatx # μm^-1
hatrho = 1/hatx**2 # μm^-2
hatepsilon = hbar/hatt # μeV
melectron = 0.510998950 * 1e12 / c**2 # μeV/(μm^2/ps^2)

m_tilde = 5e-5
m_dim = m_tilde * melectron
Kc = hbar ** 2 / (2 * m_dim * hatepsilon * hatx**2)

# =============================================================================
# 
# =============================================================================
N = 2 ** 7
dx_tilde = 0.5

N_steps = 2000000
dt_tilde = 5e-2
every = 10000
i1 = 10000
i2 = N_steps
lengthwindow = i2-i1
t = ext.time(dt_tilde, N_steps, i1, i2, every)

x, y = ext.space_momentum(N, dx_tilde)
isotropic_indices = ext.get_indices(x)
np.savetxt('/home6/konstantinos/t_test.dat', t)

'''
h = np.zeros((N, N))
for rad in range(N//2):
    mysum = 0
    indices = isotropic_indices.get('r = ' + str(rad))
    for i in range(len(indices)):
        h[indices[i][0], indices[i][1]] = rad
        mysum += h[indices[i][0], indices[i][1]]/len(indices)
    print(mysum)
print(h)
'''

kx = (2 * np.pi) / dx_tilde * np.fft.fftfreq(N, d = 1)
ky = (2 * np.pi) / dx_tilde * np.fft.fftfreq(N, d = 1)

class model:
    def __init__(self, p, sigma, gamma0, gamma2, g, gr, ns):
        self.KX, self.KY = np.meshgrid(kx, ky, sparse=True)
        self.gamma2_tilde = gamma2  * hatt / hatx **2
        self.gamma0_tilde = gamma0 * hatt
        self.gammar_tilde = 0.1 * self.gamma0_tilde
        
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
        return np.exp(-1j * dt * (self.KX ** 2 + self.KY ** 2) * (Kc - 1j * self.Kd))

# =============================================================================
# Time evolution
# =============================================================================
    def time_evolution(self):
        np.random.seed()
        center_indices = isotropic_indices.get('r = ' + str(0))
        psi_correlation_x = np.zeros((len(t), N//2), dtype = complex)
        psi_correlation_t = np.zeros(len(t), dtype = complex)
        n_avg = np.zeros((len(t), N//2), dtype = complex)
        for i in range(N_steps):
            self.psi_x *= self.exp_x(0.5 * dt_tilde, self.n(self.psi_x))
            psi_k = fft2(self.psi_x)
            psi_k *= self.exp_k(dt_tilde)
            self.psi_x = ifft2(psi_k)
            self.psi_x *= self.exp_x(0.5 * dt_tilde, self.n(self.psi_x))
            self.psi_x += np.sqrt(dt_tilde) * np.sqrt(self.sigma / dx_tilde ** 2) * (np.random.normal(0, 1, (N,N)) + 1j * np.random.normal(0, 1, (N,N)))
            if i >= i1 and i <= i2 and i % every == 0:
                time_array_index = (i-i1)//every
                if i == i1:
                    psi_x0t0 = self.psi_x[center_indices[0][0], center_indices[0][1]]
                #psi_correlation[time_array_index] = ext.isotropic_avg('psi correlation', self.psi_x, psi_x0t0, **isotropic_indices)
                psi_x0t = self.psi_x[center_indices[0][0], center_indices[0][1]]
                psi_correlation_x[time_array_index] = ext.isotropic_avg('psi correlation', self.psi_x, psi_x0t, **isotropic_indices)
                psi_correlation_t[time_array_index] = np.conjugate(psi_x0t0) * psi_x0t
                n_avg[time_array_index] = ext.isotropic_avg('density average', self.n(self.psi_x), None, **isotropic_indices)
        return psi_correlation_x, psi_correlation_t, n_avg

# =============================================================================
# Parallel tests
# =============================================================================
from qutip import *
parallel_tasks = 128
n_batch = 128
n_internal = parallel_tasks//n_batch
qutip.settings.num_cpus = n_batch

'''
p_array = np.array([1.8, 1.4])
gamma2_array = np.array([0.05, 0.1, 0.2, 0.4, 0.6, 0.8])
gamma0_array = np.array([0.2])
sigma_array = np.array([0.28, 0.24])
g_array = np.array([0, 0.5, 2])
'''

p_array = np.array([1.8])
gamma2_array = np.array([0.05])
gamma0_array = np.array([0.2])
sigma_array = np.array([0.28])
g_array = np.array([0])
gr = 0
ns = 50.

def g1(i_batch, p, sigma, gamma0, gamma2, g, path):
    correlation_x_batch = np.zeros((len(t), N//2), dtype = complex)
    correlation_t_batch = np.zeros(len(t), dtype = complex)
    avg_dens_batch = np.zeros((len(t), N//2), dtype = complex)
    for i_n in range(n_internal):
        gpe = model(p, sigma, gamma0, gamma2, g, gr = gr, ns = ns)
        correlation_x_run, correlation_t_run, avg_dens_run = gpe.time_evolution()
        correlation_x_batch += correlation_x_run / n_internal
        correlation_t_batch += correlation_t_run / n_internal
        avg_dens_batch += avg_dens_run / n_internal
    np.save(path + os.sep + 'space correlation' + '_' +'core' + str(i_batch + 1) + '.npy', correlation_x_batch)
    np.save(path + os.sep + 'time correlation' + '_' +'core' + str(i_batch + 1) + '.npy', correlation_t_batch)
    np.save(path + os.sep + 'avg_density' + '_' + 'core' + str(i_batch + 1) + '.npy', avg_dens_batch)
    return None

init = r'/scratch/konstantinos' + os.sep + 'N' + str(N) + '_' + 'ns' + str(int(ns)) + 'TEST'
os.mkdir(init)
ids = ext.ids(False, init, p_array, sigma_array, gamma0_array, gamma2_array, g_array, ns)
def call_avg(loc):
    for p in p_array:
        sigma = sigma_array[np.where(p_array == p)][0]
        for g in g_array:
            for gamma2 in gamma2_array:
                print('--- Secondary simulation parameters: p = %.1f, g = %.2f, ns = %.i' % (p, g, ns))
                id_string = ids.get(('p=' + str(p), 'sigma=' + str(sigma), 'gamma0=' + str(gamma0_array[0]), 'gamma2=' + str(gamma2), 'g=' + str(g)))
                save_folder = init + os.sep + id_string
                try:
                    os.mkdir(save_folder)
                except FileExistsError:
                    continue
                correlation_x = np.zeros((len(t), N//2), dtype = complex)
                correlation_t = np.zeros(len(t), dtype = complex)
                avg_n = np.zeros((len(t), N//2), dtype = complex)
                print('--- Kinetic terms: Kc = %.5f, Kd = %.5f' % (Kc, gamma2/2))
                print('--- Primary simulation parameters: sigma = %.2f, gamma0 = %.2f, gamma2 = %.2f' % (sigma, gamma0_array[0], gamma2))
                parallel_map(g1, range(n_batch), task_kwargs=dict(p = p, sigma = sigma, gamma0 = gamma0_array[0], gamma2 = gamma2, g = g, path = save_folder))
                for file in os.listdir(save_folder):
                    if 'space correlation' in file:
                        correlation_x += np.load(save_folder + os.sep + file) / n_batch
                    if 'time correlation' in file:
                        correlation_t += np.load(save_folder + os.sep + file) / n_batch
                    elif 'avg_density' in file:
                        avg_n += np.load(save_folder + os.sep + file) / n_batch
                #np.save(loc + os.sep + id_string + '__' + 'TEST' + '.npy', np.abs(correlation).real / np.sqrt(avg_dens[0, 0].real * avg_dens.real))
                for i in range(len(t)):
                    avg_n[i] *= avg_n[i, 0]
                np.save(loc + os.sep + id_string + '__' + 'TEST_SPATIAL_EVOL' + '.npy', np.abs(correlation_x).real / np.sqrt(avg_n.real))
                np.save(loc + os.sep + id_string + '__' + 'TEST_TEMP' + '.npy', np.abs(correlation_t).real / np.sqrt(avg_n[:, 0]))
        return None

final_save_remote = r'/home6/konstantinos'
call_avg(final_save_remote)