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

# =============================================================================
# 
# =============================================================================
N = 2 ** 7
dx_tilde = 0.5

sampling_begin = 10000
sampling_step = 1000
N_steps = 1000000 + sampling_begin + sampling_step
dt_tilde = 1e-2

sampling_end = N_steps
sampling_window = sampling_end - sampling_begin
t = ext.time(dt_tilde, N_steps, sampling_begin, sampling_end, sampling_step)

x, y = ext.space_momentum(N, dx_tilde)
isotropic_indices = ext.get_indices(x)

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
    def __init__(self, p, sigma, gamma0, gamma2, g, gr, ns, m):
        self.KX, self.KY = np.meshgrid(kx, ky, sparse=True)
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
    def time_evolution(self):
        np.random.seed()
        center_indices = isotropic_indices.get('r = ' + str(0))
        psipsi_evol = np.zeros((len(t), N//2), dtype = complex)
        sqrt_nn_evol = np.zeros((len(t), N//2), dtype = complex)
        '''
        psipsi_full = np.zeros((len(t), N//2), dtype = complex)
        nn_full = np.zeros((len(t), N//2), dtype = complex)
        '''
        psipsi_t = np.zeros(len(t), dtype = complex)
        sqrt_nn_t = np.zeros(len(t), dtype = complex)
        n_avg = np.zeros((len(t), N//2), dtype = complex)
        for i in range(N_steps):
            self.psi_x *= self.exp_x(0.5 * dt_tilde, self.n(self.psi_x))
            psi_k = fft2(self.psi_x)
            psi_k *= self.exp_k(dt_tilde)
            self.psi_x = ifft2(psi_k)
            self.psi_x *= self.exp_x(0.5 * dt_tilde, self.n(self.psi_x))
            self.psi_x += np.sqrt(dt_tilde) * np.sqrt(self.sigma / dx_tilde ** 2) * (np.random.normal(0, 1, (N,N)) + 1j * np.random.normal(0, 1, (N,N)))
            if i >= sampling_begin and i <= sampling_end and i % sampling_step == 0:
                time_array_index = (i - sampling_begin) // sampling_step
                if i == sampling_begin:
                    psi_x0t0 = self.psi_x[center_indices[0][0], center_indices[0][1]]
                    n_x0t0 = psi_x0t0 * np.conjugate(psi_x0t0)
                psi_x0t = self.psi_x[center_indices[0][0], center_indices[0][1]]
                n_x0t = psi_x0t * np.conjugate(psi_x0t)
                psipsi_evol[time_array_index] = ext.isotropic_avg('correlation', self.psi_x, np.conjugate(psi_x0t), **isotropic_indices)
                sqrt_nn_evol[time_array_index] = ext.isotropic_avg('correlation', np.sqrt(self.n(self.psi_x)), np.sqrt(n_x0t), **isotropic_indices)
                psipsi_t[time_array_index] = np.conjugate(psi_x0t0) * psi_x0t
                sqrt_nn_t[time_array_index] = np.sqrt(n_x0t0 * n_x0t)
                '''
                psipsi_full[time_array_index] = ext.isotropic_avg('correlation', self.psi_x, np.conjugate(psi_x0t0), **isotropic_indices)
                nn_full[time_array_index] = ext.isotropic_avg('correlation', self.n(self.psi_x) ** (1/2), n_x0t0 ** (1/2), **isotropic_indices)
                '''
                n_avg[time_array_index] = ext.isotropic_avg('density average', self.n(self.psi_x), None, **isotropic_indices)
        return psipsi_evol, sqrt_nn_evol, psipsi_t, sqrt_nn_t, n_avg

# =============================================================================
# Parallel tests
# =============================================================================
from qutip import *
parallel_tasks = 1024
n_batch = 128
n_internal = parallel_tasks//n_batch
qutip.settings.num_cpus = n_batch

p_array = np.array([2])
gamma2_array = np.array([0.1])
gamma0_array = np.array([8])
sigma_array = np.array([0.01])
g_array = np.array([0])
m_array = np.array([1e-4])
gr = 0
ns = 1.

print('------- Kinetic term (real) ------- : %.5f' % (hbar ** 2 / (2 * m_array[0] * melectron * hatepsilon * hatx ** 2)))
init = r'/scratch/konstantinos' + os.sep + 'N' + str(N) + '_' + 'ns' + str(int(ns)) + '_' + 'm' + str(m_array[0])
if os.path.isdir(init) == False:
    os.mkdir(init)
ids = ext.ids(p_array, sigma_array, gamma0_array, gamma2_array, g_array)

def g1(i_batch, p, sigma, gamma0, gamma2, g, path):
    psipsi_evol_batch = np.zeros((len(t), N//2), dtype = complex)
    sqrt_nn_evol_batch = np.zeros((len(t), N//2), dtype = complex)
    psipsi_t_batch = np.zeros(len(t), dtype = complex)
    sqrt_nn_t_batch = np.zeros(len(t), dtype = complex)
    n_avg_batch = np.zeros((len(t), N//2), dtype = complex)
    for i_n in range(n_internal):
        gpe = model(p, sigma, gamma0, gamma2, g, gr = gr, ns = ns, m = m_array[0])
        psipsi_evol, sqrt_nn_evol, psipsi_t, sqrt_nn_t, n_avg = gpe.time_evolution()
        psipsi_evol_batch += psipsi_evol / n_internal
        sqrt_nn_evol_batch += sqrt_nn_evol / n_internal
        psipsi_t_batch += psipsi_t / n_internal
        sqrt_nn_t_batch += sqrt_nn_t / n_internal
        n_avg_batch += n_avg / n_internal
        if (i_n + 1) % 2 == 0:
            print('Core %.i finished realisation %.i \n' % (i_batch, i_n + 1))
    np.save(path + os.sep + 'psipsi_evol' + '_' +'core' + str(i_batch + 1) + '.npy', psipsi_evol_batch)
    np.save(path + os.sep + 'sqrt_nn_evol' + '_' +'core' + str(i_batch + 1) + '.npy', sqrt_nn_evol_batch)
    np.save(path + os.sep + 'psipsi_t' + '_' + 'core' + str(i_batch + 1) + '.npy', psipsi_t_batch)
    np.save(path + os.sep + 'sqrt_nn_t' + '_' + 'core' + str(i_batch + 1) + '.npy', sqrt_nn_t_batch)
    np.save(path + os.sep + 'n_avg' + '_' + 'core' + str(i_batch + 1) + '.npy', n_avg_batch)
    return None

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
                psipsi_evol = np.zeros((len(t), N//2), dtype = complex)
                sqrt_nn_evol = np.zeros((len(t), N//2), dtype = complex)
                psipsi_t = np.zeros(len(t), dtype = complex)
                sqrt_nn_t = np.zeros(len(t), dtype = complex)
                n_avg = np.zeros((len(t), N//2), dtype = complex)
                print('--- Primary simulation parameters: sigma = %.2f, gamma0 = %.2f, gamma2 = %.2f' % (sigma, gamma0_array[0], gamma2))
                parallel_map(g1, range(n_batch), task_kwargs=dict(p = p, sigma = sigma, gamma0 = gamma0_array[0], gamma2 = gamma2, g = g, path = save_folder))
                for file in os.listdir(save_folder):
                    if 'psipsi_evol' in file:
                        psipsi_evol += np.load(save_folder + os.sep + file) / n_batch
                    elif 'sqrt_nn_evol' in file:
                        sqrt_nn_evol += np.load(save_folder + os.sep + file) / n_batch
                    elif 'psipsi_t' in file:
                        psipsi_t += np.load(save_folder + os.sep + file) / n_batch
                    elif 'sqrt_nn_t' in file:
                        sqrt_nn_t += np.load(save_folder + os.sep + file) / n_batch 
                    elif 'n_avg' in file:
                        n_avg += np.load(save_folder + os.sep + file) / n_batch
                nn = np.multiply(n_avg, n_avg[:, 0][:, np.newaxis])
                np.save(loc + os.sep + id_string + '__' + 'g1_SPATIAL_EVOL' + '.npy', (np.abs(psipsi_evol) / np.sqrt(nn)).real)
                np.save(loc + os.sep + id_string + '__' + 'g1_TEMP' + '.npy', (np.abs(psipsi_t) / np.sqrt(nn[:, 0])).real)
                np.save(loc + os.sep + id_string + '__' + 'g2_SPATIAL_EVOL' + '.npy', (np.abs(sqrt_nn_evol) / np.sqrt(nn)).real)
                np.save(loc + os.sep + id_string + '__' + 'g2_TEMP' + '.npy', (np.abs(sqrt_nn_t) / np.sqrt(nn[:, 0])).real)
        return None

final_save_remote = r'/home6/konstantinos'
call_avg(final_save_remote)