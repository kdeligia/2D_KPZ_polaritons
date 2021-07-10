#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 15 16:19:12 2021

@author: delis
"""

c = 3e2 #μm/ps
hbar = 6.582119569 * 1e2 # μeV ps

import os
import numpy as np
import external as ext
from scipy.fftpack import fft2, ifft2
#import pyfftw
#pyfftw.interfaces.cache.enable()

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

sampling_begin = 1000000
sampling_step = 1000
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
        '''
        self.initcond[N//2, N//2] = 0
        theta = np.zeros((N, N))
        rot = np.ones((N, N), dtype = complex)
        for x0 in range(N):
            for y0 in range(N):
                theta[y0, x0] = m.atan2(x[y0], x[x0])
        rot = np.exp(1 * 1j * theta)
        '''
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
        a_unw = 32 * dx_tilde
        xc = x[N//2]
        yc = y[N//2]
        wound_sampling = np.zeros((4, len(t)))
        unwound_sampling = np.zeros((4, len(t)))
        for i in range(N_steps):
            self.psi_x *= self.exp_x(0.5 * dt_tilde, self.n(self.psi_x))
            psi_k = fft2(self.psi_x)
            psi_k *= self.exp_k(dt_tilde)
            self.psi_x = ifft2(psi_k)
            self.psi_x *= self.exp_x(0.5 * dt_tilde, self.n(self.psi_x))
            self.psi_x += np.sqrt(dt_tilde) * np.sqrt(self.sigma / dx_tilde ** 2) * (np.random.normal(0, 1, (N,N)) + 1j * np.random.normal(0, 1, (N,N)))
            if i == 0:
                theta_wound_old = np.angle([self.psi_x[np.where(abs(y - yc) <= a_unw)[0][0], np.where(abs(x - xc) <= a_unw)[0][0]], 
                                           self.psi_x[np.where(abs(y - yc) <= a_unw)[0][0], np.where(abs(x - xc) <= a_unw)[0][-1]],
                                           self.psi_x[np.where(abs(y - yc) <= a_unw)[0][-1], np.where(abs(x - xc) <= a_unw)[0][-1]],
                                           self.psi_x[np.where(abs(y - yc) <= a_unw)[0][-1], np.where(abs(x - xc) <= a_unw)[0][0]]])
                theta_wound_new = theta_wound_old
                theta_unwound_old = theta_wound_old
                theta_unwound_new = theta_wound_old
            else:
                theta_wound_new = np.angle([self.psi_x[np.where(abs(y - yc) <= a_unw)[0][0], np.where(abs(x - xc) <= a_unw)[0][0]], 
                                           self.psi_x[np.where(abs(y - yc) <= a_unw)[0][0], np.where(abs(x - xc) <= a_unw)[0][-1]],
                                           self.psi_x[np.where(abs(y - yc) <= a_unw)[0][-1], np.where(abs(x - xc) <= a_unw)[0][-1]],
                                           self.psi_x[np.where(abs(y - yc) <= a_unw)[0][-1], np.where(abs(x - xc) <= a_unw)[0][0]]])
                theta_unwound_new = ext.unwinding(theta_wound_new, theta_wound_old, theta_unwound_old, 0.99)
                theta_wound_old = theta_wound_new
                theta_unwound_old = theta_unwound_new
            if i >= sampling_begin and i <= sampling_end and i % sampling_step == 0:
                time_index = (i - sampling_begin) // sampling_step
                unwound_sampling[:, time_index] = theta_unwound_new
                wound_sampling[:, time_index] = theta_wound_new
        return unwound_sampling

# =============================================================================
# 
# =============================================================================
from qutip import *
parallel_tasks = 1024
n_batch = 128
n_internal = parallel_tasks//n_batch
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
init = r'/scratch/konstantinos' + os.sep + 'N' + str(N) + '_' + 'ns' + str(int(ns)) + '_' + 'm' + str(m_array[0])
if os.path.isdir(init) == False:
    os.mkdir(init)
ids = ext.ids(p_array, sigma_array, gamma0_array, gamma2_array, g_array)

def phase(i_batch, p, sigma, gamma0, gamma2, g, path):
    for i_n in range(n_internal):
        gpe = model(p, sigma, gamma0, gamma2, g, gr = gr, ns = ns, m = m_array[0])
        theta = gpe.time_evolution()
        np.savetxt(path + os.sep + 'trajectories' + '_' + 'core' + str(i_batch + 1) + '_' + str(i_n + 1) + '.dat', theta)
        if (i_n + 1) % 2 == 0:
            print('Core %.i finished realisation %.i \n' % (i_batch, i_n + 1))
    return None

def call_avg(final_save_path):
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
                trajectories = []
                print('Starting phase simulations: sigma = %.2f, p = %.1f, gamma2 = %.e' % (sigma, p, gamma2))
                parallel_map(phase, range(n_batch), task_kwargs=dict(p = p, sigma = sigma, gamma0 = gamma0_array[0], gamma2 = gamma2, g = g, path = save_folder))
                for file in os.listdir(save_folder):
                    if 'trajectories' in file:
                        trajectories.append(np.loadtxt(save_folder + os.sep + file))
                np.savetxt(final_save_path + os.sep + id_string + '_' + 'trajectories' + '.dat', np.concatenate(trajectories, axis = 0))
        return None

final_save_remote = r'/home6/konstantinos'
call_avg(final_save_remote)