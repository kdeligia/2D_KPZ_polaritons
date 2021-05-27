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
hatpsi = 1/hatx # μm^-1
hatrho = 1/hatx**2 # μm^-2
hatepsilon = hbar/hatt # μeV
melectron = 0.510998950 * 1e12 / c**2 # μeV/(μm^2/ps^2)

'''
m_tilde = 6.2e-5
gamma0_tilde = 0.22
gammar_tilde = gamma0_tilde * 0.1
P_tilde = gamma0_tilde * 500
R_tilde = gammar_tilde / 500
'''

m_tilde = 3.8e-5 * 3
m_dim = m_tilde * melectron

# =============================================================================
# 
# =============================================================================
N = 2 ** 6
dx_tilde = 0.5

N_steps = 5000000
dt_tilde = 1e-3
every = 5000
i1 = 200000
i2 = N_steps
t = ext.time(dt_tilde, N_steps, i1, i2, every)

x, y = ext.space_momentum(N, dx_tilde)
isotropic_indices = ext.get_indices(x)

kx = (2*np.pi)/dx_tilde * np.fft.fftfreq(N, d=1)
ky = (2*np.pi)/dx_tilde * np.fft.fftfreq(N, d=1)

class model:
    def __init__(self, p, sigma, gamma2, gamma0, g, gr, ns):
        self.KX, self.KY = np.meshgrid(kx, ky, sparse=True)
        self.gamma2_tilde = gamma2  * hatt / hatx **2
        self.gamma0_tilde = gamma0 * hatt
        self.gammar_tilde = 0.1 * self.gamma0_tilde
        self.Kc = hbar ** 2 / (2 * m_dim * hatepsilon * hatx**2)
        self.Kd = gamma2 / 2
        self.g_tilde = g * hatrho / hatepsilon
        self.gr_tilde = gr* hatrho / hatepsilon
        
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
        print('--- Rest Parameters (dimensional) ---')
        print('g = %.2f, gamma0 = %.2f, ns = %.i' % (g, gamma0, self.ns_tilde))
        print('--- Kinetic Terms ---')
        print(' Kc = %.4f, Kd = %.4f' % (self.Kc, self.Kd))

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
        '''
        a_unw = 16 * dx_tilde
        xc = x[N//2]
        yc = y[N//2]
        wound_sampling = np.zeros((4, len(t)))
        unwound_sampling = np.zeros((4, len(t)))
        '''
        wound_sampling = np.zeros(len(t))
        unwound_sampling = np.zeros(len(t))
        for i in range(N_steps):
            if i == 0:
                '''
                theta_wound_old = np.angle([self.psi_x[np.where(abs(y - yc) <= a_unw)[0][0], np.where(abs(x - xc) <= a_unw)[0][0]], 
                                           self.psi_x[np.where(abs(y - yc) <= a_unw)[0][0], np.where(abs(x - xc) <= a_unw)[0][-1]],
                                           self.psi_x[np.where(abs(y - yc) <= a_unw)[0][-1], np.where(abs(x - xc) <= a_unw)[0][-1]],
                                           self.psi_x[np.where(abs(y - yc) <= a_unw)[0][-1], np.where(abs(x - xc) <= a_unw)[0][0]]])
                '''
                theta_wound_old = np.angle([self.psi_x[N//2, N//2]])
                theta_wound_new = theta_wound_old
                theta_unwound_old = theta_wound_old
                theta_unwound_new = theta_wound_old
                self.psi_x *= self.exp_x(0.5 * dt_tilde, self.n(self.psi_x))
                psi_k = fft2(self.psi_x)
                psi_k *= self.exp_k(dt_tilde)
                self.psi_x = ifft2(psi_k)
                self.psi_x *= self.exp_x(0.5 * dt_tilde, self.n(self.psi_x))
                self.psi_x += np.sqrt(dt_tilde) * np.sqrt(self.sigma / dx_tilde ** 2) * (np.random.normal(0, 1, (N,N)) + 1j * np.random.normal(0, 1, (N,N)))
            else:
                '''
                theta_wound_new = np.angle([self.psi_x[np.where(abs(y - yc) <= a_unw)[0][0], np.where(abs(x - xc) <= a_unw)[0][0]], 
                                           self.psi_x[np.where(abs(y - yc) <= a_unw)[0][0], np.where(abs(x - xc) <= a_unw)[0][-1]],
                                           self.psi_x[np.where(abs(y - yc) <= a_unw)[0][-1], np.where(abs(x - xc) <= a_unw)[0][-1]],
                                           self.psi_x[np.where(abs(y - yc) <= a_unw)[0][-1], np.where(abs(x - xc) <= a_unw)[0][0]]])
                '''
                theta_wound_new = np.angle([self.psi_x[N//2, N//2]])
                theta_unwound_new = ext.unwinding(theta_wound_new, theta_wound_old, theta_unwound_old, 0.99)
                theta_wound_old = theta_wound_new
                theta_unwound_old = theta_unwound_new
                self.psi_x *= self.exp_x(0.5 * dt_tilde, self.n(self.psi_x))
                psi_k = fft2(self.psi_x)
                psi_k *= self.exp_k(dt_tilde)
                self.psi_x = ifft2(psi_k)
                self.psi_x *= self.exp_x(0.5 * dt_tilde, self.n(self.psi_x))
                self.psi_x += np.sqrt(dt_tilde) * np.sqrt(self.sigma / dx_tilde ** 2) * (np.random.normal(0, 1, (N,N)) + 1j * np.random.normal(0, 1, (N,N)))
            if i >= i1 and i <= i2 and i % every == 0:
                print(i)
                time_index = (i-i1)//every
                '''
                unwound_sampling[:, time_index] = theta_unwound_new
                wound_sampling[:, time_index] = theta_wound_new
                '''
                unwound_sampling[time_index] = theta_unwound_new
                wound_sampling[time_index] = theta_wound_new
        return unwound_sampling

# =============================================================================
# 
# =============================================================================
from qutip import *
parallel_tasks = 1
n_batch = 1
n_internal = parallel_tasks//n_batch
qutip.settings.num_cpus = n_batch

sigma_array = np.array([1e-2])
p_array = np.array([2])
gamma0_array = np.array([20])
gamma2_array = np.array([1e-10])
gr = 0
g = 0
ns = 1

'''
g_dim = 6.82
sigma_th = gamma0_array * (p_array + 1) / 4
xi = hbar / (np.sqrt(2 * m_dim * g_dim * ns * (p_array - 1) * hatrho))
print(sigma_th)
'''

path_remote = r'/scratch/konstantinos'
final_save_remote = r'/home6/konstantinos'
path_local = r'/Users/delis/Desktop'
final_save_local = r'/Users/delis/Desktop'

#subfolders = ext.names_subfolders(True, path_local, sigma_array, p_array, gamma2_array, gamma0_array, g, gr, ns)

def phase(i_batch, p, sigma, gamma2, gamma0, g, gr, ns):
    '''
    theta1_batch = np.zeros((4, len(t)))
    theta2_batch = np.zeros((4, len(t)))
    theta3_batch = np.zeros((4, len(t)))
    theta4_batch = np.zeros((4, len(t)))
    '''
    theta1_batch = np.zeros(len(t))
    theta2_batch = np.zeros(len(t))
    theta3_batch = np.zeros(len(t))
    theta4_batch = np.zeros(len(t))
    path_current = subfolders.get(('p=' + str(p), 'sigma=' + str(sigma), 'gamma2=' + str(gamma2), 'gamma0=' + str(gamma0)))
    for i_n in range(n_internal):
        gpe = model(p, sigma, gamma2, gamma0, g, gr, ns)
        theta = gpe.time_evolution()
        theta1_batch += theta / n_internal
        theta2_batch += theta ** 2 / n_internal
        theta3_batch += theta ** 3 / n_internal
        theta4_batch += theta ** 4 / n_internal
        np.savetxt(path_current + os.sep + 'trajectories' + '_' + 'core' + str(i_batch + 1) + '_' + str(i_n + 1) + '.dat', theta)
    np.savetxt(path_current + os.sep + 'theta_linear' + '_' +'core' + str(i_batch + 1) + '.dat', theta1_batch)
    np.savetxt(path_current + os.sep + 'theta_square' + '_' +'core' + str(i_batch + 1) + '.dat', theta2_batch)
    np.savetxt(path_current + os.sep + 'theta_cube' + '_' +'core' + str(i_batch + 1) + '.dat', theta3_batch)
    np.savetxt(path_current + os.sep + 'theta_fourth' + '_' +'core' + str(i_batch + 1) + '.dat', theta4_batch)
    return None

def call_avg(final_save_path):
    for sigma in sigma_array:
        for p in p_array:
            for gamma2 in gamma2_array:
                for gamma0 in gamma0_array:
                    id_string = 'sigma' + str(sigma) + '_' + 'p' + str(p) + '_' + 'gamma2' + str(gamma2) + '_' + 'gamma' + str(gamma0) + '_' + 'g' + str(g) + '_' + 'gr' + str(gr)
                    path_current = subfolders.get(('p=' + str(p), 'sigma=' + str(sigma), 'gamma2=' + str(gamma2), 'gamma0=' + str(gamma0)))
                    os.mkdir(path_current)
                    trajectories = []
                    '''
                    theta1 = np.zeros((4, len(t)))
                    theta2 = np.zeros((4, len(t)))
                    theta3 = np.zeros((4, len(t)))
                    theta4 = np.zeros((4, len(t)))
                    '''
                    theta1 = np.zeros(len(t))
                    theta2 = np.zeros(len(t))
                    theta3 = np.zeros(len(t))
                    theta4 = np.zeros(len(t))
                    print('Starting phase simulations: sigma = %.2f, p = %.1f, gamma2 = %.2f' % (sigma, p, gamma2))
                    parallel_map(phase, range(n_batch), task_kwargs=dict(p = p, sigma = sigma, gamma2 = gamma2, gamma0 = gamma0, g=g, gr = gr, ns = ns), progress_bar=True)
                    for file in os.listdir(path_current):
                        if 'trajectories' in file:
                            trajectories.append(np.loadtxt(path_current + os.sep + file))
                        elif 'theta_linear' in file:
                            theta1 += np.loadtxt(path_current + os.sep + file) / n_batch
                        elif 'theta_square' in file:
                            theta2 += np.loadtxt(path_current + os.sep + file) / n_batch
                        elif 'theta_cube' in file:
                            theta3 += np.loadtxt(path_current + os.sep + file) / n_batch
                        elif 'theta_fourth' in file:
                            theta4 += np.loadtxt(path_current + os.sep + file) / n_batch
                    np.savetxt(final_save_path + os.sep + id_string + '_' + 'trajectories' + '.dat', np.concatenate(trajectories, axis = 0))
                    np.savetxt(final_save_path + os.sep + id_string + '_' + 'deltatheta2'  + '.dat', theta2 - theta1**2)
                    np.savetxt(final_save_path + os.sep + id_string + '_' + 'deltatheta3'  + '.dat', theta3 - 3*theta2*theta1 + 2*theta1**3)
                    np.savetxt(final_save_path + os.sep + id_string + '_' + 'deltatheta4'  + '.dat', theta4 - 4*theta3*theta1 + 6*theta2*theta1**2 - 3*theta1**4)
        return None

#call_avg(final_save_local)