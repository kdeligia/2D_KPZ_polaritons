#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 15 16:19:12 2021

@author: delis
"""

c = 3e2 #μm/ps
hbar = 6.582119569 * 1e2 # μeV ps

from scipy import signal
import os
from scipy.fftpack import fft2, ifft2
import numpy as np
import external as ext
import math as m
import matplotlib.pyplot as pl
pl.close('all')

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
gamma0_tilde = 0.2 * 100
gammar_tilde = gamma0_tilde * 0.1
P_tilde = gamma0_tilde * 1
R_tilde = gammar_tilde / 1
ns_tilde = gammar_tilde / R_tilde
m_dim = m_tilde * melectron
Kc = hbar**2 / (2 * m_dim * hatepsilon * hatx**2)

print('ns = %.i' % (ns_tilde * hatrho))
print('Kc = %.4f' % Kc)

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
lengthwindow = i2-i1
t = ext.time(dt_tilde, N_steps, i1, i2, every)

x, kx =  ext.space_momentum(N, dx_tilde)
y = x
#isotropic_indices = ext.get_indices(x)
X, Y = np.meshgrid(x, y, sparse=True, indexing = 'xy')
KX, KY = np.meshgrid(kx, kx, sparse=True, indexing = 'xy')

class model:
    def __init__(self, p, sigma, om_tilde, g_dim, gr_dim, psi_x=0):
        self.dkx_tilde = kx[1] - kx[0]
        self.X, self.Y = np.meshgrid(x, x)
        self.KX, self.KY = np.meshgrid(kx, kx)
        self.p = p
        self.sigma = sigma
        self.om_tilde = om_tilde
        self.g_tilde = g_dim * hatrho / hatepsilon
        self.gr_tilde = gr_dim * hatrho / hatepsilon
        self.damp = 1 + 2 * self.p * self.gr_tilde / (R_tilde * self.om_tilde) + self.p / (2 * self.om_tilde) * 1j 
        self.psi_x = psi_x
        self.initcond = np.full((N,N), 0.01**(1/2))
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
        '''
        a_vort = 2 * dx_tilde
        vortex_positions_field, ignore = ext.vortices(a_vort, np.angle(self.psi_x), x, y)
        ext.vortex_plots(x, t, 0, vortex_positions_field, np.angle(self.psi_x), self.n(self.psi_x))
        '''

    def _set_fourier_psi_x(self, psi_x):
        self.psi_mod_x = psi_x * np.exp(-1j * self.KX[0,0] * self.X - 1j * self.KY[0,0] * self.Y) * dx_tilde * dx_tilde / (2 * np.pi)

    def _get_psi_x(self):
        return self.psi_mod_x * np.exp(1j * self.KX[0,0] * self.X + 1j * self.KY[0,0] * self.Y) * 2 * np.pi / (dx_tilde * dx_tilde)

    def _set_fourier_psi_k(self, psi_k):
        self.psi_mod_k = psi_k * np.exp(1j * self.X[0,0] * self.dkx_tilde * np.arange(N) + 1j * self.Y[0,0] * self.dkx_tilde * np.arange(N))

    def _get_psi_k(self):
        return self.psi_mod_k * np.exp(-1j * self.X[0,0] * self.dkx_tilde * np.arange(N) - 1j * self.Y[0,0] * self.dkx_tilde * np.arange(N))

    psi_x = property(_get_psi_x, _set_fourier_psi_x)
    psi_k = property(_get_psi_k, _set_fourier_psi_k)
# =======================================================================f======
# Definition of the split steps
# =============================================================================
    def n(self, psi):
        return (psi * np.conjugate(psi)).real

    def exp_x(self, n):
        if self.g_tilde == 0:
            self.uc_tilde = 0
        else:
            self.uc_tilde = self.g_tilde * (n + 2 * self.p * (self.gr_tilde / self.g_tilde) * (gamma0_tilde / R_tilde) * (1 / (1 + n / ns_tilde)))
        self.rd_tilde = (gamma0_tilde / 2) * (self.p / (1 + n / ns_tilde) - 1)
        return np.exp(-1j * 0.5 * dt_tilde * (self.uc_tilde + 1j * self.rd_tilde) / self.damp)

    def exp_k(self):
        return np.exp(-1j * dt_tilde * (KX ** 2 + KY ** 2) * Kc / self.damp)

# =============================================================================
# Time evolution
# =============================================================================
    def time_evolution(self):
        np.random.seed()
        a_unw = 16 * dx_tilde
        a_vort = 2 * dx_tilde
        xc = x[N//2]
        yc = y[N//2]
        #vortex_positions_scalar = np.zeros(len(t))
        #n = np.zeros(len(t))
        wound_sampling = np.zeros((4, len(t)))
        unwound_sampling = np.zeros((4, len(t)))
        #Vpin = 5 * signal.unit_impulse((N, N), (N//2, N//2)) + 5 * signal.unit_impulse((N, N), (N//2, N//2))
        Vpin = 0
        for l in range(N_steps):
            if l == 0:
                theta_wound_old = np.angle([self.psi_x[np.where(abs(y - yc) <= a_unw)[0][0], np.where(abs(x - xc) <= a_unw)[0][0]], 
                                           self.psi_x[np.where(abs(y - yc) <= a_unw)[0][0], np.where(abs(x - xc) <= a_unw)[0][-1]],
                                           self.psi_x[np.where(abs(y - yc) <= a_unw)[0][-1], np.where(abs(x - xc) <= a_unw)[0][-1]],
                                           self.psi_x[np.where(abs(y - yc) <= a_unw)[0][-1], np.where(abs(x - xc) <= a_unw)[0][0]]])
                theta_wound_new = theta_wound_old
                theta_unwound_old = theta_wound_old
                theta_unwound_new = theta_wound_old
                self.psi_x *= self.exp_x(self.n(self.psi_x)) * np.exp(-1j * 0.5 * dt_tilde * Vpin / self.damp)
                self.psi_mod_k = fft2(self.psi_mod_x)
                self.psi_k *= self.exp_k()
                self.psi_mod_x = ifft2(self.psi_mod_k)
                self.psi_x *= self.exp_x(self.n(self.psi_x)) * np.exp(-1j * 0.5 * dt_tilde * Vpin / self.damp)
                self.psi_x += np.sqrt(dt_tilde) * np.sqrt(self.sigma / dx_tilde ** 2) * (np.random.normal(0, 1, (N,N)) + 1j * np.random.normal(0, 1, (N,N))) / self.damp
            else:
                theta_wound_new = np.angle([self.psi_x[np.where(abs(y - yc) <= a_unw)[0][0], np.where(abs(x - xc) <= a_unw)[0][0]], 
                                           self.psi_x[np.where(abs(y - yc) <= a_unw)[0][0], np.where(abs(x - xc) <= a_unw)[0][-1]],
                                           self.psi_x[np.where(abs(y - yc) <= a_unw)[0][-1], np.where(abs(x - xc) <= a_unw)[0][-1]],
                                           self.psi_x[np.where(abs(y - yc) <= a_unw)[0][-1], np.where(abs(x - xc) <= a_unw)[0][0]]])
                theta_unwound_new = ext.unwinding(theta_wound_new, theta_wound_old, theta_unwound_old, 0.99)
                theta_wound_old = theta_wound_new
                theta_unwound_old = theta_unwound_new
                self.psi_x *= self.exp_x(self.n(self.psi_x))
                self.psi_mod_k = fft2(self.psi_mod_x)
                self.psi_k *= self.exp_k()
                self.psi_mod_x = ifft2(self.psi_mod_k)
                self.psi_x *= self.exp_x(self.n(self.psi_x))
                self.psi_x += np.sqrt(dt_tilde) * np.sqrt(self.sigma / dx_tilde ** 2) * (np.random.normal(0, 1, (N,N)) + 1j * np.random.normal(0, 1, (N,N))) / self.damp
            if l>=i1 and l<=i2 and l%every==0:
                time_index = (l-i1)//every
                unwound_sampling[:, time_index] = theta_unwound_new
                wound_sampling[:, time_index] = theta_wound_new
                '''
                print(l)
                vortex_positions_field, ignore = ext.vortices(a_vort, np.angle(self.psi_x), x, y)
                ext.vortex_plots(x, t, time_index, vortex_positions_field, np.angle(self.psi_x), self.n(self.psi_x))
                vortex_positions_scalar[time_index] = len(np.where(vortex_positions_field == 1)[0]) + len(np.where(vortex_positions_field == -1)[0])
                n[time_index] = np.mean(self.n(self.psi_x))
                '''
                '''
                indices = []
                for rad in range(N//2):
                    count = 0
                    indices += isotropic_indices.get('r = ' + str(rad))
                    for index_pair in range(len(indices)):
                        if vortex_field[indices[index_pair][0], indices[index_pair][1]] == 1 or vortex_field[indices[index_pair][0], indices[index_pair][1]] == -1:
                            count += 1
                    local_number[rad, time_array_index] = count
                    '''
        '''
        return local_number, wound_sampling, unwound_sampling
        '''
        #return wound_sampling, unwound_sampling, vortex_positions_scalar, n
        return unwound_sampling

# =============================================================================
# 
# =============================================================================
from qutip import *
parallel_tasks = 512
n_batch = 128
n_internal = parallel_tasks//n_batch
qutip.settings.num_cpus = n_batch

sigma_array = np.array([1e-2])
p_knob_array = np.array([2.0])
om_array = np.array([1e9])
p_array = p_knob_array * P_tilde * R_tilde / (gamma0_tilde * gammar_tilde)
gr_dim = 0
g_dim = 0

'''
g_dim = 6.82
sigma_th = gamma0_tilde * (p_array - 1) / 4
xi = hbar / (np.sqrt(2 * m_dim * g_dim * ns_tilde * (p_array - 1) * hatrho))
'''

path_remote = r'/scratch/konstantinos'
final_save_remote = r'/home6/konstantinos'
path_local = r'/Users/delis/Desktop'
final_save_local = r'/Users/delis/Desktop'

subfolders = ext.names_subfolders(True, path_remote, sigma_array, p_array, om_array, g_dim, gr_dim, gamma0_tilde, ns_tilde)

def phase(i_batch, p, sigma, om_tilde, g_dim, gr_dim):
    theta1_batch = np.zeros((4, len(t)))
    theta2_batch = np.zeros((4, len(t)))
    theta3_batch = np.zeros((4, len(t)))
    theta4_batch = np.zeros((4, len(t)))
    for i_n in range(n_internal):
        gpe = model(p, sigma, om_tilde, g_dim, gr_dim)
        theta = gpe.time_evolution()
        theta1_batch += theta / n_internal
        theta2_batch += theta ** 2 / n_internal
        theta3_batch += theta ** 3 / n_internal
        theta4_batch += theta ** 4 / n_internal
        np.savetxt(subfolders['p=' + str(p), 'sigma=' + str(sigma)] + os.sep + 'theta_points' + '_' + 'core' + str(i_batch + 1) + '_' + str(i_n + 1) + '.dat', theta)
        print('Core %.i finished realisation %.i' % (i_batch, i_n))
    np.savetxt(subfolders['p=' + str(p), 'sigma=' + str(sigma)] + os.sep + 'theta_linear' + '_' +'core' + str(i_batch + 1) + '.dat', theta1_batch)
    np.savetxt(subfolders['p=' + str(p), 'sigma=' + str(sigma)] + os.sep + 'theta_square' + '_' +'core' + str(i_batch + 1) + '.dat', theta2_batch)
    np.savetxt(subfolders['p=' + str(p), 'sigma=' + str(sigma)] + os.sep + 'theta_cube' + '_' +'core' + str(i_batch + 1) + '.dat', theta3_batch)
    np.savetxt(subfolders['p=' + str(p), 'sigma=' + str(sigma)] + os.sep + 'theta_fourth' + '_' +'core' + str(i_batch + 1) + '.dat', theta4_batch)
    return None

def call_avg(final_save):
    for sigma in sigma_array:
        for p in p_array:
            theta_list = []
            theta1 = np.zeros((4, len(t)))
            theta2 = np.zeros((4, len(t)))
            theta3 = np.zeros((4, len(t)))
            theta4 = np.zeros((4, len(t)))
            os.mkdir(subfolders['p=' + str(p), 'sigma=' + str(sigma)])
            print('Starting phase simulations: sigma = %.2f, p = %.1f, g = %.1f, gamma0 = %.i' % (sigma, p, g_dim, gamma0_tilde))
            parallel_map(phase, range(n_batch), task_kwargs=dict(p=p, sigma=sigma, om_tilde=om_array[0], g_dim=g_dim, gr_dim=gr_dim))
            for file in os.listdir(subfolders['p=' + str(p), 'sigma=' + str(sigma)]):
                if 'theta_points' in file:
                    theta_list.append(np.loadtxt(subfolders['p=' + str(p), 'sigma=' + str(sigma)] + os.sep + file))
                elif 'theta_linear' in file:
                    theta1 += np.loadtxt(subfolders['p=' + str(p), 'sigma=' + str(sigma)] + os.sep + file) / n_batch
                elif 'theta_square' in file:
                    theta2 += np.loadtxt(subfolders['p=' + str(p), 'sigma=' + str(sigma)] + os.sep + file) / n_batch
                elif 'theta_cube' in file:
                    theta3 += np.loadtxt(subfolders['p=' + str(p), 'sigma=' + str(sigma)] + os.sep + file) / n_batch
                elif 'theta_fourth' in file:
                    theta4 += np.loadtxt(subfolders['p=' + str(p), 'sigma=' + str(sigma)] + os.sep + file) / n_batch
            theta = np.concatenate(theta_list, axis = 0)
            np.savetxt(final_save + os.sep + 'realisations' + 
                '_' + 'sigma' + str(sigma) + 
                '_' + 'p' + str(p) + 
                '_' + 'om' + str(int(om_array[0])) + 
                '_' + 'g' + str(g_dim) + '_' + 'gr' + str(gr_dim) + '_' + 'gamma' + str(gamma0_tilde) +'.dat', theta)
            np.savetxt(final_save + os.sep + 'deltatheta2' + 
                '_' + 'sigma' + str(sigma) + 
                '_' + 'p' + str(p) + 
                '_' + 'om' + str(int(om_array[0])) + 
                '_' + 'g' + str(g_dim) + '_' + 'gr' + str(gr_dim) + '_' + 'gamma' + str(gamma0_tilde) +'.dat', theta2 - theta1 ** 2)
            np.savetxt(final_save + os.sep + 'deltatheta3' + 
                '_' + 'sigma' + str(sigma) + 
                '_' + 'p' + str(p) + 
                '_' + 'om' + str(int(om_array[0])) + 
                '_' + 'g' + str(g_dim) + '_' + 'gr' + str(gr_dim) + '_' + 'gamma' + str(gamma0_tilde) +'.dat', theta3 - 3 * theta2 * theta1 + 2 * theta1 ** 3)
            np.savetxt(final_save + os.sep + 'deltatheta4' + 
                '_' + 'sigma' + str(sigma) + 
                '_' + 'p' + str(p) + 
                '_' + 'om' + str(int(om_array[0])) + 
                '_' + 'g' + str(g_dim) + '_' + 'gr' + str(gr_dim) + '_' + 'gamma' + str(gamma0_tilde) +'.dat', theta4 - 4 * theta3 * theta1 + 6 * theta2 * theta1 ** 2 - 3 * theta1 ** 4)
    return None

call_avg(final_save_remote)

'''
gpe = model(p_array[0], sigma_array[0], om_array[0], g_dim, gr_dim)
wound, unwound, number, n = gpe.time_evolution()
'''