#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  9 15:36:47 2021

@author: delis
"""

import numpy as np
import external as ext
from scipy.fftpack import fft2, ifft2

c = 3e2 #μm/ps
hbar = 6.582119569 * 1e2 # μeV ps
melectron = 0.510998950 * 1e12 / c ** 2 # μeV/(μm^2/ps^2)

hatt = 32 # ps
hatx = 4 * np.sqrt(2)
hatpsi = 1 / hatx # μm^-1
hatrho = 1 / hatx ** 2 # μm^-2
hatepsilon = hbar / hatt # μeV

class gpe:
    def __init__(self, **args):
        self.N = args.get('N')
        self.dx = args.get('dx')
        self.x, self.y = ext.space_grid(self.N, self.dx)

        self.kx = (2 * np.pi) / self.dx * np.fft.fftfreq(self.N, d = 1)
        self.ky = (2 * np.pi) / self.dx * np.fft.fftfreq(self.N, d = 1)
        self.KX, self.KY = np.meshgrid(self.kx, self.ky, sparse = True)
        
        self.gamma2_tilde = args.get('gamma2')  * hatt / hatx ** 2
        self.gamma0_tilde = args.get('gamma0') * hatt
        self.gammar_tilde = 0.1 * self.gamma0_tilde

        self.Kc = hbar ** 2 / (2 * args.get('m') * melectron * hatepsilon * hatx ** 2)
        self.Kd = self.gamma2_tilde / 2
        self.g_tilde = args.get('g') * hatrho / hatepsilon
        self.gr_tilde = args.get('gr') * hatrho / hatepsilon

        self.R_tilde = self.gammar_tilde / args.get('ns')
        self.ns_tilde = self.gammar_tilde / self.R_tilde
        self.P_tilde = args.get('p') * self.gamma0_tilde * self.ns_tilde
        self.p = self.P_tilde * self. R_tilde / (self.gamma0_tilde * self.gammar_tilde)
        self.sigma = args.get('sigma')

        self.psi_x = 0.4 * (np.ones((self.N, self.N)) + 1j * np.ones((self.N, self.N)))
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
    def time_evolution_vortices(self, folder, **time_dict):
        np.random.seed()
        N_input = time_dict.get('N_input')
        i_start = time_dict.get('i_start')
        di = time_dict.get('di')
        dt = time_dict.get('dt')
        t = ext.time(dt, N_input, i_start, di)
        N_i = N_input + i_start + di
        
        sigma = 0
        a = 2
        #vortex_number = np.zeros(int(N_input / di) + 1)
        density = np.zeros(int(N_input / di) + 1)
        for i in range(N_i):
            self.psi_x *= self.exp_x(0.5 * dt, self.n(self.psi_x))
            psi_k = fft2(self.psi_x)
            psi_k *= self.exp_k(dt)
            self.psi_x = ifft2(psi_k)
            self.psi_x *= self.exp_x(0.5 * dt, self.n(self.psi_x))
            self.psi_x += np.sqrt(dt) * np.sqrt(sigma / self.dx ** 2) * (np.random.normal(0, 1, (self.N, self.N)) + 1j * np.random.normal(0, 1, (self.N, self.N)))
            if i >= i_start and i <= N_i and i % di == 0:
                time_index = (i - i_start) // di
                vortex_positions = ext.vortex_detect(a, np.angle(self.psi_x), self.x, self.y)
                ext.vortex_plots(folder, self.x, t, time_index, vortex_positions, np.angle(self.psi_x), self.n(self.psi_x))
                #vortex_number[time_index] = len(np.where(vortex_positions == 1)[0]) + len(np.where(vortex_positions == -1)[0])
                density[time_index] = np.mean(self.n(self.psi_x))
                if time_index >= 1250 and time_index <= 1750:
                    sigma = self.sigma
                else:
                    sigma = 0
        return density
    
    def time_evolution_theta(self, **time_dict):
        np.random.seed()
        N_input = time_dict.get('N_input')
        i_start = time_dict.get('i_start')
        di = time_dict.get('di')
        dt = time_dict.get('dt')
        N_i = N_input + i_start + di

        a_unw = (self.N // 4) * self.dx
        xc = self.x[self.N // 2]
        yc = self.y[self.N // 2]
        unwound_sampling = np.zeros((4, int(N_input / di) + 1))
        for i in range(N_i):
            self.psi_x *= self.exp_x(0.5 * dt, self.n(self.psi_x))
            psi_k = fft2(self.psi_x)
            psi_k *= self.exp_k(dt)
            self.psi_x = ifft2(psi_k)
            self.psi_x *= self.exp_x(0.5 * dt, self.n(self.psi_x))
            self.psi_x += np.sqrt(dt) * np.sqrt(self.sigma / self.dx ** 2) * (np.random.normal(0, 1, (self.N, self.N)) + 1j * np.random.normal(0, 1, (self.N, self.N)))
            if i == 0:
                theta_wound_old = np.angle([self.psi_x[np.where(abs(self.y - yc) <= a_unw)[0][0], np.where(abs(self.x - xc) <= a_unw)[0][0]], 
                                           self.psi_x[np.where(abs(self.y - yc) <= a_unw)[0][0], np.where(abs(self.x - xc) <= a_unw)[0][-1]],
                                           self.psi_x[np.where(abs(self.y - yc) <= a_unw)[0][-1], np.where(abs(self.x - xc) <= a_unw)[0][-1]],
                                           self.psi_x[np.where(abs(self.y - yc) <= a_unw)[0][-1], np.where(abs(self.x - xc) <= a_unw)[0][0]]])
                theta_wound_new = theta_wound_old
                theta_unwound_old = theta_wound_old
                theta_unwound_new = theta_wound_old
            else:
                theta_wound_new = np.angle([self.psi_x[np.where(abs(self.y - yc) <= a_unw)[0][0], np.where(abs(self.x - xc) <= a_unw)[0][0]], 
                                           self.psi_x[np.where(abs(self.y - yc) <= a_unw)[0][0], np.where(abs(self.x - xc) <= a_unw)[0][-1]],
                                           self.psi_x[np.where(abs(self.y - yc) <= a_unw)[0][-1], np.where(abs(self.x - xc) <= a_unw)[0][-1]],
                                           self.psi_x[np.where(abs(self.y - yc) <= a_unw)[0][-1], np.where(abs(self.x - xc) <= a_unw)[0][0]]])
                theta_unwound_new = theta_unwound_old + ext.unwinding(theta_wound_new - theta_wound_old,  0.99)
                theta_wound_old = theta_wound_new
                theta_unwound_old = theta_unwound_new
            if i >= i_start and i <= N_i and i % di == 0:
                time_index = (i - i_start) // di
                unwound_sampling[:, time_index] = theta_unwound_new
        return unwound_sampling

    def time_evolution_psi(self, **time_dict):
        np.random.seed()
        N_input = time_dict.get('N_input')
        i_start = time_dict.get('i_start')
        di = time_dict.get('di')
        dt = time_dict.get('dt')
        N_i = N_input + i_start + di

        isotropic_indices = ext.get_indices(self.x)
        center_indices = isotropic_indices.get('r = ' + str(0))
        psipsi_full = np.zeros((int(N_input / di) + 1, self.N // 2), dtype = complex)
        n_avg = np.zeros((int(N_input / di) + 1, self.N // 2), dtype = complex)
        for i in range(N_i):
            self.psi_x *= self.exp_x(0.5 * dt, self.n(self.psi_x))
            psi_k = fft2(self.psi_x)
            psi_k *= self.exp_k(dt)
            self.psi_x = ifft2(psi_k)
            self.psi_x *= self.exp_x(0.5 * dt, self.n(self.psi_x))
            self.psi_x += np.sqrt(dt) * np.sqrt(self.sigma / self.dx ** 2) * (np.random.normal(0, 1, (self.N, self.N)) + 1j * np.random.normal(0, 1, (self.N, self.N)))
            if i >= i_start and i <= N_i and i % di == 0:
                time_index = (i - i_start) // di
                if i == i_start:
                    psi_x0t0 = self.psi_x[center_indices[0][0], center_indices[0][1]]
                psipsi_full[time_index] = ext.isotropic_avg('correlation', self.psi_x, np.conjugate(psi_x0t0), **isotropic_indices)
                n_avg[time_index] = ext.isotropic_avg('density average', self.n(self.psi_x), None, **isotropic_indices)
        return psipsi_full, n_avg