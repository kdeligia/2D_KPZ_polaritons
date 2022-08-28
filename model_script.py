#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  9 15:36:47 2021

@author: delis
"""

import numpy as np
import utils
from scipy.fftpack import fft2, ifft2
import matplotlib.pyplot as pl

c = 3e2  # μm ps^-1
hbar = 658.2119569  # μeV ps
melectron = 0.510998950 * 1e12 / c ** 2  # μeV ps^2 μm^-2)


class gpe:
    def __init__(self, N, dx, **args):
        self.l0 = args.get('l0')
        self.tau0 = args.get('tau0')
        self.psi0 = 1 / self.l0
        self.rho0 = 1 / self.l0 ** 2
        self.epsilon0 = hbar / self.tau0
        self.N = N
        self.dx = dx
        self.x, self.y = utils.space_grid(self.N, self.dx)
        self.kx = 2 * np.pi * np.fft.fftfreq(self.N, d=self.dx)
        self.ky = 2 * np.pi * np.fft.fftfreq(self.N, d=self.dx)
        self.KX, self.KY = np.meshgrid(self.kx, self.ky, sparse=True)
        self.gamma2_tilde = args.get('gamma2') * self.tau0 / self.l0 ** 2
        self.gamma0_tilde = args.get('gamma0') * self.tau0
        self.gammar_tilde = 0.1 * self.gamma0_tilde

        self.Kc = hbar ** 2 / (2 * args.get('m') * melectron * self.epsilon0 * self.l0 ** 2)
        self.tb0 = 5.8
        self.J = hbar ** 2 / (2 * args.get('m') * melectron * 2 * (self.dx * self.l0) ** 2)

        self.g_tilde = args.get('g') * self.rho0 / self.epsilon0
        self.gr_tilde = args.get('gr') * self.rho0 / self.epsilon0

        self.ns_tilde = args.get('ns') / self.rho0
        self.R_tilde = self.gammar_tilde / self.ns_tilde
        self.P_tilde = args.get('p') * self.gamma0_tilde * self.ns_tilde
        self.p = self.P_tilde * self. R_tilde / (self.gamma0_tilde * self.gammar_tilde)
        self.psi_x = 0.4 * (np.ones((self.N, self.N)) + 1j * np.ones((self.N, self.N)))
        self.psi_x /= self.psi0

    def n(self, psi):
        return np.real(psi * np.conjugate(psi))

    def tb_dispersion(self):
        return self.tb0 - 2 * (self. J / self.epsilon0) * (np.cos(self.KX * self.dx) + np.cos(self.KY * self.dx))

    def exp_x(self, dt, n):
        if self.g_tilde == 0:
            self.uc_tilde = 0
        else:
            self.uc_tilde = self.g_tilde * (n + 2 * self.p * (self.gr_tilde / self.g_tilde) * (self.gamma0_tilde / self.R_tilde) * (1 / (1 + n / self.ns_tilde)))
        self.rd_tilde = (self.gamma0_tilde / 2) * (self.p / (1 + n / self.ns_tilde) - 1)
        return np.exp(-1j * dt * (self.uc_tilde + 1j * self.rd_tilde))
    '''
    def exp_k(self, dt):
        return np.exp(-1j * dt * (self.KX ** 2 + self.KY ** 2) * (self.Kc - 1j * self.gamma2_tilde / 2))
    '''

    def exp_k(self, dt):
        return np.exp(-1j * dt * ((self.KX ** 2 + self.KY ** 2) * (- 1j * self.gamma2_tilde / 2) + self.tb_dispersion()))

# =============================================================================
# Time evolution
# =============================================================================
    def time_evolution_spacetime_vortices(self, **time):
        np.random.seed()
        N_input = int(time.get('N_input'))
        dt = time.get('dt')
        di = time.get('di')
        tmin = time.get('tmin')
        tmax = time.get('tmax')
        self.sigma = self.gamma0_tilde * (self.p + 1) / 4 * (time.get('dt') / self.dx ** 2)
        theta_unw = []
        flag = False
        for i in range(N_input + 1):
            ti = i * dt * self.tau0
            self.psi_x *= self.exp_x(0.5 * dt, self.n(self.psi_x))
            psi_k = fft2(self.psi_x)
            psi_k *= self.exp_k(dt)
            self.psi_x = ifft2(psi_k)
            self.psi_x *= self.exp_x(0.5 * dt, self.n(self.psi_x))
            self.psi_x += np.sqrt(self.sigma) * (np.random.normal(0, 1, (self.N, self.N)) + 1j * np.random.normal(0, 1, (self.N, self.N)))
            if i == 0:
                theta_wound_old = np.angle(self.psi_x)
                theta_wound_new = theta_wound_old
                theta_unwound_old = theta_wound_old
                theta_unwound_new = theta_wound_old
            else:
                theta_wound_new = np.angle(self.psi_x)
                theta_unwound_new = theta_unwound_old + utils.unwinding(theta_wound_new - theta_wound_old)
                theta_wound_old = theta_wound_new
                theta_unwound_old = theta_unwound_new
            if int(ti) >= tmin and int(ti) <= tmax:
                flag = True
            if flag is True and i % di == 0:
                theta_unw.append(theta_unwound_new)
        return theta_unw

    def time_evolution_theta(self, **time):
        np.random.seed()
        N_input = int(time.get('N_input'))
        dt = time.get('dt')
        di = time.get('di')
        self.sigma = time.get('dt') / self.dx ** 2 * self.gamma0_tilde * (self.p + 1) / 4
        unwound_sampling = np.zeros((int(N_input / di) + 1))
        for i in range(N_input + 1):
            ti = i * dt * self.tau0
            self.psi_x *= self.exp_x(0.5 * dt, self.n(self.psi_x))
            psi_k = fft2(self.psi_x)
            psi_k *= self.exp_k(dt)
            self.psi_x = ifft2(psi_k)
            self.psi_x *= self.exp_x(0.5 * dt, self.n(self.psi_x))
            self.psi_x += np.sqrt(self.sigma) * (np.random.normal(0, 1, (self.N, self.N)) + 1j * np.random.normal(0, 1, (self.N, self.N)))
            if ti == 0:
                theta_wound_old = np.angle(self.psi_x)
                theta_wound_new = theta_wound_old
                theta_unwound_old = theta_wound_old
                theta_unwound_new = theta_wound_old
            else:
                theta_wound_new = np.angle(self.psi_x)
                theta_unwound_new = theta_unwound_old + utils.unwinding(theta_wound_new - theta_wound_old)
                theta_wound_old = theta_wound_new
                theta_unwound_old = theta_unwound_new
            if i % di == 0:
                time_index = i // di
                unwound_sampling[time_index] = theta_unwound_new[self.N//2, self.N//2]              # Here possibly you might want to update, depending on what you want to extract. Maybe in θ in multiple space points etc
        return unwound_sampling

    def time_evolution_psi(self, **time):
        np.random.seed()
        Nsteps = time.get('Nsteps')
        di = time.get('di')
        dt = time.get('dt')
        tss = time.get('tss')

        self.sigma = time.get('dt') / self.dx ** 2 * self.gamma0_tilde * (self.p + 1) / 4
        isotropic_indices = utils.get_radial_indices(self.x)
        center_indices = isotropic_indices.get('r = ' + str(0))

        psi_correlation = []
        n_correlation = []
        n_avg = []
        deltatheta_full = []

        for i in range(Nsteps + 1):
            ti = i * dt * self.tau0
            self.psi_x *= self.exp_x(0.5 * dt, self.n(self.psi_x))
            psi_k = fft2(self.psi_x)
            psi_k *= self.exp_k(dt)
            self.psi_x = ifft2(psi_k)
            self.psi_x *= self.exp_x(0.5 * dt, self.n(self.psi_x))
            self.psi_x += np.sqrt(self.sigma) * (np.random.normal(0, 1, (self.N, self.N)) + 1j * np.random.normal(0, 1, (self.N, self.N)))
            if ti >= tss and i % di == 0:
                if ti == tss:
                    psi_x0t0 = self.psi_x[center_indices[0][0], center_indices[0][1]]
                    n_x0t0 = psi_x0t0 * np.conjugate(psi_x0t0)
                    theta_x0t0 = np.angle(psi_x0t0)
                psi_correlation.append(utils.isotropic_avg('psi correlation',
                                                           self.psi_x, np.conjugate(psi_x0t0), **isotropic_indices))
                n_correlation.append(utils.isotropic_avg('n correlation',
                                                         np.sqrt(self.n(self.psi_x)), np.sqrt(n_x0t0), **isotropic_indices))
                n_avg.append(utils.isotropic_avg('n average',
                                                 self.n(self.psi_x), None, **isotropic_indices))
                deltatheta_full.append(utils.isotropic_avg('deltatheta',
                                                           np.angle(self.psi_x), theta_x0t0, **isotropic_indices))
        return np.array(psi_correlation), np.array(n_correlation), np.array(n_avg), np.exp(1j * np.array(deltatheta_full))
