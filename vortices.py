#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 19 13:53:23 2021

@author: delis
"""

c = 3e2 #μm/ps
hbar = 6.582119569 * 1e2 # μeV ps

import os
import numpy as np
import external as ext
from scipy.fftpack import fft2, ifft2
import matplotlib.pyplot as pl
from matplotlib import rc
from matplotlib.texmanager import TexManager
import re
pl.rc('font', family='sans-serif')
pl.rc('text', usetex=True)
pl.close('all')
#import pyfftw
#pyfftw.interfaces.cache.enable()

hatt = 1 # ps
hatx = 1 # μm
hatpsi = 1/hatx # μm^-1
hatrho = 1/hatx**2 # μm^-2
hatepsilon = hbar/hatt # μeV
melectron = 0.510998950 * 1e12 / c**2 # μeV/(μm^2/ps^2)

m_tilde = 1.14e-4
m_dim = m_tilde * melectron
Kc = hbar ** 2 / (2 * m_dim * hatepsilon * hatx**2)

# =============================================================================
# 
# =============================================================================
N = 2 ** 7
dx_tilde = 0.5

N_steps = 100000
dt_tilde = 5e-3
every = 200
i1 = 0
i2 = N_steps
lengthwindow = i2-i1
t = ext.time(dt_tilde, N_steps, i1, i2, every)

x, y = ext.space_momentum(N, dx_tilde)
isotropic_indices = ext.get_indices(x)

kx = (2 * np.pi) / dx_tilde * np.fft.fftfreq(N, d = 1)
ky = (2 * np.pi) / dx_tilde * np.fft.fftfreq(N, d = 1)

class model:
    def __init__(self, p, sigma, gamma2, gamma0, g, gr, ns):
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
    def time_evolution(self, folder):
        np.random.seed()
        a_vort = 2 * dx_tilde
        vortex_number = np.zeros(len(t))
        density = np.zeros(len(t))
        for i in range(N_steps):
            self.psi_x *= self.exp_x( 0.5 * dt_tilde, self.n(self.psi_x))
            psi_k = fft2(self.psi_x)
            psi_k *= self.exp_k(dt_tilde)
            self.psi_x = ifft2(psi_k)
            self.psi_x *= self.exp_x( 0.5 * dt_tilde, self.n(self.psi_x))
            self.psi_x += np.sqrt(dt_tilde) * np.sqrt(self.sigma / dx_tilde ** 2) * (np.random.normal(0, 1, (N,N)) + 1j * np.random.normal(0, 1, (N,N)))
            if i >= i1 and i <= i2 and i % every == 0:
                time_index = (i - i1) // every
                vortex_positions_field, ignore = ext.vortices(a_vort, np.angle(self.psi_x), x, y)
                ext.vortex_plots(folder, x, t, time_index, vortex_positions_field, np.angle(self.psi_x), self.n(self.psi_x))
                vortex_number[time_index] = len(np.where(vortex_positions_field == 1)[0]) + len(np.where(vortex_positions_field == -1)[0])
                density[time_index] = np.mean(self.n(self.psi_x))
        return vortex_number, density

# =============================================================================
# 
# =============================================================================
from qutip import *
n_batch = 4
qutip.settings.num_cpus = n_batch

sigma_array = np.array([2e-2, 3e-2, 4e-2])
p_array = np.array([1.6, 1.8])
gamma2_array = np.array([5e-1, 1e-1, 5e-2, 1e-2])
gamma0_array = np.array([16, 18])
gr = 0
g = 0
ns = 80

path_remote = r'/scratch/konstantinos'
path_local = r'/Users/delis/Desktop'

init = path_remote + os.sep + 'ns' + str(int(ns)) + '_' + 'g' + str(g)
os.mkdir(init)

def vortices(gamma2, gamma0, p, sigma):
    id_string = 'sigma' + str(sigma) + '_' + 'p' + str(p) + '_' + 'gammak' + str(gamma2) + '_' + 'gamma' + str(gamma0)
    os.mkdir(init + os.sep + id_string)
    gpe = model(p, sigma, gamma2, gamma0, g = g, gr = gr, ns = ns)
    nv, n = gpe.time_evolution(init + os.sep + id_string)
    np.savetxt(init + os.sep + id_string + '_' + 'nv' + '.dat', nv)
    np.savetxt(init + os.sep + id_string + '_' + 'dens' + '.dat', n)
    return None


def call_avg():
    print('Parallel in gamma2 = ', gamma2_array)
    for gamma0 in gamma0_array:
        for p in p_array:
            for sigma in sigma_array:
                print('Auxiliary parameters: gamma0 = %.i, p = %.1f, sigma = %.2f' % (gamma0, p, sigma))
                parallel_map(vortices, gamma2_array, task_kwargs=dict(gamma0 = gamma0, p = p, sigma = sigma))
                print('Done!')
call_avg()

'''
fig, ax = pl.subplots(1,1, figsize=(8, 6))
for file in os.listdir(init):
    if 'nv.dat' in file:
        s = [float(s) for s in re.findall(r'-?\d+\.?\d*', file)]
        if s[3] in gamma0_array and s[2] in gamma2_array:
            ax.plot(t, np.loadtxt(init + os.sep + file), label=r'$p$ = %.1f, $\sigma$ = %.2f, $\gamma_2$ = %.i, $\gamma_0$ = %.2f' % (s[1], s[0], int(s[2]), s[3]))
ax.tick_params(axis='both', which='both', direction='in', labelsize=16, pad=12, length=12)
ax.legend(prop=dict(size=12))
ax.set_xlabel(r'$t$', fontsize=20)
ax.set_ylabel(r'$n_v$', fontsize=20)
pl.show()

fig, ax = pl.subplots(1,1, figsize=(8, 6))
for file in os.listdir(init):
    if 'dens.dat' in file:
        s = [float(s) for s in re.findall(r'-?\d+\.?\d*', file)]
        if s[3] in gamma0_array and s[2] in gamma2_array:
            ax.plot(t, np.loadtxt(init + os.sep + file), label=r'$p$ = %.1f, $\sigma$ = %.2f, $\gamma_2$ = %.i, $\gamma_0$ = %.2f' % (s[1], s[0], int(s[2]), s[3]))
ax.hlines(y=ns * (p_array[0] - 1), xmin=t[0], xmax=t[-1], color='red')
ax.tick_params(axis='both', which='both', direction='in', labelsize=16, pad=12, length=12)
ax.legend(prop=dict(size=12))
ax.set_xlabel(r'$t$', fontsize=20)
ax.set_ylabel(r'$n$', fontsize=20)
pl.show()
'''