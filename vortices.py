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
import re
'''
from matplotlib import rc
from matplotlib.texmanager import TexManager
import re
pl.rc('font', family='sans-serif')
pl.rc('text', usetex=True)
pl.close('all')
'''
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

N_steps = 400000
dt_tilde = 1e-2
every = 500
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
n_batch = 2
qutip.settings.num_cpus = n_batch

sigma_array = np.array([1e-2])
p_array = np.array([2])
gamma2_array = np.array([0.1, 0.1, 0.1])
gamma0_array = np.array([0.1, 0.2, 2])

gr = 0
g = 0
ns = 1

path_remote = r'/scratch/konstantinos'
path_local = r'/Users/delis/Desktop'

init = path_local + os.sep + 'sigma' + str(sigma_array[0]) + '_' + 'p' + str(p_array[0]) + '_' + 'ns' + str(int(ns)) + '_' + 'g' + str(int(g))
os.mkdir(init)

def vortices(gamma0, gamma2, p, sigma):
    id_string = 'gammak' + str(gamma2) + '_' + 'gamma' + str(gamma0)
    os.mkdir(init + os.sep + id_string)
    gpe = model(p, sigma, gamma2, gamma0, g = g, gr = gr, ns = ns)
    nv, n = gpe.time_evolution(init + os.sep + id_string)
    np.savetxt(init + os.sep + id_string + '_' + 'nv' + '.dat', nv)
    np.savetxt(init + os.sep + id_string + '_' + 'dens' + '.dat', n)
    return None

def call_avg():
    parfor(vortices, gamma0_array, gamma2_array, p = p_array[0], sigma = sigma_array[0])
call_avg()

'''
from mpl_toolkits import mplot3d
fig = pl.figure(figsize=(8, 6))
ax = pl.axes(projection ='3d')

gamma0_mesh, gamma2_mesh = np.meshgrid(gamma0_array, gamma2_array)
P = np.ones_like(gamma0_mesh)
for p in p_array:
    ax.plot_surface(gamma0_mesh, gamma2_mesh, p * P)
ax.scatter(gamma0_array[2], gamma2_array[1], p_array[1], 'o', color='red', s = 100)
ax.set_title(r'Phase diagram: $\sigma$ = %.2f, $n_s$ = %.i' % (sigma_array[0], ns))
ax.set_xlabel(r'$\gamma_0$', fontsize = 16)
ax.set_ylabel(r'$\gamma_2$', fontsize = 16)
ax.set_zlabel(r'$p$', fontsize = 16)
pl.show()
'''


fig, ax = pl.subplots(1,1, figsize=(8, 6))
for file in os.listdir(init):
    if 'nv.dat' in file:
        s = [float(s) for s in re.findall(r'-?\d+\.?\d*', file)]
        ax.plot(t, np.loadtxt(init + os.sep + file), label=r'$\gamma_2$ = %.e, $\gamma_0$ = %.1f' % (s[0], s[1]))
ax.tick_params(axis='both', which='both', direction='in', labelsize=16, pad=12, length=12)
ax.legend(prop=dict(size=12))
ax.set_xlabel(r'$t$', fontsize=20)
ax.set_ylabel(r'$n_v$', fontsize=20)
pl.tight_layout()
pl.show()


fig, ax = pl.subplots(1,1, figsize=(8, 6))
for file in os.listdir(init):
    if 'dens.dat' in file:
        s = [float(s) for s in re.findall(r'-?\d+\.?\d*', file)]
        ax.plot(t, np.loadtxt(init + os.sep + file), label=r'$\gamma_2$ = %.e, $\gamma_0$ = %.1f' % (s[0], s[1]))
ax.hlines(y=ns * (p_array[0] - 1), xmin=t[0], xmax=t[-1], color='black')
ax.tick_params(axis='both', which='both', direction='in', labelsize=16, pad=12, length=12)
ax.legend(prop=dict(size=12))
ax.set_xlabel(r'$t$', fontsize=20)
ax.set_ylabel(r'$n$', fontsize=20)
ax.set_ylim(0, 2)
pl.tight_layout()
pl.show()