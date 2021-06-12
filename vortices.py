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

m_tilde = 5e-5
m_dim = m_tilde * melectron
Kc = hbar ** 2 / (2 * m_dim * hatepsilon * hatx**2)

# =============================================================================
# 
# =============================================================================
N = 2 ** 7
dx_tilde = 0.5

N_steps = 50000
dt_tilde = 1e-2
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

        '''
        Pth = self.gamma0_tilde * self.ns_tilde
        print(self.P_tilde, Pth)
        '''

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
        noise = self.sigma
        density = np.zeros(len(t))
        for i in range(N_steps):
            #if i == 0:
                #noise = 0
            self.psi_x *= self.exp_x(0.5 * dt_tilde, self.n(self.psi_x))
            psi_k = fft2(self.psi_x)
            psi_k *= self.exp_k(dt_tilde)
            self.psi_x = ifft2(psi_k)
            self.psi_x *= self.exp_x(0.5 * dt_tilde, self.n(self.psi_x))
            self.psi_x += np.sqrt(dt_tilde) * np.sqrt(noise / dx_tilde ** 2) * (np.random.normal(0, 1, (N,N)) + 1j * np.random.normal(0, 1, (N,N)))
            if i >= i1 and i <= i2 and i % every == 0:
                time_index = (i - i1) // every
                #if time_index == 400:
                    #noise = self.sigma2
                #if time_index == 900:
                    #noise = 0
                vortex_positions, ignore = ext.vortex_positions(a_vort, np.angle(self.psi_x), x, y)
                #ext.vortex_plots(folder, x, t, time_index, vortex_positions, np.angle(self.psi_x), self.n(self.psi_x))
                vortex_number[time_index] = len(np.where(vortex_positions == 1)[0]) + len(np.where(vortex_positions == -1)[0])
                density[time_index] = np.mean(self.n(self.psi_x))
        return vortex_number, density

# =============================================================================
# 
# =============================================================================
from qutip import *
n_batch = 3
qutip.settings.num_cpus = n_batch

p_array = np.array([1.4])
gamma2_array = np.array([0.2, 0.2, 0.2])
gamma0_array = np.array([0.15, 0.2, 0.25])
sigma_array = np.array([0.06, 0.08, 0.18, 0.24, 0.3])

gr = 0
g_array = np.array([0.05, 0.1, 0.2])
ns = 100
sigma_th = gamma0_array * (p_array + 1) / 2
#xi = hbar / np.sqrt(2 * m_dim * g * ns * (p_array[0] - 1))

path_remote = r'/scratch/konstantinos'
save_remote = r'/home6/konstantinos'
path_local = r'/Users/delis/Desktop'

def vortices(gamma0, gamma2, p, sigma, g):
    print(r'--- Parameters in parallel: (gamma0, gamma2) = (%.2f, %.2f)' % (gamma0, gamma2))
    gpe = model(p, sigma, gamma2, gamma0, g = g, gr = gr, ns = ns)
    parallel_string = 'gamma' + str(gamma0) + '_' + 'gammak' + str(gamma2)
    # os.mkdir(path + os.sep + parallel_string)
    nvort, dens = gpe.time_evolution(path + os.sep + parallel_string)
    np.savetxt(path + os.sep + parallel_string + '_' + 'nv' + '.dat', nvort)
    np.savetxt(path + os.sep + parallel_string + '_' + 'dens' + '.dat', dens)
    '''
    np.savetxt(init + os.sep + id_string + '_' + 'dens' + '.dat', n)
    os.system(
        'ffmpeg -framerate 10 -i ' + 
        init + os.sep + id_string + os.sep + 
        'fig%d.jpg -vcodec mpeg4 -y ' + 
        save_remote + os.sep + 
        id_string + '.mp4')
    '''
    return None

for p in p_array:
    for g in g_array:
        mu = g * ns * (p - 1)
        for sigma in sigma_array:
            iteration_string = 'sigma' + str(sigma) + '_' + 'p' + str(p) + '_' + 'mu' + str(int(mu))
            path = path_remote + os.sep + iteration_string
            try:
                os.mkdir(path)
            except FileExistsError:
                continue
            print(r'--- Starting for (p, $\sigma$, g) = (%.1f, %.2f, %.2f)' % (p, sigma, g))
            parfor(vortices, gamma0_array, gamma2_array, p = p, sigma = sigma, g = g)
            fig, ax = pl.subplots(1,1, figsize=(8, 6))
            for file in os.listdir(path):
                if 'nv.dat' in file:
                    s = [float(s) for s in re.findall(r'-?\d+\.?\d*', file)]
                    ax.plot(t, np.loadtxt(path + os.sep + file), label=r'$\gamma_0$ = %.2f, $\gamma_2$ = %.2f' % (s[0], s[1]))
            ax.tick_params(axis='both', which='both', direction='in', labelsize=16, pad=12, length=12)
            ax.legend(prop=dict(size=12))
            ax.set_xlabel(r'$t$', fontsize=20)
            ax.set_ylabel(r'$n_v$', fontsize=20)
            ax.set_title(r'$\sigma$ = %.2f, $p$ = %.1f, $g$ = %.2f, $n_s$ = %.i' % (sigma, p, g, ns), fontsize=20)
            pl.tight_layout()
            pl.savefig(save_remote + os.sep + iteration_string + '___' + 'vortices.jpg', format='jpg')
            pl.show()
            fig, ax = pl.subplots(1,1, figsize=(8, 6))
            for file in os.listdir(path):
                if 'dens.dat' in file:
                    s = [float(s) for s in re.findall(r'-?\d+\.?\d*', file)]
                    ax.plot(t, np.loadtxt(path + os.sep + file), label=r'$\gamma_0$ = %.2f, $\gamma_2$ = %.2f' % (s[0], s[1]))
            ax.hlines(y=ns * (p_array[0] - 1), xmin=t[0], xmax=t[-1], color='black')
            ax.tick_params(axis='both', which='both', direction='in', labelsize=16, pad=12, length=12)
            ax.legend(prop=dict(size=12))
            ax.set_xlabel(r'$t$', fontsize=20)
            ax.set_ylabel(r'$n$', fontsize=20)
            ax.set_title(r'$\sigma$ = %.2f, $p$ = %.1f, $g$ = %.2f, $n_s = %.i' % (sigma, p, g, ns), fontsize=20)
            pl.tight_layout()
            pl.savefig(save_remote + os.sep + iteration_string + '___' + 'density.jpg', format='jpg')
            pl.show()