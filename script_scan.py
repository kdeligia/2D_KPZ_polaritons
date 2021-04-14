#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 21 19:51:48 2021

@author: delis
"""

c = 3e2 #μm/ps
hbar = 6.582119569 * 1e2 # μeV ps

#from scipy.ndimage import gaussian_filter
import os
from scipy.fftpack import fft2, ifft2
import numpy as np
import external as ext

hatt = 1 # ps
hatx = 1 # μm
hatpsi = 1/hatx # μm^-1
hatrho = 1/hatx**2 # μm^-2
hatepsilon = hbar/hatt # μeV
melectron = 0.510998950 * 1e12 / c**2 # μeV/(μm^2/ps^2)

m_tilde = 3.8e-5 * 3
m_dim = m_tilde * melectron
gamma0_tilde = 0.2 * 100
gammar_tilde = gamma0_tilde * 0.1
P_tilde = gamma0_tilde * 1
R_tilde = gammar_tilde / 1
ns_tilde = gammar_tilde / R_tilde
Kc = hbar**2 / (2 * m_dim * hatepsilon * hatx**2)

print('ns = %.i' % (ns_tilde * hatrho))
print('Kc = %.4f' % Kc)

# =============================================================================
# 
# =============================================================================
N = 2 ** 6
L_tilde = 2 ** 6
dx_tilde = 0.5

N_steps = 500000
dt_tilde = 1e-2
every = 500
i1 = 20000
i2 = N_steps
lengthwindow = i2-i1

t = ext.time(dt_tilde, N_steps, i1, i2, every)
x, kx =  ext.space_momentum(N, dx_tilde)
isotropic_indices = ext.get_indices(x)
X, Y = np.meshgrid(x, x)
KX, KY = np.meshgrid(kx, kx)

class model:
    def __init__(self, p, sigma, om_tilde, g_dim, gr_dim, psi_x=0):
        self.dkx_tilde = kx[1] - kx[0]
        self.sigma = sigma
        self.om_tilde = om_tilde
        self.p = p
        self.g_tilde = g_dim * hatrho / hatepsilon
        self.gr_tilde = gr_dim * hatrho / hatepsilon
        self.damp = 1 + 2 * self.p * self.gr_tilde / (R_tilde * self.om_tilde) + self.p / (2 * self.om_tilde) * 1j 
        self.psi_x = psi_x
        self.psi_x = np.full((N, N), 0.01**(1/2))
        self.psi_x /= hatpsi
        self.psi_mod_k = fft2(self.psi_mod_x)
        print('sigma = %.4f, omega = %.i, p = %.3f, Kd = %.5f, TWR = %.3f' % (self.sigma, self.om_tilde, self.p, (Kc/self.damp).imag, self.g_tilde / (gamma0_tilde * dx_tilde**2)))
        #self.bogoliubov_lin()

    def bogoliubov_lin(self):
        n0_tilde = ns_tilde * (self.p - 1)
        if self.g_tilde == 0:
            self.uc = 0
        else:
            self.uc =  self.g_tilde * (1 - 2 * self.p * (self.gr_tilde / self.g_tilde) * (gamma0_tilde / gammar_tilde))
        Gamma_ef = gamma0_tilde * (self.p - 1) / (2 * self.p)
        mu = self.uc * n0_tilde
        kin = Kc * kx**2
        gam_a = (1/self.damp).real
        gam_b = (1/self.damp).imag
        im_plus = np.zeros(len(kx))
        im_minus = np.zeros(len(kx))
        for i in range(len(kx)):
            if -(Gamma_ef * gam_a) ** 2 - (gam_b * mu) ** 2 + 2 * Gamma_ef * gam_a * gam_b * (kin[i] + mu) + gam_a ** 2 * kin[i] * (kin[i] + 2 * mu) < 0:
                im_plus[i] = - (gam_a * Gamma_ef - gam_b * (kin[i] + mu)) + np.sqrt(np.abs(-(Gamma_ef * gam_a) ** 2 - (gam_b * mu) ** 2 + 2 * Gamma_ef * gam_a * gam_b * (kin[i] + mu) + gam_a ** 2 * kin[i] * (kin[i] + 2 * mu)))
                im_minus[i] = - (gam_a * Gamma_ef - gam_b * (kin[i] + mu)) - np.sqrt(np.abs(-(Gamma_ef * gam_a) ** 2 - (gam_b * mu) ** 2 + 2 * Gamma_ef * gam_a * gam_b * (kin[i] + mu) + gam_a ** 2 * kin[i] * (kin[i] + 2 * mu)))
            else:
                im_plus[i] = - (gam_a * Gamma_ef - gam_b * (kin[i] + mu))
                im_minus[i] = - (gam_a * Gamma_ef - gam_b * (kin[i] + mu))
        pl.plot(kx, im_plus, 'o', label=r'Im plus')
        pl.plot(kx, im_minus, '^', label=r'Im minus')
        pl.axhline(y=0, xmin=kx[0], xmax=kx[-1], linestyle='--', color='black')
        pl.xlim(0, kx[-1])
        pl.title(r'$g$ = %.i, $gr$ = %.i, $n_s$ = %.i, $p$ = %.1f, $\Omega$ = %.i, $m$ = %.2e' % (g_dim, gr_dim, ns_tilde, self.p, self.om_tilde, m_tilde))
        pl.legend()
        pl.show()
        print('Lowest mode achievable k = %.5f' % (2*np.pi/L_tilde))
        print('Im plus near it', im_plus[N//2+1])
        return im_plus, im_minus

    def _set_fourier_psi_x(self, psi_x):
        self.psi_mod_x = psi_x * np.exp(-1j * KX[0,0] * X - 1j * KY[0,0] * Y) * dx_tilde * dx_tilde / (2 * np.pi)

    def _get_psi_x(self):
        return self.psi_mod_x * np.exp(1j * KX[0,0] * X + 1j * KY[0,0] * Y) * 2 * np.pi / (dx_tilde * dx_tilde)

    def _set_fourier_psi_k(self, psi_k):
        self.psi_mod_k = psi_k * np.exp(1j * X[0,0] * self.dkx_tilde * np.arange(N) + 1j * Y[0,0] * self.dkx_tilde * np.arange(N))

    def _get_psi_k(self):
        return self.psi_mod_k * np.exp(-1j * X[0,0] * self.dkx_tilde * np.arange(N) - 1j * Y[0,0] * self.dkx_tilde * np.arange(N))

    psi_x = property(_get_psi_x, _set_fourier_psi_x)
    psi_k = property(_get_psi_k, _set_fourier_psi_k)
# =============================================================================
# Definition of the split steps
# =============================================================================
    def n(self, psi):
        return (psi * np.conjugate(psi)).real

    def prefactor_x(self, n):
        if self.g_tilde == 0:
            self.uc_tilde = 0
        else:
            self.uc_tilde = self.g_tilde * (n + 2 * self.p * (self.gr_tilde / self.g_tilde) * (gamma0_tilde / R_tilde) * (1 / (1 + n / ns_tilde)))
        self.rd_tilde = (gamma0_tilde / 2) * (self.p / (1 + n / ns_tilde) - 1)
        return np.exp(-1j * 0.5 * dt_tilde * (self.uc_tilde + 1j * self.rd_tilde) / self.damp)

    def prefactor_k(self):
        return np.exp(-1j * dt_tilde * (KX ** 2 + KY ** 2) * Kc / self.damp)

# =============================================================================
# Time evolution
# =============================================================================
    def time_evolution(self):
        np.random.seed()
        density = np.zeros(len(t))
        v = np.zeros(len(t))
        for i in range(N_steps):
            self.psi_x *= self.prefactor_x(self.n(self.psi_x))
            self.psi_mod_k = fft2(self.psi_mod_x)
            self.psi_k *= self.prefactor_k()
            self.psi_mod_x = ifft2(self.psi_mod_k)
            self.psi_x *= self.prefactor_x(self.n(self.psi_x))
            self.psi_x += np.sqrt(dt_tilde) * np.sqrt(self.sigma / dx_tilde ** 2) * (np.random.normal(0, 1, (N,N)) + 1j * np.random.normal(0, 1, (N,N))) / self.damp
            if i>=i1 and i<=i2 and i%every==0:
                time_array_index = (i-i1)//every
                v[time_array_index] = ext.vortices(time_array_index, dx_tilde, N, np.angle(self.psi_x))
                density[time_array_index] = np.mean(self.n(self.psi_x))
        psi_correlation_x = ext.isotropic_avg(self.psi_x, 'psi correlation', **isotropic_indices)
        n_avg_x = ext.isotropic_avg(self.n(self.psi_x), 'density average', **isotropic_indices)
        return psi_correlation_x, n_avg_x.real, density, v

def g1(p, sigma, om_tilde, g_dim, gr_dim):
    gpe = model(p, sigma, om_tilde, g_dim, gr_dim)
    psi_correlation_x, n_avg_x, n, v = gpe.time_evolution()
    np.savetxt(subfolders[str(p), str(sigma)] + os.sep + 'g1_x' + '.dat', np.abs(psi_correlation_x)/np.sqrt(n_avg_x[0]*n_avg_x))
    np.savetxt(subfolders[str(p), str(sigma)] + os.sep + 'n' + '.dat', n)
    np.savetxt(subfolders[str(p), str(sigma)] + os.sep + 'vortices' + '.dat', v)

# =============================================================================
# Parallel tests
# =============================================================================
from qutip import *
import matplotlib.pyplot as pl
pl.rc('font', family='sans-serif')
pl.rc('text', usetex=True)

qutip.settings.num_cpus = 4

sigma_array = np.array([1e-2])
p_knob_array = np.array([1.5, 2, 3, 5])
om_knob_array = np.array([1e9])
p_array = p_knob_array * P_tilde * R_tilde / (gamma0_tilde * gammar_tilde)
gr_dim = 0
g_dim = 0

path = r'/Users/delis/Desktop'
final_save = path

def create_subfolders(sigma_array, p_array):
    save_folder = path + os.sep + 'g' + str(g_dim)+ '_' + 'gr' + str(gr_dim) + '_' + 'gamma' + str(gamma0_tilde)
    #os.mkdir(save_folder)
    subfolders = {}
    for sigma in sigma_array:
        for p in p_array:
            subfolders[str(p), str(sigma)] = save_folder + os.sep + 'sigma' + str(sigma) + '_' + 'p' + str(p) + '_' + 'om' + str(int(om_knob_array[0]))
            #os.mkdir(subfolders[str(p), str(sigma)])
    return subfolders

subfolders = create_subfolders(sigma_array, p_array)

#for sigma in sigma_array:
#    parallel_map(g1, p_array, task_kwargs=dict(sigma = sigma, om_tilde = om_knob_array[0], g_dim = g_dim, gr_dim = gr_dim))
# =============================================================================
# Plots
# =============================================================================
dr = np.arange(N//2) * dx_tilde

ji = np.loadtxt('/Users/delis/Desktop/ji exact/64x64.txt')
fig,ax = pl.subplots(1,1, figsize=(10,10))
ax.set_xscale('log')
ax.set_yscale('log')
for sigma in sigma_array:
    for p in p_array:
        g1_x = np.loadtxt(subfolders[str(p), str(sigma)] + os.sep + 'g1_x' + '.dat')
        ax.plot(dr, g1_x, label=r'sigma = %.2f, p = %.1f' % (sigma, p))
ax.plot(ji[:, 0], ji[:,1])
ax.tick_params(axis='both', which='both', direction='in', labelsize=20, pad=12, length=12)
ax.legend(prop=dict(size=20))
pl.title('sigma = %.4f, om = %.i, gamma0 = %.3f' % (sigma_array[0], om_knob_array[0], gamma0_tilde), fontsize = 20)
pl.tight_layout()
pl.show()

fig,ax = pl.subplots(1,1, figsize=(10,10))
for sigma in sigma_array:
    for p in p_array:
        vort = np.loadtxt(subfolders[str(p), str(sigma)] + os.sep + 'vortices' + '.dat')
        ax.plot(t, vort/N**2, label=r'sigma = %.2f, p = %.1f' % (sigma, p))
ax.tick_params(axis='both', which='both', direction='in', labelsize=20, pad=12, length=12)
ax.legend(prop=dict(size=20))
pl.tight_layout()
pl.show()

fig,ax = pl.subplots(1,1, figsize=(10,10))
for sigma in sigma_array:
    for p in p_array:
        avg = np.loadtxt(subfolders[str(p), str(sigma)] + os.sep + 'n' + '.dat')
        ax.plot(t, avg, label=r'sigma = %.2f, p = %.1f' % (sigma, p))
        ax.hlines(y=(p-1)*ns_tilde, xmin=t[0], xmax=t[-1], color='black')
ax.tick_params(axis='both', which='both', direction='in', labelsize=20, pad=12, length=12)
ax.legend(prop=dict(size=20))
pl.tight_layout()
pl.show()

'''
print('--- Alternative Parametrization ---')
print('Kd = %.6f' % (m_dim * gamma2_tilde * (hatx**2 / hatt) / hbar))
print('nu = ', (p_array - 1))
print('c = %.6f' % (hbar * gamma0_tilde * (1/hatt) / (2 * g_dim * ns_tilde * hatrho)))
'''