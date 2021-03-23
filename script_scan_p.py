#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 21 19:51:48 2021

@author: delis
"""

c = 3E2 #μm/ps
hbar = 6.582119569 * 1E2 # μeV ps

from scipy.ndimage import gaussian_filter
from matplotlib import animation
import os
from scipy.fftpack import fft2, ifft2
import numpy as np
import external as ext
import matplotlib.pyplot as pl
pl.rc('font', family='sans-serif')
pl.rc('text', usetex=True)

hatt = 1 # ps
hatx = 1 # μm
hatpsi = 1/hatx # μm^-1
hatrho = 1/hatx**2 # μm^-2
hatepsilon = hbar/hatt # μeV
melectron = 0.510998950 * 1e12 / c**2 # μeV/(μm^2/ps^2)

m_tilde = 3.8e-5
m_dim = m_tilde * melectron
gamma0_tilde = 0.22
gammar_tilde = 0.1 * gamma0_tilde
P_tilde = 50 * gamma0_tilde 
R_tilde = gammar_tilde / 50
gamma2_tilde = 0.02
ns_tilde = gammar_tilde / R_tilde
Kc = hbar**2 / (2 * m_dim * hatepsilon * hatx**2)
Kd = gamma2_tilde / 2
print('Saturation in μm^-2 %.2f' % (ns_tilde * hatrho))
print('Kd = %.3f' % (gamma2_tilde/2))
print('Kc = %.4f' % Kc)

# =============================================================================
# 
# =============================================================================

N = 2**7
L_tilde = 2**7
dx_tilde = L_tilde / N
dkx_tilde = 2 * np.pi / (N * dx_tilde)

'''
def dimensional_units():
    L_dim = L_tilde * hatx                                                      # result in μm
    P_dim = P_tilde * (1/(hatx**2 * hatt))                                      # result in μm^-2 ps^-1
    R_dim = R_tilde * (hatx**2/hatt)                                            # result in μm^2 ps^-1
    gamma0_dim = gamma0_tilde * (1/hatt)                                        # result in ps^-1
    gammar_dim = gammar_tilde * (1/hatt)                                        # result in ps^-1
    gamma2_dim = gamma2_tilde * (hatx**2 / hatt)                                # result in μm^2 ps^-1
    ns_dim = ns_tilde * hatrho                                                  # result in μm^-2
    n0_dim = n0_tilde * hatrho                                                  # result in μm^-2
    nr_dim = nres_tilde * hatrho                                                # result in μm^-2
    return L_dim, P_dim, R_dim, gamma0_dim, gammar_dim, gamma2_dim, ns_dim, n0_dim, nr_dim
'''

def arrays():
    x_0 = - N * dx_tilde / 2
    kx0 = - np.pi / dx_tilde
    x = x_0 + dx_tilde * np.arange(N)
    kx = kx0 + dkx_tilde * np.arange(N)
    return x, kx

x, kx =  arrays()
X, Y = np.meshgrid(x, x)
KX, KY = np.meshgrid(kx, kx)

time_steps = 100000
dt_tilde = 4.5e-2
every = 100
i1 = 0
i2 = time_steps
lengthwindow = i2-i1
t = ext.time(dt_tilde, time_steps, i1, i2, every)

class model:
    def __init__(self, p, g_dim, gr_dim, psi_x=0):
        self.p = p
        self.g_tilde = g_dim * hatrho / hatepsilon
        self.gr_tilde = gr_dim * hatrho / hatepsilon
        self.psi_x = psi_x
        self.psi_x = np.full((N, N), np.sqrt(1 / (2 * dx_tilde**2)) + 0.01)
        self.psi_x /= hatpsi
        self.psi_mod_k = fft2(self.psi_mod_x)
        if self.g_tilde == 0:
            self.uc = 0
        else:
            self.uc =  self.g_tilde * (1 - 2 * self.p * (self.gr_tilde / self.g_tilde) * (gamma0_tilde / gammar_tilde))
        print('p = %.3f, uc = %.5f, tilde g = %.1f, tilde gr = %.3f, TWR = %.3f' % (self.p, self.uc, g_dim, gr_dim, self.g_tilde / (gamma0_tilde * dx_tilde**2)))

    def bogoliubov(self):
        n0_tilde = ns_tilde * (self.p - 1)
        self.rc = 2 * self.p * self.gr_tilde * gamma0_tilde  / R_tilde
        self.rd = gamma0_tilde * (self.p - 1) / 2
        self.ud = gamma0_tilde * self.p / (2 * ns_tilde)
        omsol = self.rc + n0_tilde * self.uc
        a = - omsol + self.Kc * kx ** 2 + self.rc + 2 * n0_tilde * self.uc
        b = - self.Kd * kx ** 2 + self.rd - 2 * n0_tilde * self.ud
        c = n0_tilde * self.uc
        d = - n0_tilde * self.ud
        im_plus = np.zeros(len(kx))
        im_minus = np.zeros(len(kx))
        for i in range(len(kx)):
            if (a[i]**2 - c**2 - d**2) < 0:
                im_plus[i] = b[i] + np.sqrt(np.abs(a[i]**2 - c**2 - d**2))
                im_minus[i] = b[i] - np.sqrt(np.abs(a[i]**2 - c**2 - d**2))
            else:
                im_plus[i] = b[i]
                im_minus[i] = b[i]
        pl.plot(kx, im_plus, 'o', label=r'Imaginary plus')
        pl.plot(kx, im_minus, '^', label=r'Imaginary minus')
        pl.axhline(y=0, xmin=kx[0], xmax=kx[-1], linestyle='--', color='black')
        pl.xlim(0, kx[-1])
        pl.title(r'gr = %.3f' % (self.gr_tilde*hatepsilon/hatrho))
        pl.legend()
        pl.show()
        return im_plus, im_minus

    def _set_fourier_psi_x(self, psi_x):
        self.psi_mod_x = psi_x * np.exp(-1j * KX[0,0] * X - 1j * KY[0,0] * Y) * dx_tilde * dx_tilde / (2 * np.pi)

    def _get_psi_x(self):
        return self.psi_mod_x * np.exp(1j * KX[0,0] * X + 1j * KY[0,0] * Y) * 2 * np.pi / (dx_tilde * dx_tilde)

    def _set_fourier_psi_k(self, psi_k):
        self.psi_mod_k = psi_k * np.exp(1j * X[0,0] * dkx_tilde * np.arange(N) + 1j * Y[0,0] * dkx_tilde * np.arange(N))

    def _get_psi_k(self):
        return self.psi_mod_k * np.exp(-1j * X[0,0] * dkx_tilde * np.arange(N) - 1j * Y[0,0] * dkx_tilde * np.arange(N))

    psi_x = property(_get_psi_x, _set_fourier_psi_x)
    psi_k = property(_get_psi_k, _set_fourier_psi_k)
# =============================================================================
# Definition of the split steps
# =============================================================================
    def n(self):
        return (self.psi_x * np.conjugate(self.psi_x)).real - 1/(2*dx_tilde**2)

    def prefactor_x(self):
        if self.g_tilde == 0:
            self.uc_tilde = 0
        else:
            self.uc_tilde = self.g_tilde * (self.n() + 2 * self.p * (self.gr_tilde / self.g_tilde) * (gamma0_tilde / R_tilde) * (1 / (1 + self.n() / ns_tilde)))
        self.I_tilde = (gamma0_tilde / 2) * (self.p * (1 / (1 + self.n() / ns_tilde)) - 1)
        return np.exp(-1j * 0.5 * dt_tilde * (self.uc_tilde + 1j * self.I_tilde))

    def prefactor_k(self):
        return np.exp(-1j * dt_tilde * (KX ** 2 + KY ** 2) * (Kc - 1j * Kd))

# =============================================================================
# Time evolution
# =============================================================================

    def time_evolution(self):
        np.random.seed()
        n_sum = np.zeros(len(t))
        v = np.zeros(len(t))
        for i in range(time_steps+1):
            self.sigma = gamma0_tilde * (self.p / (1 + self.n() / ns_tilde) + 1) / (4 * dx_tilde**2)
            self.psi_x *= self.prefactor_x()
            self.psi_mod_k = fft2(self.psi_mod_x)
            self.psi_k *= self.prefactor_k()
            self.psi_mod_x = ifft2(self.psi_mod_k)
            self.psi_x *= self.prefactor_x()
            self.psi_x += np.sqrt(dt_tilde) * np.sqrt(self.sigma) * (np.random.normal(0, 1, (N,N)) + 1j * np.random.normal(0, 1, (N,N)))
            if i>=i1 and i<=i2 and i%every==0:
                time_array_index = (i-i1)//every
                #if i % 10000 == 0:
                    #print(i)
                v[time_array_index] = vortices(time_array_index, np.angle(self.psi_x))
                n_sum[time_array_index] = np.mean(self.n())
        g1_x = np.zeros(int(N/2), dtype = complex)
        d1_x = np.zeros(int(N/2))
        for i in range(N):
            g1_x += np.conjugate(self.psi_x[i, int(N/2)]) * self.psi_x[i, int(N/2):] / N
            d1_x += self.n()[i, int(N/2):] / N
        g1_x[0] -= 1/(2*dx_tilde**2)
        return g1_x, d1_x, n_sum, v

def vortices(index, phase):
    count_v = 0
    count_av = 0
    grad = np.gradient(phase, dx_tilde)
    #v_pos = np.zeros((N, N))
    #av_pos = np.zeros((N, N))
    for i in range(1, N-1):
        for j in range(1, N-1):
            loop = (2*dx_tilde*(grad[0][j+1, i+1] - grad[1][j+1, i+1]) +
                    2*dx_tilde*(grad[0][j+1, i-1] + grad[1][j+1, i-1]) +
                    2*dx_tilde*(-grad[0][j-1, i-1] + grad[1][j-1, i-1]) +
                    2*dx_tilde*(-grad[0][j-1, i+1] - grad[1][j-1, i+1]) +
                    2*dx_tilde*(grad[0][j+1, i] + grad[1][j, i-1] - grad[0][j-1, i] - grad[1][j, i+1]))
            if loop >= 2 * np.pi:
                count_v += 1
                #v_pos[i,j] = 1
            elif loop <= - 2 * np.pi:
                count_av +=1
                #av_pos[i,j] = 1
    '''
    xv = np.array([x[i] for i in range(N) for j in range(N) if v_pos[i,j]==1])
    yv = np.array([x[j] for i in range(N) for j in range(N) if v_pos[i,j]==1])
    xav = np.array([x[i] for i in range(N) for j in range(N) if av_pos[i,j]==1])
    yav = np.array([x[j] for i in range(N) for j in range(N) if av_pos[i,j]==1])
    fig,ax = pl.subplots(1,1, figsize=(8,6))
    ax.plot(xv, yv, 'go', markersize=2)
    ax.plot(xav, yav, 'ro', markersize=2)
    ax.set_xlim(x[0], x[-1])
    ax.set_ylim(x[0], x[-1])
    im = ax.pcolormesh(X, Y, phase, vmin = -np.pi, vmax = np.pi, cmap='Greys')
    pl.colorbar(im)
    pl.title(r't = %.2f' % t[index])
    #pl.savefig('/Users/delis/Desktop/vortices' + os.sep + 'fig' + str(count) + '.jpg', format='jpg')
    pl.show()
    '''
    total = count_v + count_av
    return total

def parallel_func(p, g_dim, gr_dim):
    save_subfolder = save_folder + os.sep + 'pump' + '_' + str(np.round(p, 3))
    os.mkdir(save_subfolder)
    gpe = model(p, g_dim, gr_dim)
    g1_x, d1_x, avg, v = gpe.time_evolution()
    np.savetxt(save_subfolder + os.sep + 'g1' + '.dat', (np.abs(g1_x))/np.sqrt(d1_x[0]*d1_x).real)
    np.savetxt(save_subfolder + os.sep + 'avg' + '.dat', avg)
    np.savetxt(save_subfolder + os.sep + 'vortices' + '.dat', v)

# =============================================================================
# Parallel tests
# =============================================================================
from qutip import *
qutip.settings.num_cpus = 7

knob_array = np.array([1.2, 1.3, 1.4, 1.5, 1.6, 1.8, 2])
p_array = knob_array * P_tilde * R_tilde / (gamma0_tilde * gammar_tilde)
gr_dim = 0
g_dim = 0

path_init = r'/Users/delis/Desktop'
#path_init = r'/scratch/konstantinos'
save_folder = path_init + os.sep + 'tests' + '_' + 'Kd' + str(gamma2_tilde/2) + '_' + 'g' + str(g_dim)
#os.mkdir(save_folder)

#parallel_map(parallel_func, p_array, task_kwargs=dict(g_dim = g_dim, gr_dim = gr_dim))

# =============================================================================
#  Test plots
# =============================================================================

fig,ax = pl.subplots(1,1, figsize=(10,10))
ax.set_xscale('log')
ax.set_yscale('log')
for p in p_array:
    save_subfolder = save_folder + os.sep + 'pump' + '_' + str(np.round(p, 3))
    correlator = np.loadtxt(save_subfolder + os.sep + 'g1' + '.dat')
    ax.plot(x[int(N/2):]-x[int(N/2)], correlator, label=r'$p$ = %.3f' % p)
ax.tick_params(axis='both', which='both', direction='in', labelsize=20, pad=12, length=12)
pl.legend(prop=dict(size=20))
pl.tight_layout()
pl.show()

fig,ax = pl.subplots(1,1, figsize=(10,10))
for p in p_array:
    save_subfolder = save_folder + os.sep + 'pump' + '_' + str(np.round(p, 3))
    avgdensity = np.loadtxt(save_subfolder + os.sep + 'avg' + '.dat')
    ax.plot(t, avgdensity, label=r'$p$ = %.3f' % p)
    ax.hlines(y = ns_tilde * (p - 1), xmin=t[0], xmax=t[-1])
ax.tick_params(axis='both', which='both', direction='in', labelsize=20, pad=12, length=12)
ax.legend(prop=dict(size=20))
pl.tight_layout()
pl.show()


fig,ax = pl.subplots(1,1, figsize=(10, 10))
for p in p_array:
    save_subfolder = save_folder  + os.sep + 'pump' + '_' + str(np.round(p, 3))
    vort = np.loadtxt(save_subfolder + os.sep + 'vortices' + '.dat')
    ax.plot(t, vort/N**2, label=r'$p$ = %.3f' % p)
ax.tick_params(axis='both', which='both', direction='in', labelsize=20, pad=12, length=12)
ax.set_xlabel('$t$', fontsize = 20)
ax.set_ylabel(r'$n_v$', fontsize = 20)
ax.legend(prop=dict(size=20))
pl.tight_layout()
#pl.savefig('/Users/delis/Desktop/vortices' + 'Kd' + str(gamma2_tilde/2) + 'g' + str(g_dim) + '_01sigma' + '.eps', format='eps')
pl.show()