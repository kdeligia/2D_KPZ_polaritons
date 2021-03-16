#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 21 19:51:48 2021

@author: delis
"""

c = 3E2 #μm/ps
hbar = 6.582119569 * 1E2 # μeV ps

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

m_tilde = -3.2e-5
gamma0_tilde = 0.22
gammar_tilde = 0.1 * gamma0_tilde
gamma2_tilde = 0.06
P_tilde = 23.1 * 0.1 * 1.5
R_tilde = gammar_tilde / 10
p = P_tilde * R_tilde / (gamma0_tilde * gammar_tilde)

ns_tilde = gammar_tilde / R_tilde
n0_tilde = ns_tilde * (p - 1)
nres_tilde = P_tilde / gammar_tilde

# =============================================================================
# Positive mass tests
# =============================================================================
'''
m_tilde = 6.2e-5
gamma0_tilde = 0.22
gammar_tilde = 0.1 * gamma0_tilde
gamma2_tilde = 0.04
P_tilde = 5.153 * 22.2
R_tilde = gammar_tilde / 500
p = P_tilde * R_tilde / (gamma0_tilde * gammar_tilde)

ns_tilde = gammar_tilde / R_tilde
n0_tilde = ns_tilde * (p - 1)
nres_tilde = P_tilde / (gammar_tilde * (1 + n0_tilde/ns_tilde))

mu_res = 0 # μeV
mu_cond = 136.3 # μeV
'''
# =============================================================================
# 
# =============================================================================

N = 2**7
L_tilde = 2**7
dx_tilde = L_tilde / N
dkx_tilde = 2 * np.pi / (N * dx_tilde)

def dimensional_units():
    L_dim = L_tilde * hatx                                                      # result in μm
    P_dim = P_tilde * (1/(hatx**2 * hatt))                                      # result in μm^-2 ps^-1
    R_dim = R_tilde * (hatx**2/hatt)                                            # result in μm^2 ps^-1
    gamma0_dim = gamma0_tilde * (1/hatt)                                        # result in ps^-1
    gammar_dim = gammar_tilde * (1/hatt)                                        # result in ps^-1
    gamma2_dim = gamma2_tilde * (hatx**2 / hatt)                                # result in μm^2 ps^-1
    ns_dim = ns_tilde * hatrho                                                  # result in μm^-2
    m_dim = m_tilde * melectron
    n0_dim = n0_tilde * hatrho                                                  # result in μm^-2
    nr_dim = nres_tilde * hatrho                                                # result in μm^-2
    return L_dim, P_dim, R_dim, gamma0_dim, gammar_dim, gamma2_dim, ns_dim, m_dim, n0_dim, nr_dim

def arrays():
    x_0 = - N * dx_tilde / 2
    kx0 = - np.pi / dx_tilde
    x = x_0 + dx_tilde * np.arange(N)
    kx = kx0 + dkx_tilde * np.arange(N)
    return x, kx

L_dim, P_dim, R_dim, gamma0_dim, gammar_dim, gamma2_dim, ns_dim, m_dim, n0_dim, nr_dim = dimensional_units()
x, kx =  arrays()
X, Y = np.meshgrid(x, x)
KX, KY = np.meshgrid(kx, kx)

time_steps = 200000
dt_tilde = 2e-2
every = 100
i1 = 0
i2 = time_steps
lengthwindow = i2-i1
t = ext.time(dt_tilde, time_steps, i1, i2, every)

print('--- Densities ---')
print('Saturation in μm^-2 %.2f' % ns_dim)
print('Steady-state in μm^-2 %.2f' % n0_dim)
print('Reservoir in μm^-2 %.2f' % nr_dim)
print('--- Dimensionless pump ---')
print('p %.4f' % p)

class model:
    def __init__(self, gr_dim, g_dim, psi_x=0):
        self.g_tilde = g_dim * hatrho / hatepsilon
        self.gr_tilde = gr_dim * hatrho / hatepsilon
        self.sigma = gamma0_tilde * (p + 1) / (4 * dx_tilde**2) * 0.5
        self.psi_x = psi_x
        self.psi_x = np.full((N, N), np.sqrt(1 / (2 * dx_tilde**2)) + 0.1)
        self.psi_x /= hatpsi
        self.psi_mod_k = fft2(self.psi_mod_x)
        self.Kc = hbar**2 / (2 * m_dim * hatepsilon * hatx**2)
        self.Kd = gamma2_tilde / 2
        self.uc =  self.g_tilde * (1 - 2 * p * (self.gr_tilde / self.g_tilde) * (gamma0_tilde / gammar_tilde))
        print('uc = %.5f, Kc = %.3f, Kd = %.3f, tilde g = %.2f, tilde gr = %.2f, sigma = %.2f, TWR = %.3f' % (self.uc, self.Kc, self.Kd, g_dim, gr_dim, self.sigma, self.g_tilde / (gamma0_tilde * dx_tilde**2)))
        #self.bogoliubov()

    def bogoliubov(self):
        self.rc = 2 * p * self.gr_tilde * gamma0_tilde  / R_tilde
        self.rd = gamma0_tilde * (p - 1) / 2
        self.ud = gamma0_tilde * p / (2 * ns_tilde)
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
        return (self.psi_x * np.conjugate(self.psi_x)).real - 1/(2 * dx_tilde**2)

    def prefactor_x(self):
        self.uc_tilde = self.g_tilde * (self.n() + 2 * (self.gr_tilde / self.g_tilde) * (P_tilde / gammar_tilde) * (1 / (1 + self.n() / ns_tilde)))
        self.I_tilde = (gamma0_tilde / 2) * (p * (1 / (1 + self.n() / ns_tilde)) - 1)
        return np.exp(-1j * 0.5 * dt_tilde * (self.uc_tilde + 1j * self.I_tilde))

    def prefactor_k(self):
        return np.exp(-1j * dt_tilde * (KX ** 2 + KY ** 2) * (self.Kc - 1j * self.Kd))

# =============================================================================
# Time evolution
# =============================================================================

    def vortices(self, count):
        count_v = 0
        count_av = 0
        theta = np.angle(self.psi_x)
        grad = np.gradient(theta, dx_tilde)
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
        im = ax.pcolormesh(X, Y, theta, vmin = -np.pi, vmax = np.pi, cmap='Greys')
        pl.colorbar(im)
        pl.title(r't = %.2f' % t[count])
        pl.savefig('/Users/delis/Desktop/vortices' + os.sep + 'fig' + str(count) + '.jpg', format='jpg')
        #pl.show()
        '''
        total = count_v + count_av
        return total

    def time_evolution(self):
        np.random.seed()
        n_sum = np.zeros(len(t))
        v = np.zeros(len(t))
        for i in range(time_steps+1):
            self.psi_x *= self.prefactor_x()
            self.psi_mod_k = fft2(self.psi_mod_x)
            self.psi_k *= self.prefactor_k()
            self.psi_mod_x = ifft2(self.psi_mod_k)
            self.psi_x *= self.prefactor_x()
            self.psi_x += np.sqrt(dt_tilde) * np.sqrt(self.sigma) * (np.random.normal(0, 1, (N,N)) + 1j*np.random.normal(0, 1, (N,N)))
            if i>=i1 and i<=i2 and i%every==0:
                if i % 10000 == 0:
                    print(i)
                v[(i-i1)//every] = self.vortices((i-i1)//every)
                n_sum[(i-i1)//every] = np.mean(self.n())
        g1_x = np.zeros(int(N/2), dtype = complex)
        d1_x = np.zeros(int(N/2))
        for i in range(N):
            g1_x += np.conjugate(self.psi_x[i, int(N/2)]) * self.psi_x[i, int(N/2):] / N
            d1_x += self.n()[i, int(N/2):] / N
        g1_x[0] -= 1/(2*dx_tilde**2)
        return g1_x, d1_x, n_sum, v

def call(gr_dim, g_dim):
    gpe = model(gr_dim, g_dim)
    g1_x, d1_x, avg, v = gpe.time_evolution()
    np.savetxt(name_save_local + str(g_dim)+'_'+str(gr_dim) + os.sep + 'g1' + '_' + str(g_dim) + '_' + str(gr_dim) + '.dat', (np.abs(g1_x)/np.sqrt(d1_x[0]*d1_x)).real)
    np.savetxt(name_save_local + str(g_dim)+'_'+str(gr_dim) + os.sep + 'avg' + '_' + str(g_dim) + '_' + str(gr_dim) + '.dat', avg)
    np.savetxt(name_save_local + str(g_dim)+'_'+str(gr_dim) + os.sep + 'vortices' + '_' + str(g_dim) + '_' + str(gr_dim) + '.dat', v)

# =============================================================================
# Parallel tests
# =============================================================================
name_save_local = r'/Users/delis/Desktop/tests/'
name_save_rem = r'/scratch/konstantinos/tests/'

from qutip import *
qutip.settings.num_cpus = 8

gr_dim_array = np.array([0.26, 0.28, 0.3, 0.4, 0.5, 0.6, 0.7, 0.9])
g_dim = 4

for gr_dim in gr_dim_array:
    os.mkdir(name_save_local + os.sep + str(g_dim) + '_' + str(gr_dim))

parallel_map(call, gr_dim_array, task_kwargs=dict(g_dim = g_dim))

fig,ax = pl.subplots(1,1, figsize=(10,10))
ax.set_xscale('log')
ax.set_yscale('log')
for gr_dim in gr_dim_array:
    correlator = np.loadtxt(name_save_local + str(g_dim)+'_'+str(gr_dim) + os.sep + 'g1' + '_' + str(g_dim) + '_' + str(gr_dim) + '.dat')
    ax.plot(x[int(N/2+1):]-x[int(N/2)], -2*np.log(correlator[1:]), label=r'$gr$ = %.2f' % gr_dim)
ax.plot(x[int(N/2+1):]-x[int(N/2)], 0.01 * (x[int(N/2+1):]-x[int(N/2)]) ** 0.78, color='black', linewidth=3)
ax.tick_params(axis='both', which='both', direction='in', labelsize=20, pad=12, length=12)
pl.legend(prop=dict(size=20))
pl.title('p = %.2f, ns = %.1f' % (p, ns_tilde), fontsize=20)
pl.tight_layout()
pl.show()

fig,ax = pl.subplots(1,1, figsize=(10,10))
for gr_dim in gr_dim_array:
    avgdensity = np.loadtxt(name_save_local + str(g_dim)+'_'+str(gr_dim) + os.sep + 'avg' + '_' + str(g_dim) + '_' + str(gr_dim) + '.dat')
    ax.plot(t, avgdensity, label=r'$gr$ = %.2f' % gr_dim)
ax.tick_params(axis='both', which='both', direction='in', labelsize=20, pad=12, length=12)
ax.legend(prop=dict(size=20))
ax.hlines(y=n0_tilde, xmin=t[0], xmax=t[-1])
pl.title('p = %.2f, ns = %.1f' % (p, ns_tilde), fontsize=20)
pl.tight_layout()
pl.show()

fig,ax = pl.subplots(1,1, figsize=(10,10))
for gr_dim in gr_dim_array:
    vortices = np.loadtxt(name_save_local + str(g_dim)+'_'+str(gr_dim) + os.sep + 'vortices' + '_' + str(g_dim) + '_' + str(gr_dim) + '.dat')
    ax.plot(t, vortices/N**2, label=r'$gr$ = %.2f' % gr_dim)
ax.tick_params(axis='both', which='both', direction='in', labelsize=20, pad=12, length=12)
ax.legend(prop=dict(size=20))
pl.title('p = %.2f, ns = %.1f' % (p, ns_tilde), fontsize=20)
pl.tight_layout()
pl.show()