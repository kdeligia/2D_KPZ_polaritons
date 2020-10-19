#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  3 14:02:54 2020

@author: delis
"""

import os
import numpy as np
import external as ext
from scipy.fftpack import fft2, ifft2
import matplotlib.pyplot as pl
from mpl_toolkits import mplot3d

N = 2 ** 7
L = 2 ** 7 #* hatx

class GrossPitaevskii:
    def __init__(self, Kc, Kd, Kc2, rc, rd, uc, ud, sigma, psi_x=0):
        self.L = L
        self.N = N
        self.Kc = Kc
        self.Kd = Kd
        self.Kc2 = Kc2
        self.rc = rc
        self.rd = rd
        self.uc = uc
        self.ud = ud
        self.sigma = sigma

        self.psi_x = psi_x
        self.psi_x = np.full((N, N), 10)
        self.psi_x /= np.sqrt(np.sum((np.conjugate(self.psi_x) * self.psi_x).flatten())*dx*dy)

        self.psi_mod_k = fft2(self.psi_mod_x)
        self.X, self.Y= np.meshgrid(x,y)
        self.KX, self.KY = np.meshgrid(kx, ky)

# =============================================================================
# Discrete Fourier pairs
# =============================================================================
    def _set_fourier_psi_x(self, psi_x):
        self.psi_mod_x = psi_x * np.exp(-1j * kx[0] * x - 1j * ky[0] * y) * dx * dy / (2 * np.pi)

    def _get_psi_x(self):
        return self.psi_mod_x * np.exp(1j * kx[0] * x + 1j * ky[0] * y) * 2 * np.pi / (dx * dy)

    def _set_fourier_psi_k(self, psi_k):
        self.psi_mod_k = psi_k * np.exp(1j * x[0] * dkx * np.arange(N) + 1j * y[0] * dky)

    def _get_psi_k(self):
        return self.psi_mod_k * np.exp(-1j * x[0] * dkx * np.arange(N) - 1j * y[0] * dky)

    psi_x = property(_get_psi_x, _set_fourier_psi_x)
    psi_k = property(_get_psi_k, _set_fourier_psi_k)

# =============================================================================
# Definition of the split steps
# =============================================================================
    def prefactor_x(self):
        return np.exp(1j * 0.5 * (-1j*dt) * ((self.rc - 1j * self.rd) - (self.uc - 1j * self.ud) * self.psi_x * np.conjugate(self.psi_x) - harm))

    def prefactor_k(self):
        return np.exp(- 1j * (-1j*dt) * ((self.KX**2 + self.KY ** 2) * (self.Kc - 1j * self.Kd) - (self.KX ** 4 + self.KY ** 4) * self.Kc2))

# =============================================================================
# Time evolution and Phase unwinding
# =============================================================================
    def time_evolution(self, realisation):
        for i in range(N_steps+1):
            #self.psi_x /= np.sqrt(N*np.sum((np.conjugate(self.psi_x)[0] * self.psi_x[0]))*dx*dy)
            #self.psi_x += np.sqrt(self.sigma) * np.sqrt(dt) * ext.noise(self.psi_x.shape)
            self.psi_x *= self.prefactor_x()
            self.psi_mod_k = fft2(self.psi_mod_x)
            self.psi_k *= self.prefactor_k()
            self.psi_mod_x = ifft2(self.psi_mod_k)
            self.psi_x *= self.prefactor_x()
            self.psi_x /= np.sqrt(np.sum(np.conjugate(self.psi_x) * self.psi_x, axis=None)*dx*dy)
        density = self.psi_x * np.conjugate(self.psi_x)
        return density

# =============================================================================
# Input
# =============================================================================
dx = L / N
dy = L / N
dkx = 2 * np.pi / (N * dx)
dky = 2 * np.pi / (N * dy)

def grids():
    x_0 = - N * dx / 2
    y_0 = - N * dy / 2
    kx0 = - np.pi / dx
    ky0 = -np.pi / dy
    x = x_0 + dx * np.arange(N)
    y = y_0 + dx * np.arange(N)
    kx = kx0 + dkx * np.arange(N)
    ky = ky0 + dky * np.arange(N)
    return x, y, kx, ky

x, y, kx, ky =  grids()

N_steps = 20000
dt = 0.01

X, Y = np.meshgrid(x,y)
harm = (0.005*X**2 + 0.001*Y**2)/2
GP = GrossPitaevskii(Kc=1, Kd=0, Kc2=0, rc=0, rd=0, uc=1000, ud=0, sigma=0)
density = GP.time_evolution(1)

'''
fig,ax = pl.subplots(1,1, figsize=(10,10))
ax = pl.axes(projection='3d')
ax.plot_surface(X, Y, density.real, cmap='cividis', edgecolor='none')
pl.show()
'''

illustration = -harm[int(N/2)]/1000+0.0008
for i in range(N):
    if illustration[i] < 0 :
        illustration[i]=0

pl.plot(x, density[int(N/2)], label='Result')
pl.plot(x, illustration, label='inverted potential')
pl.legend()
pl.show()