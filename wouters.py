#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 13:08:30 2020

@author: delis
"""
import math as math
import os
import numpy as np
import external as ext
from scipy.fftpack import fft2, ifft2
import matplotlib.pyplot as pl
from mpl_toolkits import mplot3d
from scipy import signal

class GrossPitaevskii:
    def __init__(self, Kc, Kd, Kc2, rc, rd, uc, ud, sigma, psi_x=0):
        self.X, self.Y= np.meshgrid(x,y)
        self.KX, self.KY = np.meshgrid(kx, ky)

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
        self.initcond = np.full((N,N),np.sqrt(n_s))
        self.initcond[int(N/2),int(N/4)] = 0
        self.initcond[int(N/2),int(3*N/4)] = 0
        rot = []
        for i in range(N):
            for j in range(N):
                if i <= int(N/2):
                    rot.append(np.exp(-1*1j*math.atan2(x[i], y[j])))
                elif i>int(N/2):
                    rot.append(np.exp(1*1j*math.atan2(x[i], y[j])))
        self.psi_x = np.array(rot).reshape(N,N) * self.initcond
        '''
        density = (self.psi_x * np.conjugate(self.psi_x)).real
        fig,ax = pl.subplots(1,1, figsize=(8,8))
        c = ax.pcolormesh(X, Y, density, cmap='viridis')
        ax.set_title('Density')
        ax.axis([x.min(), x.max(), y.min(), y.max()])
        fig.colorbar(c, ax=ax)
        pl.show()
        '''
        '''
        self.psi_x = np.ones((N,N))
        self.psi_mod_k = fft2(self.psi_mod_x)
        print(self.psi_mod_x[5,5])
        print(ifft2(fft2(self.psi_mod_x))[5,5])

        fig,ax = pl.subplots(1,1, figsize=(8,8))
        c = ax.pcolormesh(self.KX, self.KY, np.abs(self.psi_k), cmap='viridis')
        ax.set_title('FT')
        ax.axis([kx.min(), kx.max(), ky.min(), ky.max()])
        fig.colorbar(c, ax=ax)
        pl.show()
        
        self.psi_mod_x = ifft2(self.psi_mod_k)
        fig,ax = pl.subplots(1,1, figsize=(8,8))
        c = ax.pcolormesh(self.X, self.Y, np.abs(self.psi_x), cmap='viridis')
        ax.set_title('IFFT')
        ax.axis([kx.min(), kx.max(), ky.min(), ky.max()])
        fig.colorbar(c, ax=ax)
        pl.show()
        '''

# =============================================================================
# Discrete Fourier pairs
# =============================================================================
    def _set_fourier_psi_x(self, psi_x):
        self.psi_mod_x = psi_x * np.exp(-1j * self.KX[0,0] * X - 1j * self.KY[0,0] * Y) * dx * dy / (2 * np.pi)

    def _get_psi_x(self):
        return self.psi_mod_x * np.exp(1j * self.KX[0,0] * X + 1j * self.KY[0,0] * Y) * 2 * np.pi / (dx * dy)

    def _set_fourier_psi_k(self, psi_k):
        self.psi_mod_k = psi_k * np.exp(1j * X[0,0] * dkx * np.arange(N) + 1j * Y[0,0] * dky * np.arange(N))

    def _get_psi_k(self):
        return self.psi_mod_k * np.exp(-1j * X[0,0] * dkx * np.arange(N) - 1j * Y[0,0] * dky * np.arange(N))

    psi_x = property(_get_psi_x, _set_fourier_psi_x)
    psi_k = property(_get_psi_k, _set_fourier_psi_k)
    
# =============================================================================
# Definition of the split steps
# =============================================================================
    def prefactor_x(self, wave_fn):
        return np.exp(1j*0.5*dt*((self.rc - 1j * self.rd) - (self.uc - 1j * self.ud) * wave_fn * np.conjugate(wave_fn)))

    def prefactor_k(self):
        return np.exp(-1j*dt*((self.KX**2 + self.KY ** 2) * (self.Kc - 1j * self.Kd) - (self.KX ** 4 + self.KY ** 4) * self.Kc2))

# =============================================================================
# Time evolution and Phase unwinding
# =============================================================================
    def time_evolution(self, realisation):
        t = 0
        for i in range(N_steps+1):
            if t == 0:
                V = 0
            elif t != 0:
                V = 0
            if i%50==0:
                print(i)
            self.psi_x += np.sqrt(self.sigma) * np.sqrt(dt) * ext.noise(self.psi_x.shape)
            self.psi_x *= self.prefactor_x(self.psi_x) * np.exp(-1j * 0.5 * dt * V)
            self.psi_mod_k = fft2(self.psi_mod_x)
            self.psi_k *= self.prefactor_k()
            self.psi_mod_x = ifft2(self.psi_mod_k)
            self.psi_x *= self.prefactor_x(self.psi_x) * np.exp(-1j * 0.5 * dt * V)
            t += 0.001
            density = (self.psi_x * np.conjugate(self.psi_x)).real
            if i%500==0:
                fig,ax = pl.subplots(1,1, figsize=(8,8))
                c = ax.pcolormesh(X, Y, density, cmap='viridis')
                ax.set_title('Density')
                ax.axis([x.min(), x.max(), y.min(), y.max()])
                fig.colorbar(c, ax=ax)
                pl.show()
        return density

# =============================================================================
# Input
# =============================================================================
dt=0.001
g = 0
m = 1
P = 8
n_s = 1
gamma = P/2

N = 2**8
L = 2**8

dx = 0.5
dy = 0.5
dkx = 2 * np.pi / (N * dx)
dky = 2 * np.pi / (N * dy)

def params(m, g, P, gamma):
    Kc = 1/(2*m)
    Kd = 0
    rc = 0
    rd = P - gamma
    uc = g
    ud =P/n_s
    return Kc, Kd, rc, rd, uc, ud
Kc, Kd, rc, rd, uc, ud = params(m, g, P, gamma)

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

N_steps = 50000

X, Y = np.meshgrid(x,y)
GP = GrossPitaevskii(Kc=Kc, Kd=Kd, Kc2=0, rc=rc, rd=rd, uc=uc, ud=ud, sigma=0.01)
