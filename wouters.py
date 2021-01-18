#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 13:08:30 2020

@author: delis
"""

import math
import matplotlib.pyplot as pl
from qutip import *
import os
import numpy as np
import external as ext
from scipy.fftpack import fft2, ifft2
warnings.filterwarnings("ignore", message="Casting complex values to real discards the imaginary part")

class GrossPitaevskii:
    def __init__(self, psi_x=0):
        self.X, self.Y= np.meshgrid(x,x)
        self.KX, self.KY = np.meshgrid(kx, kx)
        self.n_s = 1 
        self.L = L
        self.N = N
        self.psi_x = psi_x
        self.psi_x = np.full((N, N), 10)
        self.psi_mod_k = fft2(self.psi_mod_x)
        '''
        self.initcond = np.full((N,N),np.sqrt(self.n_s))
        self.initcond[int(N/2),int(N/4)] = 0
        self.initcond[int(N/2),int(3*N/4)] = 0
        rot = []
        for i in range(N):
            for j in range(N):
                if i <= int(N/2):
                    rot.append(np.exp(-1*1j*math.atan2(x[i], x[j])))
                elif i>int(N/2):
                    rot.append(np.exp(1*1j*math.atan2(x[i], x[j])))
        self.psi_x = np.array(rot).reshape(N,N) * self.initcond

        density = (self.psi_x * np.conjugate(self.psi_x)).real
        fig,ax = pl.subplots(1,1, figsize=(8,8))
        c = ax.pcolormesh(X, Y, density, cmap='viridis')
        ax.set_title('Density')
        ax.axis([x.min(), x.max(), x.min(), x.max()])
        fig.colorbar(c, ax=ax)
        pl.show()
        '''
        '''
        fig,ax = pl.subplots(1,1, figsize=(8,8))
        c = ax.pcolormesh(self.KX, self.KY, np.abs(self.psi_k), cmap='viridis')
        ax.set_title('FT')
        ax.axis([kx.min(), kx.max(), kx.min(), kx.max()])
        fig.colorbar(c, ax=ax)
        pl.show()
        self.psi_mod_x = ifft2(self.psi_mod_k)
        fig,ax = pl.subplots(1,1, figsize=(8,8))
        c = ax.pcolormesh(self.X, self.Y, np.abs(self.psi_x), cmap='viridis')
        ax.set_title('IFFT')
        ax.axis([kx.min(), kx.max(), kx.min(), kx.max()])
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
        n_red = wave_fn * np.conjugate(wave_fn)
        self.rc = 0
        self.rd = (P - gamma)
        self.ud = P/(self.n_s)
        self.uc = g
        self.z = 1
        return np.exp(-1j*0.5*dt*((self.rc + 1j*self.rd) + (self.uc - 1j*self.ud)*n_red)/self.z)

    def prefactor_k(self):
        self.Kc = 1/(2*m)
        return np.exp(-1j*dt*((self.KX**2 + self.KY ** 2) * self.Kc))

# =============================================================================
# Time evolution and Phase unwinding
# =============================================================================
    def time_evolution(self, realisation):
        n = np.zeros(len(t))
        n0 = np.zeros(len(t))
        for i in range(N_steps+1):
            self.psi_x *= self.prefactor_x(self.psi_x)
            self.psi_mod_k = fft2(self.psi_mod_x)
            self.psi_k *= self.prefactor_k()
            self.psi_mod_x = ifft2(self.psi_mod_k)
            self.psi_x *= self.prefactor_x(self.psi_x)
            self.psi_x += np.sqrt(sigma/dx**2) * np.sqrt(dt) * ext.noise((N, N))
        return self.psi_x[int(N/2), int(N/2):]

# =============================================================================
# Input
# =============================================================================
dt=1E-3
g = 0
m = 1
P = 20
gamma = P/2
sigma = 0.01

N = 2**6
L = 2**6

dx = 0.5
dy = 0.5
dkx = 2 * np.pi / (N * dx)
dky = 2 * np.pi / (N * dy)

print('uc', g)
print('ud', P)
print('rd', P-gamma)
print('σ', sigma/dx**2)

def arrays():
    x_0 = - N * dx / 2
    kx0 = - np.pi / dx
    x = x_0 + dx * np.arange(N)
    kx = kx0 + dkx * np.arange(N)
    return x, kx

x, kx =  arrays()
X,Y = np.meshgrid(x, x)
N_steps = 100000

secondarystep = 100
i1 = 0
i2 = N_steps
lengthwindow = i2-i1
t = ext.time(dt, N_steps, i1, i2, secondarystep)

'''
GP = GrossPitaevskii()
n, n0 = GP.time_evolution(0)
pl.plot(t, n)
pl.plot(t, n0)
pl.axhline(y=(P-gamma)/(P/1), xmin=t[0], xmax=t[-1], c='r')
pl.axhline(y=(P/gamma-1), xmin=t[0], xmax=t[-1], c='b')
pl.xlim(20,100)
pl.ylim(0,2)
pl.show()
'''

n_tasks = 100
n_batch = 50
n_internal = n_tasks//n_batch

def g1(i_batch):
    correlator_batch = np.zeros(len(t), dtype=complex)
    for i_n in range(n_internal):
        if i_n>0:
            print('The core', i_batch+1, 'is on the realisation number', i_n)
        GP = GrossPitaevskii()
        sample = GP.time_evolution(i_n)
        correlator_batch += np.conjugate(sample[0])*sample/n_internal
    name_full1 = '/scratch/konstantinos/test'+os.sep+'n_batch'+str(i_batch+1)+'.dat'
    np.savetxt(name_full1, correlator_batch, fmt='%.5f')

qutip.settings.num_cpus = n_batch
parallel_map(g1, range(n_batch))

path1 = r"/scratch/konstantinos/test"
def ensemble_average(path):
    avg = np.zeros(int(N/2), dtype=complex)
    for file in os.listdir(path):
        if '.dat' in file:
            numerator = np.loadtxt(path+os.sep+file, dtype=np.complex_)
            avg += numerator / n_batch
    return avg

c = ensemble_average(path1)
np.save('/home6/konstantinos/test.npy', np.abs(c)/(1/2))
