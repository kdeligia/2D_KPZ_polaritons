#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 18:29:32 2020

@author: delis
"""

import warnings
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

        self.L = L
        self.N = N
        self.psi_x = psi_x
        self.psi_x = np.full((N, N), 5)

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
        return np.exp(-1j*0.5*dt*(g*wave_fn*np.conjugate(wave_fn) + 1j*(P/(1+wave_fn*np.conjugate(wave_fn)/ns)-gamma)))

    def prefactor_k(self):
        return np.exp(-1j*dt*((self.KX**2 + self.KY ** 2) * (1/2*m)))

# =============================================================================
# Time evolution and Phase unwinding
# =============================================================================
    def time_evolution(self, realisation):
        sample_psi= np.zeros((len(t), int(N/2)), dtype=complex)
        for i in range(N_steps+1):
            self.psi_x += np.sqrt(sigma) * np.sqrt(dt) * ext.noise((N, N))
            self.psi_x *= self.prefactor_x(self.psi_x)
            self.psi_mod_k = fft2(self.psi_mod_x)
            self.psi_k *= self.prefactor_k()
            self.psi_mod_x = ifft2(self.psi_mod_k)
            self.psi_x *= self.prefactor_x(self.psi_x)
            if i>=i1 and i<=i2 and i%secondarystep==0:
                sample_psi[(i-i1)//secondarystep] = self.psi_x[int(N/2), int(N/2):]
        return sample_psi

# =============================================================================
# Input
# =============================================================================
dt=0.005
g = 0.2
m = 1
P = 20
ns = 1
gamma = P/2
sigma = 0.01
GAMMA = gamma*(P-gamma)/P
mu = g*ns

N = 2**6
L = 2**6

dx = 0.5
dy = 0.5
dkx = 2 * np.pi / (N * dx)
dky = 2 * np.pi / (N * dy)

def arrays():
    x_0 = - N * dx / 2
    kx0 = - np.pi / dx
    x = x_0 + dx * np.arange(N)
    kx = kx0 + dkx * np.arange(N)
    return x, kx

x, kx =  arrays()
X,Y = np.meshgrid(x, x)

N_steps = 200000
secondarystep = 10000
i1 = 10000
i2 = N_steps
lengthwindow = i2-i1

t = ext.time(dt, N_steps, i1, i2, secondarystep)

n_tasks = 200
n_batch = 20
n_internal = n_tasks//n_batch

def g1(i_batch):
    correlator_batch = np.zeros((len(t), int(N/2)), dtype=complex)
    for i_n in range(n_internal):
        GP = GrossPitaevskii()
        psi = GP.time_evolution(i_n)
        for i in range(len(t)):
            psi[i] *= np.conjugate(psi[i,0])
        correlator_batch += psi / n_internal
    name_full1 = '/scratch/konstantinos/g_0_2'+os.sep+'n_batch'+str(i_batch+1)+'.dat'
    np.savetxt(name_full1, correlator_batch, fmt='%.5f')

qutip.settings.num_cpus = n_batch
parallel_map(g1, range(n_batch))

path1 = r"/scratch/konstantinos/g_0_2"

def ensemble_average(path):
    countavg = 0
    for file in os.listdir(path):
       if '.dat' in file:
           countavg += 1
    for file in os.listdir(path):
        if '.dat' in file:
            avg = np.zeros_like(np.loadtxt(path+os.sep+file, dtype=np.complex_), dtype=np.complex_)
        continue
    for file in os.listdir(path):
        if '.dat' in file:
            numerator = np.loadtxt(path+os.sep+file, dtype=np.complex_)
            avg += numerator / countavg
    return avg

numerator = ensemble_average(path1)
result = np.absolute(numerator)/ns
np.savetxt('/home6/konstantinos/g_0_2.dat', result)