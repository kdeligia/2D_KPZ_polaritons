#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 13:08:30 2020

@author: delis
"""
import matplotlib.pyplot as pl
from qutip import *
import os
import numpy as np
import external as ext
from scipy.fftpack import fft2, ifft2
warnings.filterwarnings("ignore", message="Casting complex values to real discards the imaginary part")
#from scipy import signal

class GrossPitaevskii:
    def __init__(self, psi_x=0):
        self.X, self.Y= np.meshgrid(x,x)
        self.KX, self.KY = np.meshgrid(kx, kx)

        self.L = L
        self.N = N
        self.psi_x = psi_x
        self.psi_x = np.full((N, N), 1)
        self.psi_mod_k = fft2(self.psi_mod_x)
        '''
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
        return np.exp(-1j*(dt/2)*(g*wave_fn*np.conjugate(wave_fn) + 1j*(P/(1+wave_fn*np.conjugate(wave_fn)/ns)-gamma)))

    def prefactor_k(self):
        return np.exp(-1j*dt*((self.KX**2 + self.KY ** 2) * (1/2*m)))

# =============================================================================
# Time evolution and Phase unwinding
# =============================================================================
    def time_evolution(self, realisation):
        sample= np.zeros(int(N/2), dtype=complex)
        for i in range(N_steps+1):
            #if i%1000==0:
                #print(i)
            self.psi_x *= self.prefactor_x(self.psi_x)
            self.psi_mod_k = fft2(self.psi_mod_x)
            self.psi_k *= self.prefactor_k()
            self.psi_mod_x = ifft2(self.psi_mod_k)
            self.psi_x *= self.prefactor_x(self.psi_x)
            self.psi_x += np.sqrt(sigma) * np.sqrt(dt) * ext.noise((N, N))
            if i>=i1 and i<=i2 and i%secondarystep==0:
                sample += np.conjugate(self.psi_x[int(N/2), int(N/2)])*self.psi_x[int(N/2), int(N/2):] / len(t)
        return sample

# =============================================================================
# Input
# =============================================================================
dt=0.001
g = 2
m = 1
P = 20
ns = 1
gamma = P/2
sigma = 0.04
GAMMA = gamma*(P-gamma)/P
mu = g*ns

N = 2**6
L = 2**6

dx = 0.5
dy = 0.5
dkx = 2 * np.pi / (N * dx)
dky = 2 * np.pi / (N * dy)

'''
def params(m, g, P, gamma):
    Kc = 1/(2*m)
    rd = P - gamma
    uc = g
    ud = P/(2*ns)
    return Kc, rd, uc, ud
Kc, rd, uc, ud = params(m, g, P, gamma)

print('-----PARAMS-----')
print('Kc', Kc)
print('rd', rd)
print('uc', uc)
print('ud', ud)
'''

def arrays():
    x_0 = - N * dx / 2
    kx0 = - np.pi / dx
    x = x_0 + dx * np.arange(N)
    kx = kx0 + dkx * np.arange(N)
    return x, kx

x, kx =  arrays()
X,Y = np.meshgrid(x, x)
N_steps = 1000000

secondarystep = 1000
i1 = 100000
i2 = N_steps
lengthwindow = i2-i1

t = ext.time(dt, N_steps, i1, i2, secondarystep)

#GP = GrossPitaevskii()
#psi = GP.time_evolution(1)

#dx = x[int(N/2):] - x[int(N/2)]
#pl.plot(t, np.abs(psi[:,0]))

n_tasks = 400
n_batch = 40
n_internal = n_tasks//n_batch

def g1(i_batch):
    correlator_batch = np.zeros(int(N/2), dtype=complex)
    for i_n in range(n_internal):
        if i_n>0:
            print('The core', i_batch+1, 'is on the realisation number', i_n)
        GP = GrossPitaevskii()
        sample = GP.time_evolution(i_n)
        correlator_batch += sample/n_internal
    name_full1 = '/scratch/konstantinos/numerator_batch'+os.sep+'n_batch'+str(i_batch+1)+'.dat'
    np.savetxt(name_full1, correlator_batch, fmt='%.5f')

qutip.settings.num_cpus = n_batch
parallel_map(g1, range(n_batch))
path1 = r"/scratch/konstantinos/numerator_batch"

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
result = np.abs(numerator)/ns
np.savetxt('/home6/konstantinos/g_2.dat', result)

'''
test = np.loadtxt('/Users/delis/Desktop/test.dat')
dx = x[int(N/2):] - x[int(N/2)]
pl.loglog(dx, test)
'''