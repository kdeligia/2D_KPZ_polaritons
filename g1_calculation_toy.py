#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 15:09:50 2020

@author: delis
"""

c = 3E2 #μm/ps
hbar = 6.582119569 * 1E2 # μeV ps

import matplotlib.pyplot as pl
import os
from scipy.fftpack import fft2, ifft2
import numpy as np
import external as ext
from qutip import *
name_local = r'/Users/delis/Desktop/'
name_remote = r'/scratch/konstantinos/'

parallel_tasks = 256
n_batch = 128
n_internal = parallel_tasks//n_batch
qutip.settings.num_cpus = n_batch

N = 2**7
L = 2**7
hatt = 1 #ps
hatx = 1 #μm
hatpsi = 1/hatx #μm^-1

star_m = 5e-6
gamma0 = 0.19 #ps^-1
gammar = 0.015 #ps^-1
gamma2 = 100/hbar #μm^2 ps^-1

P = 1.026e2
R = 5e-5

p = P*R / (gamma0*gammar)
ns = gammar/R
n0 = ns*(p-1)
nres = P/(gammar+R*n0)
gr = 0.025
g = 4.5

def arrays():
    x_0 = - N * dx / 2
    kx0 = - np.pi / dx
    x = x_0 + dx * np.arange(N)
    kx = kx0 + dkx * np.arange(N)
    return x, kx

L *= hatx
L /= hatx
dx = L/N
dkx = 2 * np.pi / (N * dx)
x, kx =  arrays()
X, Y = np.meshgrid(x, x)
KX, KY = np.meshgrid(kx, kx)

m =  star_m * 0.510998950 * 1E12 / c**2 #μeV/(μm^2/ps^2)
star_gamma_l0 = (gamma0*hbar)  # μeV 
star_gamma_l2 = (gamma2*hbar) # μeV μm^2 
star_gamma_r = (gammar*hbar) # μeV

time_steps = 100000
dt = 4e-2 * hatt
every = 100
i1 = 0
i2 = time_steps
lengthwindow = i2-i1
t = ext.time(dt, time_steps, i1, i2, every)

#np.savetxt('/Users/delis/Desktop/dt.dat', np.arange(1001))
#np.savetxt('/Users/delis/Desktop/dr_2_7,dat', x-x[0])

print('--- Energy scales ---')
print(r'losses/kinetic %.4f' % (hbar*gamma0/(hbar**2/(2*abs(m)*dx**2))))
print(r'p-p interaction/kinetic %.4f' % (g*n0/(hbar**2/(2*abs(m)*dx**2))))
print(r'p-r interaction/kinetic %.4f' % (gr*nres/(hbar**2/(2*abs(m)*dx**2))))
print(r'Total blueshift from interactions in μeV %.4f' % (g*n0 + gr*nres))
print(r'Truncated Wigner  ratio %.4f' % (g/(hbar*gamma0*dx**2)))
print(r'dx/healing length %.4f' % (dx / (hbar/np.sqrt(2*abs(m)*g*n0))))
print('--- Losses ---')
print('gamma_0 in μeV %.4f' % star_gamma_l0)
print('gamma_r in μeV %.4f' % star_gamma_r)
print('gamma_2 in μeV μm^2 %.4f' % star_gamma_l2)
print('--- Interactions ---')
print('Polariton-reservoir in μeV μm^2 %.4f' % gr)
print('Polariton-polariton in μev μm^2 %.4f' % g)
print('--- Densities ---')
print('Saturation in μm^-2 %.2f' % (gammar/R))
print('Steady-state in μm^-2 %.2f' % n0)
print('Reservoir in μm^-2 %.2f' % (nres))
print('--- Dimensionless pump ---')
print('p %.4f' % p)

class model:
    def __init__(self, psi_x=0):
        self.sigma = hatt * gamma0*(p+1)/(2*dx**2)
        self.Kc = (hatt/hatx**2) * hbar/(2*m)
        self.Kd = (hatt/hatx**2) * gamma2/2
        self.psi_x = psi_x
        self.psi_x = np.full((N, N), 2)
        self.psi_x /= hatpsi
        self.psi_mod_k = fft2(self.psi_mod_x)

# =============================================================================
# Discrete Fourier pairs
# =============================================================================
    def _set_fourier_psi_x(self, psi_x):
        self.psi_mod_x = psi_x * np.exp(-1j * KX[0,0] * X - 1j * KY[0,0] * Y) * dx * dx / (2 * np.pi)

    def _get_psi_x(self):
        return self.psi_mod_x * np.exp(1j * KX[0,0] * X + 1j * KY[0,0] * Y) * 2 * np.pi / (dx * dx)

    def _set_fourier_psi_k(self, psi_k):
        self.psi_mod_k = psi_k * np.exp(1j * X[0,0] * dkx * np.arange(N) + 1j * Y[0,0] * dkx * np.arange(N))

    def _get_psi_k(self):
        return self.psi_mod_k * np.exp(-1j * X[0,0] * dkx * np.arange(N) - 1j * Y[0,0] * dkx * np.arange(N))

    psi_x = property(_get_psi_x, _set_fourier_psi_x)
    psi_k = property(_get_psi_k, _set_fourier_psi_k)
# =============================================================================
# Definition of the split steps
# =============================================================================
    def n(self):
        return np.abs(self.psi_x * np.conjugate(self.psi_x)) - 1/(2*dx**2)

    def n_r(self, nr_update):
        q = gammar + R*self.n()
        return P/q - P/q*np.exp(-q*dt/2) - nr_update*np.exp(-q*dt/2)

    def prefactor_x(self):
        self.uc_tilde = hatt/(hbar*hatx**2) * g*(self.n() + (hatx**2*gr/g) * (p*gamma0/R) * (1/(1+self.n()/(hatx**2*ns))))
        self.I_tilde = 1j*hatt*gamma0/2 * (p/(1+self.n()/(hatx**2*ns)) - 1)
        return np.exp(-1j*0.5*dt*(self.uc_tilde + self.I_tilde))

    def prefactor_k(self):
        return np.exp(-1j*dt*((KX**2 + KY**2)*(self.Kc - 1j*self.Kd)))

# =============================================================================
# Time evolution
# =============================================================================
    def vortices(self):
        count_v = 0
        count_av = 0
        theta = np.angle(self.psi_x)
        grad = np.gradient(theta, dx)
        #v_pos = np.zeros((N, N))
        #av_pos = np.zeros((N, N))
        for i in range(1, N-1):
            for j in range(1, N-1):
                loop = (2*dx*(grad[0][j+1, i+1] - grad[1][j+1, i+1]) +
                        2*dx*(grad[0][j+1, i-1] + grad[1][j+1, i-1]) +
                        2*dx*(-grad[0][j-1, i-1] + grad[1][j-1, i-1]) +
                        2*dx*(-grad[0][j-1, i+1] - grad[1][j-1, i+1]) +
                        2*dx*(grad[0][j+1, i] + grad[1][j, i-1] - grad[0][j-1, i] - grad[1][j, i+1]))
                if loop >= 2 * np.pi:
                    count_v += 1
                    #v_pos[i,j] = 1
                elif loop <= - 2 * np.pi:
                    count_av +=1
                    #av_pos[i,j] = 1
        total = count_v + count_av
        '''
        fig,ax = pl.subplots(1,1, figsize=(8,6))
        xv = np.array([x[i] for i in range(N) for j in range(N) if v_pos[i,j]==1])
        yv = np.array([x[j] for i in range(N) for j in range(N) if v_pos[i,j]==1])
        xav = np.array([x[i] for i in range(N) for j in range(N) if av_pos[i,j]==1])
        yav = np.array([x[j] for i in range(N) for j in range(N) if av_pos[i,j]==1])
        ax.plot(xv, yv, 'go', markersize=2)
        ax.plot(xav, yav, 'ro', markersize=2)
        ax.set_xlim(x[0], x[-1])
        ax.set_ylim(x[0], x[-1])
        im = ax.pcolormesh(X, Y, theta, cmap='Greys')
        pl.colorbar(im)
        pl.show()
        '''
        return total

    def time_evolution(self, seed):
        '''
        g1_x = np.zeros(int(N/2), dtype = complex)
        d1_x = np.zeros(int(N/2))
        '''
        vortexnumber = np.zeros(len(t))
        np.random.seed()
        for i in range(time_steps+1):
            self.psi_x *= self.prefactor_x()
            self.psi_mod_k = fft2(self.psi_mod_x)
            self.psi_k *= self.prefactor_k()
            self.psi_mod_x = ifft2(self.psi_mod_k)
            self.psi_x *= self.prefactor_x()
            self.psi_x += np.sqrt(dt) * np.sqrt(self.sigma) * (np.random.normal(0, 1, (N,N)) + 1j*np.random.normal(0, 1, (N,N)))
            if i>=i1 and i<=i2 and i%every==0:
                vortexnumber[(i-i1)//every] = self.vortices()
            return vortexnumber
        '''
        for i in range(0, N, int(N/8)):
            g1_x += np.conjugate(self.psi_x[i, int(N/2)]) * self.psi_x[i, int(N/2):] / 8
            d1_x += self.n()[i, int(N/2):] / 8
        g1_x[0] -= 1/(2*dx**2)
        return g1_x, d1_x
    '''


saveresult = r'/home6/konstantinos/'
def g1(i_batch):
    seed = i_batch
    #correlation_batch = np.zeros((2, int(N/2)), dtype=complex)
    vortexnumber_batch = np.zeros(len(t))
    for i_n in range(n_internal):
        gpe = model()
        '''
        g1_x_run, d1_x_run = gpe.time_evolution(seed)
        correlation_batch += np.vstack((g1_x_run, d1_x_run)) / n_internal
        '''
        vortexnumber_run = gpe.time_evolution(seed)
        vortexnumber_batch += vortexnumber_run / n_internal
        seed += n_batch
        print('The core', i_batch, 'has completed realisation number', i_n+1)
    #np.save(name_remote+'correlation_g'+str(g)+'gr'+str(gr)+os.sep+'file_core'+str(i_batch+1)+'.npy', correlation_batch)
    np.save(name_remote+'vortices_g'+str(g)+'gr'+str(gr)+os.sep+'file_core'+str(i_batch+1)+'.npy', vortexnumber_batch)

parallel_map(g1, range(n_batch))
'''
result = ext.ensemble_average_space(name_remote+'correlation_g'+str(g)+'gr'+str(gr), 10, int(N/2), n_batch)
np.savetxt(saveresult+'correlation_g'+str(g)+'gr'+str(gr)+'.dat', result)
'''
result = ext.ensemble_average_time(name_remote+'vortices_g'+str(g)+'gr'+str(gr), t, n_batch)
np.savetxt(saveresult+'vortices_g'+str(g)+'gr'+str(gr)+'.dat', result)