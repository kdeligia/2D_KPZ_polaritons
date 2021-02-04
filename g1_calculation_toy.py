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

parallel_tasks = 512
n_batch = 128
n_internal = parallel_tasks//n_batch
qutip.settings.num_cpus = n_batch

N = 2**6
L = 2**6
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
g = 4.42

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
    def time_evolution(self, seed):
        g1_x = np.zeros(int(N/2), dtype = complex)
        d1_x = np.zeros(int(N/2))
        g2_x = np.zeros(int(N/2), dtype = complex)
        d2_x = np.zeros(int(N/2))
        g3_x = np.zeros(int(N/2), dtype = complex)
        d3_x = np.zeros(int(N/2))
        g4_x = np.zeros(int(N/2), dtype = complex)
        d4_x = np.zeros(int(N/2))
        g5_x = np.zeros(int(N/2), dtype = complex)
        d5_x = np.zeros(int(N/2))
        np.random.seed(seed)
        for i in range(time_steps+1):
            #np.random.seed()
            self.psi_x *= self.prefactor_x()
            self.psi_mod_k = fft2(self.psi_mod_x)
            self.psi_k *= self.prefactor_k()
            self.psi_mod_x = ifft2(self.psi_mod_k)
            self.psi_x *= self.prefactor_x()
            self.psi_x += np.sqrt(dt) * np.sqrt(self.sigma) * (np.random.normal(0, 1, (N,N)) + 1j*np.random.normal(0, 1, (N,N)))
        for i in range(0, N, int(N/2)):
            g1_x += np.conjugate(self.psi_x[i, int(N/2)]) * self.psi_x[i, int(N/2):] / 2 #+ np.conjugate(self.psi_x[int(N/2), i]) * self.psi_x[int(N/2):, i] / 4
            d1_x += self.n()[i, int(N/2):] / 2 #+ self.n()[int(N/2):, i] / 4
        g1_x[0] -= 1/(2*dx**2)
        for i in range(0, N, int(N/4)):
            g2_x += np.conjugate(self.psi_x[i, int(N/2)]) * self.psi_x[i, int(N/2):] / 4 #+ np.conjugate(self.psi_x[int(N/2), i]) * self.psi_x[int(N/2):, i] / 8
            d2_x += self.n()[i, int(N/2):] / 4 #+ self.n()[int(N/2):, i] / 8
        g2_x[0] -= 1/(2*dx**2)
        for i in range(0, N, int(N/8)):
            g3_x += np.conjugate(self.psi_x[i, int(N/2)]) * self.psi_x[i, int(N/2):] / 8 #+ np.conjugate(self.psi_x[int(N/2), i]) * self.psi_x[int(N/2):, i] / 16
            d3_x += self.n()[i, int(N/2):] / 8 #+ self.n()[int(N/2):, i] / 16
        g3_x[0] -= 1/(2*dx**2)
        for i in range(0, N, int(N/32)):
            g4_x += np.conjugate(self.psi_x[i, int(N/2)]) * self.psi_x[i, int(N/2):] / 32 #+ np.conjugate(self.psi_x[int(N/2), i]) * self.psi_x[int(N/2):, i] / 64
            d4_x += self.n()[i, int(N/2):] / 32 #+ self.n()[int(N/2):, i] / 64
        g4_x[0] -= 1/(2*dx**2)
        g5_x = np.conjugate(self.psi_x[int(N/2), int(N/2)]) * self.psi_x[int(N/2), int(N/2):]
        d5_x = self.n()[int(N/2), int(N/2):]
        g5_x[0] -= 1/(2*dx**2)
        return g1_x, d1_x, g2_x, d2_x, g3_x, d3_x, g4_x, d4_x, g5_x, d5_x

nametosave = name_local
def g1(i_batch):
    seed = i_batch
    num_obs = 5
    batch = np.zeros((2*num_obs, int(N/2)), dtype=complex)
    for i_n in range(n_internal):
        gpe = model()
        g1_x, d1_x, g2_x, d2_x, g3_x, d3_x, g4_x, d4_x, g5_x, d5_x = gpe.time_evolution(seed)
        batch += np.vstack((g1_x, d1_x, g2_x, d2_x, g3_x, d3_x, g4_x, d4_x, g5_x, d5_x)) / n_internal
        seed += n_batch
        print('The core', i_batch, 'has completed realisation number', i_n+1)
    np.save(nametosave+'g'+str(g)+'gr'+str(gr)+os.sep+'file_core'+str(i_batch+1)+'.npy', batch)

parallel_map(g1, range(n_batch))
result = ext.ensemble_average_space(nametosave+'g'+str(g)+'gr'+str(gr), 10, int(N/2), n_batch)
np.savetxt(r'/home6/konstantinos/'+'g'+str(g)+'gr'+str(gr)+'_result_x_512.dat', result)

# =============================================================================
# 
# =============================================================================
'''
result = np.loadtxt('/Users/delis/Desktop/g4.42gr0.025_result_x_256.dat', dtype=np.complex_)
resulty = np.loadtxt('/Users/delis/Desktop/g4.42gr0.025_result_xy.dat', dtype=np.complex_)
g1 = np.abs(result[0])/np.sqrt(result[1,0]*result[1]).real
g2 = np.abs(result[2])/np.sqrt(result[3,0]*result[3]).real
g3 = np.abs(result[4])/np.sqrt(result[5,0]*result[5]).real
g4 = np.abs(result[6])/np.sqrt(result[7,0]*result[7]).real
g5 = np.abs(result[8])/np.sqrt(result[9,0]*result[9]).real
pl.loglog(g1, label=r'$N/2$')
pl.loglog(g2, label=r'$N/4$')
pl.loglog(g3, label=r'$N/8$')
pl.loglog(g4, label=r'$N/32$')
pl.loglog(g5, label=r'No avg')
pl.legend()
pl.title('ROWS ONLY')
pl.show()

g1y = np.abs(resulty[0])/np.sqrt(resulty[1,0]*resulty[1]).real
g2y = np.abs(resulty[2])/np.sqrt(resulty[3,0]*resulty[3]).real
g3y = np.abs(resulty[4])/np.sqrt(resulty[5,0]*resulty[5]).real
g4y = np.abs(resulty[6])/np.sqrt(resulty[7,0]*resulty[7]).real
g5y = np.abs(resulty[8])/np.sqrt(resulty[9,0]*resulty[9]).real
pl.loglog(g1y, label=r'$N/2$')
pl.loglog(g2y, label=r'$N/4$')
pl.loglog(g3y, label=r'$N/8$')
pl.loglog(g4y, label=r'$N/32$')
pl.loglog(g5y, label=r'No avg')
pl.legend()
pl.title('ROWS AND COLUMNS')
pl.show()

pl.title('ROWS AND COLUMNS')
pl.loglog(g1, c='b', label=r'$N/2$')
pl.loglog(g2, c='b', label=r'$N/4$')
pl.loglog(g3, c='b', label=r'$N/8$')
pl.loglog(g4, c='b', label=r'$N/32$')
pl.loglog(g5, c='b', label=r'No avg')
pl.loglog(g1y, c='g', label=r'$N/2$')
pl.loglog(g2y, c='g', label=r'$N/4$')
pl.loglog(g3y, c='g', label=r'$N/8$')
pl.loglog(g4y, c='g', label=r'$N/32$')
pl.loglog(g5y, c='g', label=r'No avg')
pl.show()
'''

# =============================================================================
# 
# =============================================================================
'''
file1 = np.load('/Users/delis/Desktop/g4.42gr0.025/file_core1.npy')
file2 = np.load('/Users/delis/Desktop/g4.42gr0.025/file_core2.npy')
file3 = np.load('/Users/delis/Desktop/g4.42gr0.025/file_core3.npy')
file4 = np.load('/Users/delis/Desktop/g4.42gr0.025/file_core4.npy')

n1 = (np.abs(file1[8])).real
n2 = (np.abs(file2[8])).real
n3 = (np.abs(file3[8])).real
n4 = (np.abs(file4[8])).real

d1 = (np.sqrt(file1[9,0]*file1[9])).real
d2 = (np.sqrt(file2[9,0]*file2[9])).real
d3 = (np.sqrt(file3[9,0]*file3[9])).real
d4 = (np.sqrt(file4[9,0]*file4[9])).real
pl.loglog(n1)
pl.loglog(n2)
pl.loglog(n3)
pl.loglog(n4)
pl.title('No spatial avg, numerator, each core')
pl.tight_layout()
pl.show()

pl.loglog(d1)
pl.loglog(d2)
pl.loglog(d3)
pl.loglog(d4)
pl.title('No spatial avg, denominator, each core')
pl.tight_layout()
pl.show()

pl.loglog(n1/d1)
pl.loglog(n2/d2)
pl.loglog(n3/d3)
pl.loglog(n4/d4)
pl.title('No spatial avg, fraction, each core')
pl.tight_layout()
pl.show()

n1234 = (np.abs((file1[8]+file2[8]+file3[8]+file4[8])/4)).real
denom = (file1[9]+file2[9]+file3[9]+file4[9])/4
pl.loglog(n1234)
pl.title('No spatial avg, numerator, avg between cores')
pl.tight_layout()
pl.show()

pl.loglog(np.sqrt(denom[0]*denom))
pl.title('No spatial avg, denominator, avg between cores')
pl.tight_layout()
pl.show()

pl.loglog(n1234/np.sqrt(denom[0]*denom))
pl.title('No spatial avg, fraction, avg between cores')
pl.tight_layout()
pl.show()
'''