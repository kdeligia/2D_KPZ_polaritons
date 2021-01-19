#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 15:09:50 2020

@author: delis
"""

c = 3E2 #μm/ps
hbar = 6.582119569 * 1E2 # μeV ps

import os
from scipy.fftpack import fft2, ifft2
import numpy as np
import external as ext
import warnings
from qutip import *
import matplotlib.pyplot as pl
pl.rc('font', family='sans-serif')
pl.rc('text', usetex=True)

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

P = 2.6e2
R = 2e-5

p = P*R / (gamma0*gammar)
ns = gammar/R
n0 = ns*(p-1)
nres = P/(gammar+R*n0)
gr = 0
g = 0.0001

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

time_steps = 20000
dt = 1e-2 * hatt
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
print('--- Dimensionless pump ---')
print('p %.4f' % p)

def finalparams():
    alpha = 1
    beta = 0
    #om = 50*gamma0
    #alpha = 1 + p*gr*gamma0/(hbar*om*R)
    #beta = p*gamma0/(2*om)
    Kc = (hatt/hatx**2) * hbar/(2*m)
    Kd = (hatt/hatx**2) * gamma2/2
    rc = hatt * p*gamma0*gr/(R*hbar)
    rd = hatt * gamma0*(p-1)/2
    ud = hatt/(hatx**2) * p*R*gamma0/(2*gammar)
    uc = hatt/(hbar*hatx**2) * g*(1 - p*(gr/g)*(gamma0/gammar))
    sigma = hatt * gamma0*(p+1)/(2*dx**2)
    z = alpha + beta*1j
    return Kc, Kd, rc, rd, uc, ud, sigma, z

def bogoliubov():
    z=1
    r = (1/z).real
    q = (1/z).imag
    n0 = (rd - z.imag*rc/z.real)/(ud + uc*z.imag/z.real)
    omsol = (rc+n0*uc)/z.real
    a = -z.real*omsol + Kc*kx**2 + rc + 2*n0*uc
    b = -Kd*kx**2 + rd - 2*n0*ud - z.imag*omsol
    c = n0 * uc
    d = -n0 * ud
    im_plus = np.zeros(len(kx))
    im_minus = np.zeros(len(kx))
    #re_plus = np.zeros(len(kx))
    #re_minus = np.zeros(len(kx))
    for i in range(len(kx)):
        if (a[i]**2 - c**2 - d**2) < 0:
            im_plus[i] = b[i]*r + r*np.sqrt(np.abs(a[i]**2 - c**2 - d**2))
            im_minus[i] = b[i]*r - r*np.sqrt(np.abs(a[i]**2 - c**2 - d**2))
            #re_plus[i] = -b[i]*q + q*1j*np.sqrt(np.abs(a[i]**2 - c**2 - d**2))
            #re_minus[i] = -b[i]*q - q*1j*np.sqrt(np.abs(a[i]**2 - c**2 - d**2))
        else:
            im_plus[i] = b[i]*r + q*np.sqrt(a[i]**2 - c**2 - d**2)
            im_minus[i] = b[i]*r - q*np.sqrt(a[i]**2 - c**2 - d**2)
            ##re_plus[i] = -b[i]*q + r*np.sqrt(a[i]**2 - c**2 - d**2)
            #re_minus[i] = -b[i]*q - r*np.sqrt(a[i]**2 - c**2 - d**2)
    return im_plus, im_minus

class model:
    #def __init__(self, Kc, Kd, Kc2, rc, rd, uc, ud, sigma, z, psi_x=0):
        #self.Kc = Kc
        #self.Kd = Kd
        #self.Kc2 = Kc2
        #self.rc = rc
        #self.rd = rd
        #self.uc = uc
        #self.ud = ud
        #self.sigma = sigma
        #self.z = z
        
        #self.psi_x = psi_x
        #self.psi_x = np.full((N, N), 2)
        #self.psi_x /= hatpsi
        #self.psi_mod_k = fft2(self.psi_mod_x)
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
        #return np.exp(-1j*0.5*dt*((self.rc + 1j*self.rd) + (self.uc - 1j*self.ud)*self.n())/self.z)
        return np.exp(-1j*0.5*dt*(self.uc_tilde + self.I_tilde))

    def prefactor_k(self):
        #return np.exp(-1j*dt*((KX**2 + KY**2)*(self.Kc - 1j*self.Kd)-(KX**4 + KY**4)*self.Kc2)/self.z)
        return np.exp(-1j*dt*((KX**2 + KY**2)*(self.Kc - 1j*self.Kd)))

# =============================================================================
# Time evolution
# =============================================================================
    def vortex_count(self):
        count_v = 0
        count_av = 0
        theta = np.angle(self.psi_x)
        grad = np.gradient(theta, dx)
        v_pos = np.zeros((N, N))
        av_pos = np.zeros((N, N))
        list_v = []
        list_av = []
        count_v = 0
        count_av = 0
        for i in range(1, N-1):
            for j in range(1, N-1):
                loop = (2*dx*(grad[0][i+1, j+1] - grad[1][i+1, j+1]) +
                        2*dx*(grad[0][i+1, j-1] + grad[1][i+1, j-1]) +
                        2*dx*(grad[0][i-1, j-1] - grad[1][i-1, j-1]) -
                        2*dx*(grad[0][i-1, j+1] + grad[i-1, j+1]) +
                        2*dx*(grad[0][i+1, j] + grad[1][i. j-1] - grad[0][i-1, j] - grad[1][i, j+1]))
                if loop >= 2 * np.pi:
                    count_v += 1
                elif loop <= -2 * np.pi:
                    count_av +=1
                v_pos[i,j] = 1
                av_pos[i, j] = 1
        list_v.append(count_v)
        list_av.append(count_av)
        return np.array(list_v), np.array(list_av), v_pos, av_pos

    def time_evolution(self, realisation):
        #psi_t = np.zeros(len(t), dtype=complex)
        g1_x = np.zeros(int(N/2), dtype = complex)
        d1_x = np.zeros(int(N/2))
        #n_t = np.zeros(len(t))
        #n_sum = np.zeros(len(t))
        for i in range(time_steps+1):
            self.psi_x *= self.prefactor_x()
            self.psi_mod_k = fft2(self.psi_mod_x)
            self.psi_k *= self.prefactor_k()
            self.psi_mod_x = ifft2(self.psi_mod_k)
            self.psi_x *= self.prefactor_x()
            self.psi_x += np.sqrt(dt) * np.sqrt(self.sigma) * ext.noise((N,N))
            #if i>=i1 and i<=i2 and i%every==0:
                #print(i)
                #psi_t[(i-i1)//every] = self.psi_x[int(N/2), int(N/2)]
                #n_t[(i-i1)//every] = self.n()[int(N/2), int(N/2)]
                #n_sum[(i-i1)//every] = np.mean(self.n())
        '''
        n_t_list = []
        psi_t_list = []
        index = 0
        for i in range(1, len(t)):
            if t[i]%1000==0:
                psi_t_list.append(np.conjugate(psi_t[index])*psi_t[index:i+1])
                n_t_list.append(n_t[index:i+1])
                index = i
        g1_t = np.mean(np.array(psi_t_list), axis=0)
        d1_t = np.mean(np.array(n_t_list), axis=0)
        '''
        ######################################################################################
        for i in range(0, N, int(N/8)):
            g1_x += np.conjugate(self.psi_x[i, int(N/2)]) * self.psi_x[i, int(N/2):] / 8 + np.conjugate(self.psi_x[int(N/2), i]) * self.psi_x[int(N/2):, i] / 8
            d1_x += self.n()[i, int(N/2):] / 8 + self.n()[int(N/2):, i] / 8
        g1_x[0] -= 1/(2*dx**2)
        return g1_x, d1_x, #n_t, n_sum, 

'''
Kc, Kd, rc, rd, uc, ud, sigma, z = finalparams()
print('--- Simulation Parameters ---')
print('Kc', Kc)
print('Kd', Kd)
print('rc', rc)
print('rd', rd)
print('uc', uc)
print('ud', ud)
print('σ', sigma)
print('z', z)
'''

def g1(i_batch):
    g1_x_batch = np.zeros(N, dtype=complex)
    d1_x_batch = np.zeros(N)
    #g1_t_batch = np.zeros(int(10*every+1), dtype=complex)
    #d1_t_batch = np.zeros(int(10*every+1))
    for i_n in range(n_internal):
        gpe = model()
        g1_x, d1_x, g2_x, d2_x, g3_x, d3_x, g4_x, d4_x = gpe.time_evolution(i_n)
        g1_x_batch += g1_x / n_internal
        d1_x_batch += d1_x / n_internal
        #g1_t_batch += g1_t_run / n_internal
        #d1_t_batch += d1_t_run / n_internal
        print('The core', i_batch, 'has completed realisation number', i_n)
    name_g1_x = '/scratch/konstantinos/g1_x'+os.sep+'g1_x'+str(i_batch+1)+'.npy'
    name_d1_x = '/scratch/konstantinos/d1_x'+os.sep+'d1_x'+str(i_batch+1)+'.npy'
    #name_g1_t = '/scratch/konstantinos/g1_t'+os.sep+'g1_t'+str(i_batch+1)+'.npy'
    #name_d1_t = '/scratch/konstantinos/d1_t'+os.sep+'d1_t'+str(i_batch+1)+'.npy'
    np.save(name_g1_x, g1_x_batch)
    np.save(name_d1_x, d1_x_batch)

parallel_map(g1, range(n_batch))
g1_x = ext.ensemble_average_space(r'/scratch/konstantinos/g1_x', int(N/2), n_batch)
d1_x = ext.ensemble_average_space(r'/scratch/konstantinos/d1_x', int(N/2), n_batch)
D1_x = np.sqrt(d1_x[0]*d1_x)
np.save('/home6/konstantinos/g1_x_g0_gr0.npy', np.abs(g1_x))

#g1_t = ext.ensemble_average_time(r'/scratch/konstantinos/g1_t', t, n_batch)
#d1_t = ext.ensemble_average_time(r'/scratch/konstantinos/d1_t', t, n_batch)
#D1_t = np.sqrt(d1_t[0]*d1_t)
#np.save('/Users/delis/Desktop/g1_t_p_1pt89_smallg.npy', np.abs(g1_t))
#np.save('/Users/delis/Desktop/D1_t_p_1pt89_smallg.npy', D1_t)

'''
im_plus, im_minus = bogoliubov()
pl.plot(kx, im_plus, 'o', label=r'Imaginary plus')
pl.plot(kx, im_minus, '^', label=r'Imaginary minus')
pl.axhline(y=0, xmin=kx[0], xmax=kx[-1], linestyle='--', color='black')
pl.xlim(0, kx[-1])
pl.legend()
pl.show()

gpe = model()
g1_x, d1_x, g2_x, d2_x, g3_x, d3_x, g4_x, d4_x, n0, n_sum = gpe.time_evolution(0)
fig,ax = pl.subplots(1,1, figsize=(14,6))
ax.semilogx(t, n_sum, label=r'$\overline{n}$')
ax.semilogx(t, n0, label=r'$n(L/2, L/2)$')
ax.tick_params(axis='both', which='both', direction='in', labelsize=20, pad=12, length=12)
ax.hlines(y=ns*(p-1), xmin=t[0], xmax=t[-1], color='r', label=r'$n_0$')
ax.legend(prop=dict(size=20))
ax.set_xlabel(r'$t$', fontsize=22)
ax.set_ylabel(r'$n$', fontsize=22)
pl.show()

pl.loglog(x[int(N/2):]-x[int(N/2)], np.abs(g1_x)/np.sqrt(d1_x[0]*d1_x), label='2')
pl.loglog(x[int(N/2):]-x[int(N/2)], np.abs(g2_x)/np.sqrt(d2_x[0]*d2_x), label='8')
pl.loglog(x[int(N/2):]-x[int(N/2)], np.abs(g3_x)/np.sqrt(d3_x[0]*d3_x), label='16')
pl.loglog(x[int(N/2):]-x[int(N/2)], np.abs(g4_x)/np.sqrt(d4_x[0]*d4_x), label='32')
pl.legend()
pl.show()
'''