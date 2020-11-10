#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  3 14:02:54 2020

@author: delis
"""

from qutip import *
from mpl_toolkits import mplot3d
from scipy.fftpack import fft2, ifft2
import os
import numpy as np
import external as ext
import matplotlib.pyplot as pl
import warnings
warnings.filterwarnings("ignore", message="Casting complex values to real discards the imaginary part")

class GrossPitaevskii:
    def __init__(self, Kc, Kd, Kc2, rc, rd, uc, ud, sigma, psi_x=0):

        self.X, self.Y= np.meshgrid(x,x)
        self.KX, self.KY = np.meshgrid(kx, kx)

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
        self.psi_x = np.full((N, N), 5*hatpsi)
        self.psi_x /= hatpsi
        self.psi_mod_k = fft2(self.psi_mod_x)
# =============================================================================
# Vortices
# =============================================================================
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
        return np.exp(-1j*0.5*dt*((self.rc + 1j*self.rd)+(self.uc - 1j*self.ud)*wave_fn*np.conjugate(wave_fn)))

    def prefactor_k(self):
        return np.exp(-1j*dt*((self.KX**2 + self.KY**2)*(self.Kc - 1j*self.Kd)-(self.KX**4 + self.KY**4)*self.Kc2))

# =============================================================================
# Time evolution and Phase unwinding
# =============================================================================
    def time_evolution(self, realisation):
        psi = np.zeros((len(t),int(N/2)), dtype=complex)
        for i in range(N_steps+1):
            self.psi_x *= self.prefactor_x(self.psi_x)
            self.psi_mod_k = fft2(self.psi_mod_x)
            self.psi_k *= self.prefactor_k()
            self.psi_mod_x = ifft2(self.psi_mod_k)
            self.psi_x *= self.prefactor_x(self.psi_x)
            self.psi_x += np.sqrt(self.sigma) * np.sqrt(dt) * ext.noise((N,N))
            if i>=i1 and i<=i2 and i%secondarystep==0:
                psi[(i-i1)//secondarystep]= self.psi_x[int(N/2), int(N/2):]
            '''
            if i>=i1 and i<=i2 and i%secondarystep==0:
                name = '/Users/delis/Desktop/figures'+os.sep+'f'+str((i-i1)//secondarystep)+'.png'
                fig,ax = pl.subplots(2,1, figsize=(8,8))
                c1 = ax[0].pcolormesh(X, Y, np.abs(self.psi_x*np.conjugate(self.psi_x)), cmap='viridis')
                ax[0].set_title('Density')
                ax[0].axis([x.min(), x.max(), x.min(), x.max()])
                fig.colorbar(c1, ax=ax[0])
                c2 = ax[1].pcolormesh(X, Y, np.angle(self.psi_x), cmap='cividis')
                ax[1].set_title('Phase')
                ax[1].axis([x.min(), x.max(), x.min(), x.max()])
                fig.colorbar(c2, ax=ax[1])
                pl.savefig(name, format='png')
                pl.show()
            '''
        return psi
# =============================================================================
# Input typical values
# =============================================================================
tstar=0.1 #should typically be much lower than typical polariton lifetime in 2D which is 6 up until 150
xstar=1
mstar=-3.4E-6

gamma_0star=90 #can be lower for better 2D sample, ask experimentalists
gamma_2star=1E4
gamma_rstar=5
gamma=8
gstar=2
grstar=4
p=1.4

# =============================================================================
# Adimensional parameters
# =============================================================================
hatx = xstar * 1E-6 # metre
hatpsi = 1/hatx # 1/metre
hatt = tstar * 1E-12 # second

c = 3E8
hbar = 6.582119569 * 1E-16 # eV times second
hatm = mstar*0.510998950 * 1E6 #eV/c^2
hatgamma_l0 = (gamma_0star/hbar) * 1E-6 #1/second
hatgamma_l2 = (gamma_2star/hbar) * 1E-6*1E-12 #metre^2/second
hatgamma_r = (gamma_rstar/hbar) * 1E-6 #1/second
hatg = gstar * 1E-12 * 1E-6 #metre^2 * eV
hatg_r = grstar * 1E-12 * 1E-6 #metre^2 * eV

N = 2 ** 7
L = 2 ** 7 * hatx

L /= hatx
dx = L / N
dy = L / N
dkx = 2 * np.pi / (N * dx)
dky = 2 * np.pi / (N * dy)

def params(hatx, hatt, hatpsi):
    Kc = hbar*hatt/(2*hatm*hatx**2/c**2)
    Kd = hatgamma_l2*hatt/(2*hatx**2)
    rc = 2*hatt*hatgamma_l0*gamma*p
    rd = (p-1)*hatt*hatgamma_l0/2
    uc = hatg*hatt/(hbar*hatx**2)*(0 - 2*hatgamma_l0*hatg_r*p/(hatgamma_r*hatg))
    ud = (p*hatt/(2*hatx**2))*hatgamma_l0*hatg_r/(hbar*gamma*hatgamma_r)
    sigma = (p+1)/2 * hatgamma_l0*hatt
    print('-----PARAMS-----')
    print('Kc', Kc)
    print('Kd', Kd)
    print('rc', rc)
    print('rd', rd)
    print('uc', uc)
    print('ud', ud)
    print('σ', sigma)
    return Kc, Kd, rc, rd, uc, ud, sigma
Kc, Kd, rc, rd, uc, ud, sigma = params(hatx, hatt,hatpsi)

'''
print('----STEADY STATE----')
print('-rc/uc', -rc/uc)
print('rd/ud', rd/ud)
'''

def arrays():
    x_0 = - N * dx / 2
    kx0 = - np.pi / dx
    x = x_0 + dx * np.arange(N)
    kx = kx0 + dkx * np.arange(N)
    return x, kx

x, kx =  arrays()
X,Y = np.meshgrid(x, x)
KX, KY = np.meshgrid(kx, kx)

N_steps = 200000
dt = tstar/20
secondarystep = 1000
i1 = 50000
i2 = N_steps
lengthwindow = i2-i1
t = ext.time(dt, N_steps, i1, i2, secondarystep)

#GP = GrossPitaevskii(Kc=Kc, Kd=Kd, Kc2=0, rc=rc, rd=rd, uc=uc, ud=ud, sigma=0.1)
# =============================================================================
# Computation
# =============================================================================
'''
n_tasks = 750
n_batch = 50
n_internal = n_tasks//n_batch

def g1(i_batch):
    sqrtrho_batch = np.zeros((len(t), int(N/2)), dtype=complex)
    correlator_batch = np.zeros((len(t), int(N/2)), dtype=complex)
    for i_n in range(n_internal):
        GP = GrossPitaevskii(Kc=Kc, Kd=Kd, Kc2=0, rc=rc, rd=rd, uc=uc, ud=ud, sigma=sigma)
        psi = GP.time_evolution(i_n)
        sqrtrho = np.sqrt(np.conjugate(psi) * psi)
        for i in range(len(t)):
            psi[i] *= np.conjugate(psi[i,0])
            sqrtrho[i] *= sqrtrho[i,0]
        correlator_batch += psi / n_internal
        sqrtrho_batch += sqrtrho / n_internal
        if i_n>0:
            print('The core', i_batch, 'has completed realisation number', i_n)
    name_full1 = '/scratch/konstantinos/numerator_batch'+os.sep+'n'+str(i_batch+1)+'.dat'
    name_full2 = '/scratch/konstantinos/denominator_batch'+os.sep+'d'+str(i_batch+1)+'.dat'
    np.savetxt(name_full1, correlator_batch, fmt='%.5f')
    np.savetxt(name_full2, sqrtrho_batch, fmt='%.5f')

qutip.settings.num_cpus = n_batch
parallel_map(g1, range(n_batch))

path1 = r"/scratch/konstantinos/numerator_batch"
path2 = r"/scratch/konstantinos/denominator_batch"

numerator = ext.ensemble_average(path1)
denominator = ext.ensemble_average(path2)
result = np.absolute(numerator)/denominator

tosave = '/home6/konstantinos'+os.sep+'σ'+str(sigma)+'_p'+str(p) + \
    '_γ'+str(gamma)+'_g'+str(gstar)+'_gr'+str(grstar)+'_spatial'+'.dat'
np.savetxt(tosave, result)
'''