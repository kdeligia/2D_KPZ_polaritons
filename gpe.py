#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  3 14:02:54 2020

@author: delis
"""

import matplotlib.pyplot as pl
from scipy.fftpack import fft2, ifft2
import numpy as np
import external as ext
import warnings
warnings.filterwarnings("ignore", message="Casting complex values to real discards the imaginary part")

class gpe:
    def __init__(self, Kc, Kd, Kc2, rc, rd, uc, ud, sigma, 
                 L, N, dx, dkx, x, kx, hatpsi, 
                 dt, N_steps, secondarystep, i1, i2, t,
                 psi_x=0):
# =============================================================================
#       Initialitze
# =============================================================================
        self.x = x
        self.kx =kx
        self.X, self.Y= np.meshgrid(self.x,self.x)
        self.KX, self.KY = np.meshgrid(self.kx, self.kx)
        self.L = L
        self.N = N
        self.dx = dx
        self.dkx = dkx
        self.hatpsi = hatpsi
# =============================================================================
#       Time
# =============================================================================
        self.dt=dt
        self.N_steps=N_steps
        self.secondarystep=secondarystep
        self.i1=i1
        self.i2=i2
        self.t=t
# =============================================================================
#       Params
# =============================================================================
        self.Kc = Kc
        self.Kd = Kd
        self.Kc2 = Kc2
        self.rc = rc
        self.rd = rd
        self.uc = uc
        self.ud = ud
        self.sigma = sigma
# =============================================================================
# Initialize ψ
# =============================================================================
        self.psi_x = psi_x
        self.psi_x = np.full((self.N, self.N), 5*self.hatpsi)
        self.psi_x /= self.hatpsi
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
        self.psi_mod_x = psi_x * np.exp(-1j * self.KX[0,0] * self.X - 1j * self.KY[0,0] * self.Y) * self.dx * self.dx / (2 * np.pi)

    def _get_psi_x(self):
        return self.psi_mod_x * np.exp(1j * self.KX[0,0] * self.X + 1j * self.KY[0,0] * self.Y) * 2 * np.pi / (self.dx * self.dx)

    def _set_fourier_psi_k(self, psi_k):
        self.psi_mod_k = psi_k * np.exp(1j * self.X[0,0] * self.dkx * np.arange(self.N) + 1j * self.Y[0,0] * self.dkx * np.arange(self.N))

    def _get_psi_k(self):
        return self.psi_mod_k * np.exp(-1j * self.X[0,0] * self.dkx * np.arange(self.N) - 1j * self.Y[0,0] * self.dkx * np.arange(self.N))

    psi_x = property(_get_psi_x, _set_fourier_psi_x)
    psi_k = property(_get_psi_k, _set_fourier_psi_k)
# =============================================================================
# Definition of the split steps
# =============================================================================
    def prefactor_x(self, wave_fn):
        return np.exp(-1j*0.5*self.dt*((self.rc + 1j*self.rd)+(self.uc - 1j*self.ud)*wave_fn*np.conjugate(wave_fn)))

    def prefactor_k(self):
        return np.exp(-1j*self.dt*((self.KX**2 + self.KY**2)*(self.Kc - 1j*self.Kd)-(self.KX**4 + self.KY**4)*self.Kc2))

# =============================================================================
# Time evolution and Phase unwinding
# =============================================================================
    def time_evolution(self, realisation):
        psix = np.zeros((len(self.t), int(self.N/2)), dtype=complex)
        #l=np.zeros(len(self.t))
        for i in range(self.N_steps+1):
            self.psi_x *= self.prefactor_x(self.psi_x)
            self.psi_mod_k = fft2(self.psi_mod_x)
            self.psi_k *= self.prefactor_k()
            self.psi_mod_x = ifft2(self.psi_mod_k)
            self.psi_x *= self.prefactor_x(self.psi_x)
            self.psi_x += np.sqrt(self.sigma * self.dt / self.dx**2) * ext.noise((self.N,self.N))
            if i>=self.i1 and i<=self.i2 and i%self.secondarystep==0:
                #l[(i-self.i1)//self.secondarystep] = np.abs(self.psi_x*np.conjugate(self.psi_x))[int(self.N/2), int(self.N/2)]
                psix[(i-self.i1)//self.secondarystep] = self.psi_x[int(self.N/2), int(self.N/2):]
                '''
                #name = '/Users/delis/Desktop/figures'+os.sep+'f'+str((i-i1)//secondarystep)+'.png'
                fig,ax = pl.subplots(2,1, figsize=(8,8))
                c1 = ax[0].pcolormesh(self.X, self.Y, np.abs(self.psi_x*np.conjugate(self.psi_x)), cmap='viridis')
                ax[0].set_title('Density')
                ax[0].axis([self.x.min(), self.x.max(), self.x.min(), self.x.max()])
                fig.colorbar(c1, ax=ax[0])
                c2 = ax[1].pcolormesh(self.X, self.Y, np.angle(self.psi_x), cmap='cividis')
                ax[1].set_title('Phase')
                ax[1].axis([self.x.min(), self.x.max(), self.x.min(), self.x.max()])
                fig.colorbar(c2, ax=ax[1])
                #pl.savefig(name, format='png')
                pl.show()
                '''
        return psix