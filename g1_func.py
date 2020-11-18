#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 14:24:48 2020

@author: delis
"""

import numpy as np
import gpe as gpe
import os

def g1(i_batch, Kc, Kd, Kc2, rc, rd, uc, ud, sigma,
               L, N, dx, dkx, x, kx, hatpsi,
               dt, N_steps, secondarystep, i1, i2, t, 
               n_internal):
    sqrtrho_batch = np.zeros((len(t), int(N/2)), dtype=complex)
    correlator_batch = np.zeros((len(t), int(N/2)), dtype=complex)
    for i_n in range(n_internal):
        myGPE = gpe.gpe(Kc=Kc, Kd=Kd, Kc2=0, rc=rc, rd=rd, uc=uc, ud=ud, sigma=sigma,
                      L=L, N=N, dx=dx, dkx=dkx, x=x, kx=kx, hatpsi=hatpsi,
                      dt=dt, N_steps=N_steps, secondarystep=secondarystep, i1=i1, i2=i2, t=t)
        psi = myGPE.time_evolution(i_n)
        sqrtrho = np.sqrt(np.conjugate(psi) * psi)
        for i in range(len(t)):
            psi[i] *= np.conjugate(psi[i,0])
            sqrtrho[i] *= sqrtrho[i,0]
        correlator_batch += psi / n_internal
        sqrtrho_batch += sqrtrho / n_internal
        if i_n>0:
            print('The core', i_batch, 'has completed realisation number', i_n)
    #name_full1 = '/Users/delis/Desktop/numerator_batch'+os.sep+'n'+str(i_batch+1)+'.dat'
    #name_full2 = '/Users/delis/Desktop/denominator_batch'+os.sep+'d'+str(i_batch+1)+'.dat'
    #np.savetxt(name_full1, correlator_batch, fmt='%.5f')
    #np.savetxt(name_full2, sqrtrho_batch, fmt='%.5f')