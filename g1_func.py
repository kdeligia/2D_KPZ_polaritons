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
    C1_batch = np.zeros((len(t), int(N/2)))
    C2_batch = np.zeros((len(t), int(N/2)))
    for i_n in range(n_internal):
        myGPE = gpe.gpe(Kc=Kc, Kd=Kd, Kc2=0, rc=rc, rd=rd, uc=uc, ud=ud, sigma=sigma,
                      L=L, N=N, dx=dx, dkx=dkx, x=x, kx=kx, hatpsi=hatpsi,
                      dt=dt, N_steps=N_steps, secondarystep=secondarystep, i1=i1, i2=i2, t=t)
        psi_x = myGPE.time_evolution(i_n)
        norm = np.abs(np.conjugate(psi_x) * psi_x) - 1/(2*dx**2)
        C1_batch += norm / n_internal
        C2_batch += np.sqrt(norm[0,0]) * np.sqrt(norm) / n_internal
        if i_n>0:
            print('The core', i_batch, 'has completed realisation number', i_n)
    name_full1 = '/home6/konstantinos/C1_batch'+os.sep+'C1_'+str(i_batch+1)+'.npy'
    name_full2 = '/home6/konstantinos/C2_batch'+os.sep+'C2_'+str(i_batch+1)+'.npy'
    np.save(name_full1, C1_batch)
    np.save(name_full2, C2_batch)