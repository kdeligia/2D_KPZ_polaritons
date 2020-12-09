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
    avg_nx_batch = np.zeros((len(t), int(N/2)))
    g1_batch = np.zeros((len(t), int(N/2)), dtype=complex)
    g2_batch = np.zeros((len(t), int(N/2)))
    for i_n in range(n_internal):
        myGPE = gpe.gpe(Kc=Kc, Kd=Kd, Kc2=0, rc=rc, rd=rd, uc=uc, ud=ud, sigma=sigma,
                      L=L, N=N, dx=dx, dkx=dkx, x=x, kx=kx, hatpsi=hatpsi,
                      dt=dt, N_steps=N_steps, secondarystep=secondarystep, i1=i1, i2=i2, t=t)
        psi_x = myGPE.time_evolution(i_n)
        avg_nx_batch +=  np.abs(np.conjugate(psi_x) * psi_x) / n_internal
        g1_batch += np.conjugate(psi_x[0,0]) * psi_x / n_internal
        g2_batch += np.abs(np.conjugate(psi_x) * psi_x)[0,0] * np.abs(np.conjugate(psi_x) * psi_x) / n_internal
        if i_n>0:
            print('The core', i_batch, 'has completed realisation number', i_n)
    name_full1 = '/scratch/konstantinos/g1_batch'+os.sep+'g1_'+str(i_batch+1)+'.npy'
    name_full2 = '/scratch/konstantinos/g2_batch'+os.sep+'g2_'+str(i_batch+1)+'.npy'
    name_full3 = '/scratch/konstantinos/avg_nx_batch'+os.sep+'avg_nx'+str(i_batch+1)+'.npy'
    np.save(name_full1, g1_batch)
    np.save(name_full2, g2_batch)
    np.save(name_full3, avg_nx_batch)