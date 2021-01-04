#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 14:24:48 2020

@author: delis
"""

import external
import numpy as np
import gpe as gpe
import os

time_steps = 400000
every = 500
i1 = 50000
i2 = time_steps
lengthwindow = i2-i1

def g1(i_batch, Kc, Kd, Kc2, rc, rd, uc, ud, sigma, z,
               L, N, dx, dkx, x, kx, hatpsi,
               dt, n_internal):
    t = external.time(dt, time_steps, i1, i2, every)
    cor_psi_batch = np.zeros((len(t), int(N/2)), dtype=complex)
    d2_batch = np.zeros((len(t), int(N/2)))
    d1_batch = np.zeros((len(t), int(N/2)))
    for i_n in range(n_internal):
        model = gpe.gpe(Kc=Kc, Kd=Kd, Kc2=0, rc=rc, rd=rd, uc=uc, ud=ud, sigma=sigma, z=z,
                      L=L, N=N, dx=dx, dkx=dkx, x=x, kx=kx, hatpsi=hatpsi, dt=dt)
        cor_psi, d2, d1 = model.time_evolution(i_n)
        cor_psi_batch += cor_psi / n_internal
        d2_batch += d2 / n_internal
        d1_batch += d1 / n_internal
        print('The core', i_batch, 'has completed realisation number', i_n)
    name_cor_psi = '/scratch/konstantinos/cor_psi'+os.sep+'cor_psi'+str(i_batch+1)+'.npy'
    name_d1 = '/scratch/konstantinos/d1'+os.sep+'d1_'+str(i_batch+1)+'.npy'
    name_d2 = '/scratch/konstantinos/d2'+os.sep+'d2_'+str(i_batch+1)+'.npy'
    np.save(name_cor_psi, cor_psi_batch)
    np.save(name_d1, d1_batch)
    np.save(name_d2, d2_batch)