#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 14:24:48 2020

@author: delis
"""

import numpy as np
import gpe as gpe
import os

def g1(i_batch, Kc, Kd, Kc2, rc, rd, uc, ud, sigma, z,
               L, N, dx, dkx, x, kx, hatpsi,
               dt, N_steps, secondarystep, i1, i2, t, 
               n_internal):
    numerator_batch = np.zeros((len(t), int(N/2)), dtype=complex)
    denominator_batch = np.zeros((len(t), int(N/2)))
    density_avg_batch = np.zeros((len(t), int(N/2)))
    for i_n in range(n_internal):
        model = gpe.gpe(Kc=Kc, Kd=Kd, Kc2=0, rc=rc, rd=rd, uc=uc, ud=ud, sigma=sigma, z=z,
                      L=L, N=N, dx=dx, dkx=dkx, x=x, kx=kx, hatpsi=hatpsi,
                      dt=dt, N_steps=N_steps, secondarystep=secondarystep, i1=i1, i2=i2, t=t)
        dens_avg, num, denom = model.time_evolution(i_n)
        density_avg_batch += dens_avg / n_internal
        numerator_batch += num / n_internal
        denominator_batch += denom / n_internal
        print('The core', i_batch, 'has completed realisation number', i_n)
    name_full1 = '/scratch/konstantinos/g1_numerator'+os.sep+'num_'+str(i_batch+1)+'.npy'
    name_full2 = '/scratch/konstantinos/g1_denominator'+os.sep+'denom_'+str(i_batch+1)+'.npy'
    name_dens = '/scratch/konstantinos/avg_sqrt_density'+os.sep+'test_'+str(i_batch+1)+'.npy'
    np.save(name_full1, numerator_batch)
    np.save(name_full2, denominator_batch)
    np.save(name_dens, density_avg_batch)