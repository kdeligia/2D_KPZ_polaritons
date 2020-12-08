#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 15:09:50 2020

@author: delis
"""

import matplotlib.pyplot as pl
import external as ext
import g1_func as g1
import os
import numpy as np
import pickle
from qutip import *
from toolbox_launcher import *

c = 3E8
hbar = 6.582119569 * 1E-16 # eV times second

systemdict = {}
systemdict['N'] = [int(2**6)]
systemdict['L'] = [int(2**6)]
systemdict['t_star'] = [0.1]
systemdict['x_star'] = [1]

paramsdict = {}
paramsdict['mstar'] = [-3.4E-6]
paramsdict['gamma0star'] = [90]
paramsdict['gammarstar'] = [4]
paramsdict['gamma2star'] = [1E4]
paramsdict['gammatilde'] = [8]
paramsdict['gstar'] = [0.1]
paramsdict['grstar'] = [4]
paramsdict['p'] = [1.4]

d1 = get_Ilist(systemdict)
dname1 = 'data_system'+os.sep
if not os.path.exists(dname1):
    os.makedirs(dname1)
fl1 = [get_input_file_python(x, prefix=dname1) for x in d1]

d2 = get_Ilist(paramsdict)
dname2 = 'data_params'+os.sep
if not os.path.exists(dname2):
    os.makedirs(dname2)
fl2 = [get_input_file_python(x, prefix=dname2) for x in d2]

def init_system():
    params1 = open(fl1[0],'rb')
    size = pickle.load(params1)
    N, L, xstar, tstar = size['N'], size['L'], size['x_star'], size['t_star']
    return N, L, xstar, tstar

def init_params():
    params2 = open(fl2[0],'rb')
    params = pickle.load(params2)
    m_star, gamma0_star, gamma2_star, gammar_star, gamma, g_star, gr_star, p = \
        params['mstar'], params['gamma0star'], params['gamma2star'], params['gammarstar'], \
            params['gammatilde'], params['gstar'], params['grstar'], params['p']
    return m_star, gamma0_star, gamma2_star, gammar_star, gamma, g_star, gr_star, p

def hats():
    hatx = xstar * 1E-6 # metre
    hatpsi = 1/hatx # 1/metre
    hatt = tstar * 1E-12 # second
    hatm = m_star*0.510998950 * 1E6 #eV/c^2
    hatgamma_l0 = (gamma0_star/hbar) * 1E-6 #1/second
    hatgamma_l2 = (gamma2_star/hbar) * 1E-6*1E-12 #metre^2/second
    hatgamma_r = (gammar_star/hbar) * 1E-6 #1/second
    hatg = g_star * 1E-12 * 1E-6 #metre^2 * eV
    hatg_r = gr_star * 1E-12 * 1E-6 #metre^2 * eV
    return hatx, hatpsi, hatt, hatm, hatgamma_l0, hatgamma_l2, hatgamma_r, hatg, hatg_r

def arrays():
    x_0 = - N * dx / 2
    kx0 = - np.pi / dx
    x = x_0 + dx * np.arange(N)
    kx = kx0 + dkx * np.arange(N)
    return x, kx

def finalparams(hatx, hatt, hatpsi):
    Kc = hbar*hatt/(2*hatm*hatx**2/c**2)
    Kd = hatgamma_l2*hatt/(2*hatx**2)
    rc = 2*hatt*hatgamma_l0*gamma*p
    rd = (p-1)*hatt*hatgamma_l0/2
    uc = hatg*hatt/(hbar*hatx**2)*(0 - 2*hatgamma_l0*hatg_r*p/(hatgamma_r*hatg))
    ud = (p*hatt/(2*hatx**2))*hatgamma_l0*hatg_r/(hbar*gamma*hatgamma_r)
    sigma = hatgamma_l0 * hatt * (p+1) / 2
    #print('-----PARAMS-----')
    #print('Kc', Kc)
    #print('Kd', Kd)
    #print('rc', rc)
    #print('rd', rd)
    #print('uc', uc)
    #print('ud', ud)
    #print('σ', sigma)
    return Kc, Kd, rc, rd, uc, ud, sigma

N, L, xstar, tstar = init_system()
m_star, gamma0_star, gamma2_star, gammar_star, gamma, g_star, gr_star, p = init_params()
hatx, hatpsi, hatt, hatm, hatgamma_l0, hatgamma_l2, hatgamma_r, hatg, hatg_r = hats()

L *= hatx
L /= hatx
dx = L / N
dkx = 2 * np.pi / (N * dx)

x, kx =  arrays()
Kc, Kd, rc, rd, uc, ud, sigma = finalparams(hatx, hatt, hatpsi)

N_steps = 200000
dt = tstar/10
secondarystep = 200
i1 = 0
i2 = N_steps
lengthwindow = i2-i1
t = ext.time(dt, N_steps, i1, i2, secondarystep)

'''
n_tasks = 100
n_batch = 4
n_internal = n_tasks//n_batch
qutip.settings.num_cpus = n_batch
print('A total of', n_internal, 'realisations will be performed per core.')
parallel_map(g1.g1_alt, range(n_batch), task_kwargs=dict(Kc=Kc, Kd=0, Kc2=0, rc=rc, rd=rd, uc=uc, ud=ud, sigma=sigma,
                                                              L=L, N=N, dx=dx, dkx=dkx, x=x, kx=kx, hatpsi=hatpsi,
                                                              dt=dt, N_steps=N_steps, secondarystep=secondarystep, i1=i1, i2=i2, t=t,
                                                              n_internal=n_internal))
'''
'''
myGPE = gpe.gpe(Kc=Kc, Kd=Kd, Kc2=0, rc=rc, rd=rd, uc=uc, ud=ud, sigma=sigma,
                      L=L, N=N, dx=dx, dkx=dkx, x=x, kx=kx, hatpsi=hatpsi,
                      dt=dt, N_steps=N_steps, secondarystep=secondarystep, i1=i1, i2=i2, t=t)
'''

'''
path1 = r"/Users/delis/Desktop/numerator_batch"
path2 = r"/Users/delis/Desktop/denominator_batch"

numerator = ext.ensemble_average(path1)
denominator = ext.ensemble_average(path2)
result = -2*np.log(np.absolute(numerator)/denominator)

np.savetxt('/Users/delis/Desktop/spatial_correlation.dat', result.real)
'''