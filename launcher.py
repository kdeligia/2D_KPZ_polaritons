#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 15:09:50 2020

@author: delis
"""

import external as ext
import g1_func
import os
import numpy as np
import pickle
from qutip import *
from toolbox_launcher import *

c = 3E2 #μm/ps
hbar = 6.582119569 * 1E2 # μeV ps
a = 1 * 1E3 #μeV
b = 1.908

systemdict = {}
systemdict['N'] = [int(2**7)]
systemdict['L'] = [int(2**7)]
systemdict['hat_t'] = [1]  #ps
systemdict['hat_x'] = [1]  #μm

paramsdict = {}
paramsdict['star_m'] = [5E-6] 
paramsdict['gamma_0'] = [0.22]  #ps^-1
paramsdict['gamma_r'] = [0.02]  #ps^-1
paramsdict['gamma_2'] = [1/hbar]  #μm^2 ps^-1
paramsdict['P'] = [5.2E2]  # 1/(μm^2 ps)
paramsdict['R'] = [1.6E-5]  #μm^2 ps^-1

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
    N, L, hat_x, hat_t = size['N'], size['L'], size['hat_x'], size['hat_t']
    return N, L, hat_x, hat_t

def init_params():
    params2 = open(fl2[0],'rb')
    params = pickle.load(params2)
    star_m, gamma0, gamma2, gammar, P, R = \
        params['star_m'], params['gamma_0'], params['gamma_2'], params['gamma_r'], \
        params['P'], params['R']
    return star_m, gamma0, gamma2, gammar, P, R

def arrays():
    x_0 = - N * dx / 2
    kx0 = - np.pi / dx
    x = x_0 + dx * np.arange(N)
    kx = kx0 + dkx * np.arange(N)
    return x, kx

N, L, hatx, hatt = init_system()
hatpsi = 1/hatx
L *= hatx
L /= hatx
dx = L/N
dkx = 2 * np.pi / (N * dx)
x, kx =  arrays()

star_m, gamma0, gamma2, gammar, P, R = init_params()
m =  star_m * 0.510998950 * 1E12 / c**2 #μeV/(μm^2/ps^2)
star_gamma_l0 = (gamma0*hbar)  # μeV 
star_gamma_l2 = (gamma2*hbar) # μeV μm^2 
star_gamma_r = (gammar*hbar) # μeV

N_steps = 200000
dt = hatt/100
secondarystep = 200
i1 = 50000
i2 = N_steps
lengthwindow = i2-i1
t = ext.time(dt, N_steps, i1, i2, secondarystep)

keyword = 'yes'
if keyword == 'yes':
    pump = 'no'
    p = P*R / (gamma0*gammar)
    nsat = gammar/R
    n0 = nsat*(p-1)/p
    nres = P/gammar*(1/p)
    gr = a/nres #μeV μm^2
    g = b*a/n0 #μeV μm^2
    #print('--- Checking TWA and energy scales, all ratios should be smaller than 1 ---')
    #print(r'Elos/Ekin %.4f' % (hbar*gamma0/(hbar**2/(2*m*dx**2))))
    #print(r'Eint/Ekin %.4f' % (g*n0/(hbar**2/(2*m*dx**2))))
    #print(r'TWA ratio %.4f' % (g/(hbar*gamma0*dx**2)))
    #print(r'dx/healing length %.4f' % (dx / (hbar/np.sqrt(2*m*g*n0))))
    #print('--- Physical parameters ---')
    #print('gamma_0 in μeV %.4f' % star_gamma_l0)
    #print('gamma_r in μeV %.4f' % star_gamma_r)
    #print('gamma_2 in μeV μm^2 %.4f' % star_gamma_l2)
    #print('Pumping parameter %.4f' % p)
    #print('Polariton-reservoir interaction strength in μeV μm^2 %.4f' % gr)
    #print('Polariton-polariton interaction strength in μev μm^2 %.4f' % g)
    #print('--- Steady-state density ---')
    #print('Steady-state density %.2f' % n0)
    def finalparams(pump):
        if pump == 'no':
            alpha = 1
            beta = 0
        elif pump == 'yes':
            om = 50*gamma0
            alpha = 1 + p*gr*gamma0/(hbar*om*R)
            beta = p*gamma0/(2*om)
        Kc = (hatt/hatx**2) * hbar/(2*m)
        Kd = (hatt/hatx**2) * gamma2/2
        rc = hatt * p*gamma0*gr/(R*hbar)
        rd = hatt * gamma0*(p-1)/2
        ud = hatt/(hatx**2) * p*R*gamma0/(2*gammar)
        uc = hatt/(hbar*hatx**2) * g*(1 - p*(gr/g)*(gamma0/gammar))
        sigma = hatt * gamma0*(p+1)/(4*dx**2)
        z = alpha + beta*1j
        print('--- Simulation Parameters ---')
        print('Kc', Kc)
        print('Kd', Kd)
        print('rc', rc)
        print('rd', rd)
        print('uc', uc)
        print('ud', ud)
        print('σ', sigma)
        print('z', z)
        return Kc, Kd, rc, rd, uc, ud, sigma, z
    Kc, Kd, rc, rd, uc, ud, sigma, z = finalparams(pump)
    '''
    model = gpe.gpe(Kc=Kc, Kd=Kd, Kc2=0, rc=rc, rd=rd, uc=uc, ud=ud, sigma=sigma, z=z,
                      L=L, N=N, dx=dx, dkx=dkx, x=x, kx=kx, hatpsi=hatpsi,
                      dt=dt, N_steps=N_steps, secondarystep=secondarystep, i1=i1, i2=i2, t=t)
    dens_avg, num, denom = model.time_evolution(0)
    pl.plot(t, avg, label=r'$\overline{n}$')
    pl.plot(t, n00, label=r'$n(0,0)$')
    pl.axhline(y=n0, xmin=t[0], xmax=t[-1], color='red', label=r'$n_{steady state, th}$')
    pl.legend()
    '''
    n_tasks = 256
    n_batch = 64
    n_internal = n_tasks//n_batch
    qutip.settings.num_cpus = n_batch
    parallel_map(g1_func.g1, range(n_batch), task_kwargs=dict(Kc=Kc, Kd=Kd, Kc2=0, rc=rc, rd=rd, uc=uc, ud=ud, sigma=sigma, z=z,
                                                                  L=L, N=N, dx=dx, dkx=dkx, x=x, kx=kx, hatpsi=hatpsi,
                                                                  dt=dt, N_steps=N_steps, secondarystep=secondarystep, i1=i1, i2=i2, t=t, n_internal=n_internal))
    path1 = r"/scratch/konstantinos/g1_numerator"
    path2 = r"/scratch/konstantinos/g1_denominator"
    path3 = r"/scratch/konstantinos/avg_sqrt_density"
    numerator = ext.ensemble_average(path1, t, N, n_batch)
    denominator = ext.ensemble_average(path2, t, N, n_batch)
    avg_sqrt_density = ext.ensemble_average(path3, t, N, n_batch)
    avg_sqrt_density *= avg_sqrt_density[0,0]
    np.savetxt('/home6/konstantinos/g1_v1_50k_200_200k.dat', -2*np.log(np.abs(numerator)/denominator))
    np.savetxt('/home6/konstantinos/g1_v2_50k_200_200k.dat', -2*np.log(np.abs(numerator)/avg_sqrt_density))
    np.savetxt('/home6/konstantinos/denominator_50k_200_200k.dat', denominator)
    np.savetxt('/home6/konstantinos/avg_sqrt_density_50k_200_200k.dat', avg_sqrt_density)

'''
elif keyword == 'no':
    pump = input('Use frequency-dependent pump? ')
    p = P*R / (gamma0*gammar)
    nsat = gammar/R
    n0 = nsat*(p-1)
    nres = P/gammar*p
    gr = a/nres #μeV μm^2
    g = b*a/n0 #μeV μm^2
    print('--- Checking TWA and energy scales ---')
    print(r'Elos/Ekin %.4f' % (hbar*gamma0/(hbar**2/(2*m*dx**2))))
    print(r'Eint/Ekin %.4f' % (g*n0/(hbar**2/(2*m*dx**2))))
    print(r'TWA ratio %.4f' % (g/(hbar*gamma0*dx**2)))
    print(r'dx/healing length %.4f' % (dx/(hbar / np.sqrt(2*m*g*n0))))
    print('--- Params of the simulation---')
    print('Pumping parameter %.2f' % p)
    print('Polariton-reservoir interaction strength %.4f' % gr)
    print('Polariton-polariton interaction strength %.4f' % g)
    print('Steady-state density %.2f' % n0)
    print('Saturation density %.2f' % nsat)
    if pump == 'yes':
        om = 50*gamma0
        d1 = p*gamma0*gr/(hbar*R*om)
        d2 = gamma0*p/(2*om)
    elif pump == 'no':
        d1 = 0
        d2 = 0
    GPE = testing_noexp.gpe(m=m, gamma2=gamma2, g=g, gr=gr, P=P, R=R, gamma0=gamma0, gammar=gammar, ns=nsat,
                 L=L, N=N, dx=dx, dkx=dkx, x=x, kx=kx, hatx=hatx,
                 hatt=hatt, dt=dt, N_steps=N_steps, secondarystep=secondarystep, i1=i1, i2=i2, t=t, d1=d1, d2=d2)
    avg, n00 = GPE.time_evolution(1)
    pl.plot(t, avg, label=r'$\overline{n}$')
    pl.plot(t, n00, label=r'$n(0,0)$')
    pl.axhline(y=n0, xmin=t[0], xmax=t[-1], color='red', label=r'$n_{steady state, th}$')
    pl.legend()
    pl.show()
'''
'''
    def bogoliubov():
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
        re_plus = np.zeros(len(kx))
        re_minus = np.zeros(len(kx))
        for i in range(len(kx)):
            if (a[i]**2 - c**2 - d**2) < 0:
                im_plus[i] = b[i]*r + r*np.sqrt(np.abs(a[i]**2 - c**2 - d**2))
                im_minus[i] = b[i]*r - r*np.sqrt(np.abs(a[i]**2 - c**2 - d**2))
                re_plus[i] = -b[i]*q + q*1j*np.sqrt(np.abs(a[i]**2 - c**2 - d**2))
                re_minus[i] = -b[i]*q - q*1j*np.sqrt(np.abs(a[i]**2 - c**2 - d**2))
            else:
                im_plus[i] = b[i]*r + q*np.sqrt(a[i]**2 - c**2 - d**2)
                im_minus[i] = b[i]*r - q*np.sqrt(a[i]**2 - c**2 - d**2)
                re_plus[i] = -b[i]*q + r*np.sqrt(a[i]**2 - c**2 - d**2)
                re_minus[i] = -b[i]*q - r*np.sqrt(a[i]**2 - c**2 - d**2)
        pl.plot(kx/np.sqrt(2*m*gamma0/hbar), im_plus/gamma0, label=r'Imaginary plus')
        pl.plot(kx/np.sqrt(2*m*gamma0/hbar), im_minus/gamma0, label=r'Imaginary minus')
        pl.axhline(y=0, xmin=kx[0], xmax=kx[-1], linestyle='--', color='black')
        pl.xlim(0, kx[-1])
        pl.legend()
        pl.show()
        pl.plot(kx/np.sqrt(2*m*gamma0/hbar), re_plus/gamma0, label=r'Real plus')
        pl.plot(kx/np.sqrt(2*m*gamma0/hbar), re_minus/gamma0, label=r'Real minus')
        pl.axhline(y=0, xmin=kx[0], xmax=kx[-1], linestyle='--', color='black')
        pl.legend()
        pl.xlim(0, kx[-1])
        pl.show()
'''