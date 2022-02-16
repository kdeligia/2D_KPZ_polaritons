#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 19 13:53:23 2021

@author: delis
"""

from qutip import *
import os
import numpy as np
import external as ext
import model_script
import itertools

path = r'/scratch/konstantinos' + os.sep + 'simulations_vortices'
if os.path.isdir(path) == False:
    os.mkdir(path)

params_init = {}
params_init['l0'] = [4 * 2 ** (1/2)]                                            # μm
params_init['tau0'] = [params_init.get('l0')[0] ** 2]                           # ps
params_init['N'] = [5 * 64 * 4 * 2 ** (1/2) / params_init.get('l0')[0]]         # dimensionless!
params_init['dx'] = [0.5 / 5]                                                   # dimensionless!
params_init['m'] = [8e-5]                                                       # will multiply m_el in model_script.py
params_init['p'] = [2]                                                          # dimensionless!
params_init['gamma0'] = [0.3125 * 32 / params_init.get('tau0')[0]]              # ps^-1
params_init['gamma2'] = [0.1]                                                   # μm^2 ps^-1
params_init['g'] = [0]                                                          # μeV μm^-2
params_init['gr'] = [0]                                                         # μeV μm^-2
params_init['ns'] = [3.75 * (4 * 2 ** (1/2)) ** 2 / params_init.get('l0')[0] ** 2]# μm^-2

dt = 5e-5                                                                       # dimensionless!
di = 1                                                                          # sample step
N_input = 3.125e5                                                               # number of time steps
time = {}
time['dt'] = dt
time['di'] = di
time['N_input'] = N_input

t=[]
for i in range(0, int(N_input)+1, di):
    t.append(i*dt*params_init.get('tau0')[0])
x, y = ext.space_grid(params_init.get('N')[0], params_init.get('dx')[0])

np.savetxt(path + os.sep + 
           'N' + str(int(params_init.get('N')[0])) + '_' + 
           'dx' + str(params_init.get('dx')[0]) + '_' + 
           'unit' + str(params_init.get('l0')[0]) + '_' + 'x' + '.dat', x * params_init.get('l0')[0])
np.savetxt(path + os.sep + 
           'N' + str(int(params_init.get('N')[0])) + '_' + 
           'dx' + str(params_init.get('dx')[0]) + '_' + 
           'unit' + str(params_init.get('l0')[0]) + '_' + 'y' + '.dat', y * params_init.get('l0')[0])
np.savetxt(path + os.sep + 
           'Ninput' + str(int(time.get('N_input'))) + '_' + 
           'dt' + str(time.get('dt')) + '_' + 
           'unit' + str(params_init.get('tau0')[0]) + '_' + 't' + '.dat', np.array(t))

keys = params_init.keys()
values = (params_init[key] for key in keys)
params = [dict(zip(keys, combination)) for combination in itertools.product(*values)]
for i in range(len(params)):
    params[i]['number'] = i + 1
qutip.settings.num_cpus = len(params)

def evolution_vortices(i_dict, savepath):
    for i in range(len(params)):
        if params[i]['number'] == i_dict + 1:
            current_dict = params[i]
            break
    print('Simulations parameters')
    print('----------------------')
    print(current_dict)
    print('----------------------')
    p = current_dict.get('p')
    gamma0 = current_dict.get('gamma0')
    gamma2 = current_dict.get('gamma2')
    sigma = params_init['tau0'][0] * gamma0 * (p + 1) / 4
    g = current_dict.get('g')
    gr = current_dict.get('gr')
    ns = current_dict.get('ns')
    m = current_dict.get('m')
    current_dict['sigma'] = sigma

    name = 'm' + str(m) + '_' + 'p' + str(p) + '_' + 'gamma' + str(gamma0) + '_' + 'gammak' + str(gamma2) + '_' + 'g' + str(g) + '_' + 'gr' + str(gr) + '_'  + 'ns' + str(ns)
    gpe = model_script.gpe(**current_dict)
    n, theta_unwrapped = gpe.time_evolution_spacetime_vortices(np.pi, **time)
    #np.save(savepath + os.sep + name + '_' + 'theta_unwrapped' + '.npy', theta_unwrapped)
    np.save(savepath + os.sep + name + '_' + 'density' + '.npy', theta_unwrapped)
    return None

#parallel_map(evolution, range(len(params)), task_kwargs = dict(home = final_save_path))
evolution_vortices(0, savepath = path)

# =============================================================================
# 
# =============================================================================
'''
import matplotlib.pyplot as pl
for mydict in params:
    k, plus, minus = ext.bogoliubov(**mydict)
    k_phys = np.fft.fftshift(k)
    fig, ax = pl.subplots()
    ax.plot(k_phys, plus)
    ax.plot(k_phys, minus)
    fig.show()
'''

'''
import math as m

x,y = ext.space_grid(params_init.get('N')[0], params_init.get('dx')[0])
theta = np.zeros((params_init.get('N')[0], params_init.get('N')[0]))

for i in range(params_init.get('N')[0]):
    for j in range(params_init.get('N')[0]):
        theta[i, j] = m.atan2(y[j], x[i])
X, Y = np.meshgrid(x, y)
partialx, partialy = np.gradient(theta, params_init.get('dx')[0])
fig,ax = pl.subplots()
im = ax.pcolormesh(X, Y, theta)
fig.colorbar(im, ax=ax)
fig.show()

vortex_plots = ext.vortex_plots_class()
positions = ext.vortex_detect(theta, params_init.get('N')[0], params_init.get('dx')[0], x, y)
vortex_plots(r'/Users/delis/Desktop', x, 0, positions, theta, np.zeros_like(theta))
'''