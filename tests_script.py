#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 19 13:53:23 2021

@author: delis
"""

import matplotlib.pyplot as pl
from qutip import *
import os
import numpy as np
import external as ext
import model_script
import itertools

final_save_path = r'/home6/konstantinos'
initial_path = r'/home6/konstantinos' + os.sep + 'TEST_SIMULATIONS'
if os.path.isdir(initial_path) == False:
    os.mkdir(initial_path)

params_init = {}
params_init['l0'] = [4 * np.sqrt(2)]                                            #μm
params_init['tau0'] = [32]                                                      #ps
params_init['N'] = [640]                                                        # !dimensionless!
params_init['dx'] = [0.05]                                                      # !dimensionless!
params_init['p'] = [2]                                                          # !dimensionless!
params_init['gamma0'] = [0.3125]                                                #ps^-1
params_init['gamma2'] = [0.1]                                                   #μm^2 ps^-1
params_init['g'] = [0]                                                          # μeV μm^-2
params_init['gr'] = [0]                                                         # μeV μm^-2
params_init['ns'] = [3.75]                                                      # μm^-2
params_init['m'] = [8e-5]                                                       #times m_el in model_script.py

dt = 5e-5                                                                       # !dimensionless!
di = 1                                                                          # sample step
N_input = 6.25e3                                                                # time steps
time = {}
time['dt'] = dt
time['di'] = di
time['N_input'] = N_input

keys = params_init.keys()
values = (params_init[key] for key in keys)
params = [dict(zip(keys, combination)) for combination in itertools.product(*values)]
for i in range(len(params)):
    params[i]['number'] = i + 1
qutip.settings.num_cpus = len(params)

def evolution_vortices(i_dict, home):
    for i in range(len(params)):
        if params[i]['number'] == i_dict + 1:
            current_dict = params[i]
            break
    N = current_dict.get('N')
    p = current_dict.get('p')
    gamma0 = current_dict.get('gamma0')
    gamma2 = current_dict.get('gamma2')
    sigma = params_init['tau0'][0] * gamma0 * (p + 1) / 4
    g = current_dict.get('g')
    ns = current_dict.get('ns')
    m = current_dict.get('m')
    current_dict['sigma'] = sigma
    
    name = 'N' + str(N) + '_' + 'p' + str(p) + '_' + 'sigma' + str(sigma) + '_' + 'gamma' + str(gamma0) + '_' + 'gammak' + str(gamma2) + '_' + 'g' + str(g) + '_' + 'ns' + str(ns) + '_' + 'm' + str(m) 
    misc_folder = initial_path + os.sep + name
    if os.path.isdir(misc_folder) == False:
        os.mkdir(misc_folder)
    current_dict['misc_folder'] = misc_folder
    gpe = model_script.gpe(**current_dict)
    t, theta_unwrapped, theta_wrapped = gpe.time_evolution_spacetime_vortices(1/2, misc_folder, **time)
    np.save(initial_path + os.sep + name + '_' + 'theta_unwrapped' + '.npy', theta_unwrapped)
    np.save(initial_path + os.sep + name + '_' + 'theta_wrapped' + '.npy', theta_wrapped)
    np.save(initial_path + os.sep + name + '_' + 't_test' + '.npy', t)
    return None

#parallel_map(evolution, range(len(params)), task_kwargs = dict(home = final_save_path))
evolution_vortices(0, home = final_save_path)

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