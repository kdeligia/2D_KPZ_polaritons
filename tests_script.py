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

params_init = {}
params_init['l0'] = [4 * 2 ** (1/2)]                                                                                         # μm
params_init['tau0'] = [params_init.get('l0')[0] ** 2]                                                                        # ps
params_init['N'] = [64]                                                                                                      # dimensionless!
params_init['dx'] = [0.5]                                                                                                    # dimensionless!
params_init['m'] = [8e-5]                                                                                                    # will multiply m_el in model_script.py
params_init['p'] = [2]                                                                                                       # dimensionless!
params_init['gamma0'] = [0.3125]                                                                                             # ps^-1
params_init['gamma2'] = [0.1]                                                                                                # μm^2 ps^-1
params_init['g'] = [0]                                                                                                       # μeV μm^2
params_init['gr'] = [0]                                                                                                      # μeV μm^2
params_init['ns'] = [3.75]                                                                                                   # μm^-2

dt = 5e-5
di = 5
N_input = 2.5e5
tf = N_input * dt * params_init.get('tau0')[0]
time = {}
time['dt'] = dt
time['di'] = di
time['N_input'] = N_input

t=[]
for i in range(0, int(N_input) + 1, di):
    ti = i * dt * params_init.get('tau0')[0]
    if ti >= 0 and i % di == 0:
        t.append(ti)

path = r'/Users/delis/Desktop' + os.sep + 'simulations vortices'
path_test = path + os.sep + 'dt' + str(dt) + 'dx' + str(params_init.get('dx')[0])

if os.path.isdir(path) == False:
    os.mkdir(path)
    if os.path.isdir(path_test) == False:
        os.mkdir(path_test)
else:
    if os.path.isdir(path_test) == False:
        os.mkdir(path_test)
    else:
        pass
    
x, y = ext.space_grid(params_init.get('N')[0], params_init.get('dx')[0])
np.savetxt(path_test + os.sep + 
           'N' + str(int(params_init.get('N')[0])) + '_' + 
           'dx' + str(params_init.get('dx')[0]) + '_' + 
           'unit' + str(params_init.get('l0')[0]) + '_' + 'xphys' + '.dat', x * params_init.get('l0')[0])
np.savetxt(path_test + os.sep + 
           'N' + str(int(params_init.get('N')[0])) + '_' + 
           'dx' + str(params_init.get('dx')[0]) + '_' + 
           'unit' + str(params_init.get('l0')[0]) + '_' + 'yphys' + '.dat', y * params_init.get('l0')[0])
np.savetxt(path_test + os.sep + 
           'Ninput' + str(int(time.get('N_input'))) + '_' + 
           'dt' + str(time.get('dt')) + '_' + 
           'unit' + str(params_init.get('tau0')[0]) + '_' + 'tphys' + '.dat', np.array(t))

keys = params_init.keys()
values = (params_init[key] for key in keys)
params = [dict(zip(keys, combination)) for combination in itertools.product(*values)]
for i in range(len(params)):
    params[i]['number'] = i + 1
qutip.settings.num_cpus = len(params)

def evolution(i_dict, savepath):
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
    g = current_dict.get('g')
    gr = current_dict.get('gr')
    ns = current_dict.get('ns')
    m = current_dict.get('m')
    name = 'm' + str(m) + '_' + 'p' + str(p) + '_' + 'gamma' + str(gamma0) + '_' + 'gammak' + str(gamma2) + '_' + 'g' + str(g) + '_' + 'gr' + str(gr) + '_'  + 'ns' + str(ns)
    gpe = model_script.gpe(**current_dict)
    theta_unwrapped, n = gpe.time_evolution_spacetime_vortices(**time)
    np.save(savepath + os.sep + name + '_' + 'theta_unwrapped' + '.npy', theta_unwrapped)
    np.save(savepath + os.sep + name + '_' + 'density' + '.npy', n)
    return None

parallel_map(evolution, range(len(params)), task_kwargs = dict(home = path_test))