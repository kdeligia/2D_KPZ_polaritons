#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 19 13:53:23 2021

@author: delis
"""

from qutip import *
import os
import numpy as np
import utils
import model_script
import itertools
import pprint


params_init = {}
params_init['factor'] = [1, 2, 4, 6, 8, 10]                                     # to be used for different discretizations!
params_init['l0'] = [4 * 2 ** (1/2)]                                            # μm
params_init['tau0'] = [int(np.round(params_init.get('l0')[0] ** 2))]            # ps
params_init['m'] = [8e-5]                                                       # will multiply the electron mass m_el in model_script module
params_init['p'] = [2]                                                          # dimensionless!
params_init['gamma0'] = [0.3125]                                                # ps^-1
params_init['gamma2'] = [0.1]                                                   # μm^2 ps^-1
params_init['g'] = [0]                                                          # μeV μm^2
params_init['gr'] = [0]                                                         # μeV μm^2
params_init['ns'] = [3.75]                                                      # μm^-2

dt = 5e-6
di = 200
N_input = 1.25e6
tf = N_input * dt * params_init.get('tau0')[0]
time = {}
time['dt'] = dt
time['di'] = di
time['N_input'] = N_input

t = []
flag = False
for i in range(0, int(N_input) + 1):
    ti = i * dt * params_init.get('tau0')[0]
    if int(ti) >= 100 and int(ti) <= 200:
        flag = True
    if flag is True and i % di == 0:
        t.append(ti)
print(np.array(t), np.array(t).shape)

keys = params_init.keys()
values = (params_init[key] for key in keys)
params = [dict(zip(keys, combination)) for combination in itertools.product(*values)]
for i in range(len(params)):
    params[i]['number'] = i + 1
qutip.settings.num_cpus = len(params)

root_path = r'/home6/konstantinos'
obj_path = r'convergence tests backup'

if os.path.isdir(root_path) is True and os.path.isdir(root_path + os.sep + obj_path) is False:
    os.mkdir(root_path + os.sep + obj_path)
else:
    pass


def evolution(i_dict, path=root_path+os.sep+obj_path):
    for i in range(len(params)):
        if params[i]['number'] == i_dict + 1:
            current_dict = params[i]
            break
    factor = current_dict.get('factor')
    N = 64 * factor
    dx = 0.5 / factor
    folder = path + os.sep + 'f' + str(factor)
    if os.path.isdir(folder) is False:
        os.mkdir(folder)
    else:
        pass
    gpe = model_script.gpe(N, dx, **current_dict)
    theta_unwrapped = gpe.time_evolution_spacetime_vortices(**time)
    full_name = utils.full_id(**current_dict)
    np.save(folder + os.sep + full_name + '_' + 'theta_unwrapped' + '.npy', theta_unwrapped)

    x, y = utils.space_grid(N, dx)
    np.savetxt(path + os.sep + 'N' + str(N) + '_' + 'dx' + str(dx) + '_' + 'unit' + str(current_dict.get('l0')) + '_' + 'xphys' + '.dat', x * current_dict.get('l0'))
    np.savetxt(path + os.sep + 'N' + str(N) + '_' + 'dx' + str(dx) + '_' + 'unit' + str(current_dict.get('l0')) + '_' + 'yphys' + '.dat', y * current_dict.get('l0'))
    np.savetxt(path + os.sep + 'Ninput' + str(int(time.get('N_input'))) + '_' + 'dt' + str(time.get('dt')) + '_' + 'unit' + str(current_dict.get('tau0')) + '_' + 'tphys' + '.dat', np.array(t))
    print('DONE!')
    print('Discretization')
    print('N = %.i, dx = %.3f' % (N, dx))
    print('----------------------')
    print('Simulations parameters')
    pprint.pprint(current_dict)
    print('----------------------')
    return None


parallel_map(evolution, range(len(params)))                                     # Parallel calculation: iterates through all values of the parameters