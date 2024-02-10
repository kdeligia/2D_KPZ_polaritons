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

root_path = r'/Users/konstantinosdeligiannis/Desktop'
obj_path = r'tests_scanning'
observable_path = os.path.join(root_path, obj_path)
utils.mksubfolder(observable_path)

number_of_cores = 1
qutip.settings.num_cpus = number_of_cores

params_init: dict[str, list[float | int]]= {}
params_init['factor'] = [1]                                                     # to be used for different discretizations!
params_init['l0'] = [4 * 2 ** (1/2)]                                            # μm
params_init['tau0'] = [int(np.round(params_init.get('l0')[0] ** 2))]            # ps
params_init['m'] = [8e-5]                                                       # will multiply the electron mass m_el in model_script module
params_init['p'] = [2]                                                          # dimensionless!
params_init['gamma0'] = [0.3125]                                                # ps^-1
params_init['gamma2'] = [0.1]                                                   # μm^2 ps^-1
params_init['g'] = [1, 4, 8]                                                    # μeV μm^2
params_init['gr'] = [0, 0.5]                                                    # μeV μm^2
params_init['ns'] = [3.75]                                                      # μm^-2

dt = 5e-6
di = 1
N_input = 10
tmin = 1
tmax = 2
tf = N_input * dt * params_init.get('tau0')[0]
time = {}
time['tmin'] = tmin
time['tmax'] = tmax
time['dt'] = dt
time['di'] = di
time['Nsteps'] = N_input


t = []
flag = False
for i in range(int(N_input) + 1):
    ti = i * dt * params_init.get('tau0')[0]
    if int(ti) >= tmin and int(ti) <= tmax:
        flag = True
    if flag is True and i % di == 0:
        t.append(ti)
ta = np.array(t)

keys = params_init.keys()
values = (params_init[key] for key in keys)
params = [dict(zip(keys, combination)) for combination in itertools.product(*values)]


def evolution(i_dict):
    parameters_current = np.take(params, i_dict)
    full_name = utils.full_id(**parameters_current)
    current_folder = os.path.join(observable_path, full_name)

    factor = parameters_current.get('factor')
    N = 64 * factor
    dx = 0.5 / factor
    current_folder += '_' + 'N' + str(N) + '_' + 'dx' + str(dx)
    parameters_current['current_folder'] = current_folder
    utils.mksubfolder(current_folder)

    gpe = model_script.gpe(N, dx, **parameters_current)
    theta, density = gpe.time_evolution_spacetime_vortices(**time)
    np.save(current_folder + os.sep + 'theta' + '.npy', theta)
    np.save(current_folder + os.sep + 'density' + '.npy', density)
    return None

parallel_map(evolution, range(len(params)))
