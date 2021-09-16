#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 15 16:19:12 2021

@author: delis
"""

from qutip import *
import os
import numpy as np
import external as ext
import model_script
import itertools

path = r'/scratch/konstantinos'
final_save_path = r'/home6/konstantinos'

parallel_tasks = 512
cores = 128
per_core = parallel_tasks // cores
qutip.settings.num_cpus = cores
iteration = 0

params_init = {}
params_init['N'] = [2 ** 7]
params_init['dx'] = [0.5]
params_init['p'] = [2]
params_init['sigma'] = [0.02]
params_init['gamma0'] = [0.3125]
params_init['gamma2'] = [0.1]
params_init['g'] = [0]
params_init['gr'] = [0]
params_init['ns'] = [1]
params_init['m'] = [1e-4]

time_dict = {}
time_dict['dt'] = 0.001
time_dict['i_start'] = 0
time_dict['di'] = 500
time_dict['N_input'] = 6000000
t = ext.time(time_dict.get('dt'), time_dict.get('N_input'), time_dict.get('i_start'), time_dict.get('di'))
np.savetxt(final_save_path + os.sep + 't_theta_128.dat', t)

def theta_data(i_batch, **args):
    mypath = args.get('save_folder')
    for i_n in range(per_core):
        gpe = model_script.gpe(**args)
        theta_unwound = gpe.time_evolution_theta(**time_dict)
        np.savetxt(mypath + os.sep + 'trajectories_unwound' + '_' + 'core' + str(i_batch + 1) + '_' + str(iteration * per_core + i_n + 1) + '.dat', theta_unwound)
        if (i_n + 1) % 2 == 0:
            print('Core %.i finished realisation %.i \n' % (i_batch, i_n + 1))
    return None

def call_avg(final_save_path, **args):
    keys = args.keys()
    values = (args[key] for key in keys)
    params = [dict(zip(keys, combination)) for combination in itertools.product(*values)]
    for parameters in params:
        N = parameters.get('N')
        dx = parameters.get('dx')
        p = parameters.get('p')
        sigma = parameters.get('sigma')
        gamma0 = parameters.get('gamma0')
        gamma2 = parameters.get('gamma2')
        g = parameters.get('g')
        gr = parameters.get('gr')
        ns = parameters.get('ns')
        m = parameters.get('m')
        print('--- Grid: N = %.i, dx = %.1f' % (N, dx))
        print('--- Main: p = %.1f, sigma = %.2f, g = %.2f, gr = %.2f, ns = %.i, m = %.4f' % (p, sigma, g, gr, ns, m))
        print('--- Loss rates: gamma0 = %.4f, gamma2 = %.1f' % (gamma0, gamma2))
        id_string = 'N' + str(N) + '_' + 'p' + str(p) + '_' + 'sigma' + str(sigma) + '_' + 'gamma' + str(gamma0) + '_' + 'gammak' + str(gamma2) + '_' + 'g' + str(g)
        init = path + os.sep + 'THETA_SIMULATIONS' + '_' + 'N' + str(N) + '_' + 'ns' + str(int(ns)) + '_' + 'm' + str(m)
        save_folder = init + os.sep + id_string
        if os.path.isdir(init) == False:
            os.mkdir(init)
        if os.path.isdir(save_folder) == False:
            os.mkdir(save_folder)
        unwound_trajectories = []
        parameters['save_folder'] = save_folder
        parallel_map(theta_data, range(cores), task_kwargs = parameters, progress_bar=True)
        for file in os.listdir(save_folder):
            if 'trajectories_unwound' in file:
                unwound_trajectories.append(np.loadtxt(save_folder + os.sep + file))
        np.savetxt(final_save_path + os.sep + id_string + '_' + 'trajectories' + '.dat', np.concatenate(unwound_trajectories, axis = 0))
        return None

call_avg(final_save_path, **params_init)