#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 15:09:50 2020

@author: delis
"""

from qutip import *
import os
import numpy as np
import external as ext
import model_script
import itertools

initial_path = r'/Users/delis/Desktop' + os.sep + 'g1_SIMULATIONS'
if os.path.isdir(initial_path) == False:
    os.mkdir(initial_path)

parallel_tasks = 1
number_of_cores = 1
jobs_per_core = parallel_tasks // number_of_cores
qutip.settings.num_cpus = number_of_cores
iteration = 1
number_of_files = number_of_cores * iteration

params_init = {}
params_init['N'] = [2 ** 6]
params_init['dx'] = [0.5]
params_init['p'] = [1.9, 2]
params_init['sigma'] = [7.5]
params_init['gamma0'] = [0.3125]
params_init['gamma2'] = [0.7, 0.8]
params_init['g'] = [0]
params_init['gr'] = [0]
params_init['ns'] = [15, 20]
params_init['m'] = [1e-4]

time_dict = {}
time_dict['dt'] = 0.005
time_dict['i_start'] = 1
time_dict['di'] = 1
time_dict['N_input'] = 1
t = ext.time(time_dict.get('dt'), time_dict.get('N_input'), time_dict.get('i_start'), time_dict.get('di'))
#np.savetxt(r'/home6/konstantinos' + os.sep + 'Deltat_g1.dat', t-t[0])

def g1_data(i_batch, **args):
    mypath = args.get('misc_folder')
    N_input = time_dict.get('N_input')
    N = args.get('N')
    di = time_dict.get('di')
    print(type(N), type(N_input), type(di))
    psipsi_full_batch = np.zeros((N_input//di + 1, N//2), dtype = complex)
    n_avg_batch = np.zeros((N_input//di + 1, N//2), dtype = complex)
    for job in range(jobs_per_core):
        gpe = model_script.gpe(**args)
        psipsi_full, n_avg = gpe.time_evolution_psi(**time_dict)
        psipsi_full_batch += psipsi_full / jobs_per_core
        n_avg_batch += n_avg / jobs_per_core
    np.save(mypath + os.sep + 'psipsi_full' + '_' + 'core' + str(i_batch + 1) + '_' + 'iteration' + str(iteration) + '.npy', psipsi_full_batch)
    np.save(mypath + os.sep + 'n_avg' + '_' + 'core' + str(i_batch + 1) + '_' + 'iteration' + str(iteration) + '.npy', n_avg_batch)
    return None

def call_avg(final_save_path, **args):
    keys = args.keys()
    values = (args[key] for key in keys)
    params = [dict(zip(keys, combination)) for combination in itertools.product(*values)]
    for parameters_current in params:
        N = parameters_current.get('N')
        dx = parameters_current.get('dx')
        p = parameters_current.get('p')
        sigma = parameters_current.get('sigma')
        gamma0 = parameters_current.get('gamma0')
        gamma2 = parameters_current.get('gamma2')
        g = parameters_current.get('g')
        gr = parameters_current.get('gr')
        ns = parameters_current.get('ns')
        m = parameters_current.get('m')
        print('--- Grid: N = %.i, dx = %.1f' % (N, dx))
        print('--- Main: p = %.1f, sigma = %.2f, g = %.2f, gr = %.2f, ns = %.i, m = %.4f' % (p, sigma, g, gr, ns, m))
        print('--- Loss rates: gamma0 = %.4f, gamma2 = %.1f' % (gamma0, gamma2))

        name = 'N' + str(N) + '_' + 'p' + str(p) + '_' + 'sigma' + str(sigma) + '_' + 'gamma' + str(gamma0) + '_' + 'gammak' + str(gamma2) + '_' + 'g' + str(g) + '_' + 'ns' + str(ns) + '_' + 'm' + str(m) 
        misc_folder = initial_path + os.sep + name
        if os.path.isdir(misc_folder) == False:
            os.mkdir(misc_folder)
        parameters_current['simul_id'] = name
        parameters_current['misc_folder'] = misc_folder

        N_input = time_dict.get('N_input')
        di = time_dict.get('di')
        parallel_map(g1_data, range(number_of_cores), task_kwargs = parameters_current, progress_bar=True)
        psipsi_full = np.zeros((N_input//di + 1, N//2), dtype = complex)
        n_avg = np.zeros((N_input//di + 1, N//2), dtype = complex)
        for file in os.listdir(misc_folder):
            if 'psipsi_full' in file:
                psipsi_full += np.load(misc_folder + os.sep + file) / number_of_files
            elif 'n_avg' in file:
                n_avg += np.load(misc_folder + os.sep + file) / number_of_files
        np.save(final_save_path + os.sep + name + '_' + 'full_g1' + '.npy', np.real(np.abs(psipsi_full) / np.sqrt(n_avg[0, 0] * n_avg)))
        return None

call_avg(r'/Users/delis', **params_init)