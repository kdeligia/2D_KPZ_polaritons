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

initial_path = r'/Users/delis/Desktop' + os.sep + 'THETA_SIMULATIONS'
if os.path.isdir(r'/Users/delis/Desktop') == True and os.path.isdir(initial_path) == False:
    os.mkdir(initial_path)
else:
    pass

#parallel_tasks = 10240
parallel_tasks = 1
#number_of_cores = 128
number_of_cores = 1
jobs_per_core = parallel_tasks // number_of_cores
qutip.settings.num_cpus = number_of_cores
iteration = 1

params_init = {}
params_init['N'] = [2 ** 6]
params_init['dx'] = [0.5]
params_init['p'] = [2]
params_init['sigma'] = [7.5]
params_init['gamma0'] = [0.3125]
params_init['gamma2'] = [0.1]
params_init['g'] = [0]
params_init['gr'] = [0]
params_init['ns'] = [120]
params_init['m'] = [8e-5]

dt = 5e-6
di = 8000
N_input = 40e6
time = {}
time['dt'] = dt
time['N_input'] = N_input
time['di'] = di
t = [i * dt for i in range(0, int(N_input) + 1, di)]
print(t)
#np.savetxt('/Users/delis/Desktop/t_test.dat', t)

def theta_data(i_batch, **args):
    mypath = args.get('misc_folder')
    for job in range(jobs_per_core):
        gpe = model_script.gpe(**args)
        theta_unwound = gpe.time_evolution_theta(cutoff = 0.8, **time)
        np.savetxt(mypath + os.sep + 'trajectories_unwound'+ '_' + 'core' + str(i_batch + 1) + '_' + 'job' + str(job + 1) +'_' + 'iteration' + str(iteration) +'.dat', theta_unwound)
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
        print('--- Theta Simulations ---')
        print('--- Grid: N = %.i, dx = %.1f' % (N, dx))
        print('--- Main: p = %.3f, sigma = %.3f, g = %.3f, gr = %.3f, ns = %.i, m = %.e' % (p, sigma, g, gr, ns, m))
        print('--- Loss rates: gamma0 = %.4f, gamma2 = %.4f' % (gamma0, gamma2))

        name = 'N' + str(N) + '_' + 'p' + str(p) + '_' + 'sigma' + str(sigma) + '_' + 'gamma' + str(gamma0) + '_' + 'gammak' + str(gamma2) + '_' + 'g' + str(g) + '_' + 'ns' + str(ns) + '_' + 'm' + str(m) 
        misc_folder = initial_path + os.sep + name
        if os.path.isdir(initial_path) == True and os.path.isdir(misc_folder) == False:
            os.mkdir(misc_folder)
        parameters_current['misc_folder'] = misc_folder
        parallel_map(theta_data, range(number_of_cores), task_kwargs = parameters_current, progress_bar=True)

        unwound_trajectories = []
        if os.path.isdir(initial_path) == True:
            for file in os.listdir(misc_folder):
                if 'trajectories_unwound' in file:
                    unwound_trajectories.append(np.loadtxt(misc_folder + os.sep + file))
            np.savetxt(final_save_path + os.sep + name + '_' + 'trajectories' + '.dat', np.concatenate(unwound_trajectories, axis = 0))
        return None

#call_avg(r'/Users/delis/Desktop', **params_init)