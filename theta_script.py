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

initial_path = r'/scratch/konstantinos' + os.sep + 'THETA_SIMULATIONS'
if os.path.isdir(r'/scratch/konstantinos') == True and os.path.isdir(initial_path) == False:
    os.mkdir(initial_path)
else:
    pass

parallel_tasks = 4
number_of_cores = 4
jobs_per_core = parallel_tasks // number_of_cores
qutip.settings.num_cpus = number_of_cores

iteration = 1
number_of_files = number_of_cores * iteration

params_init = {}
params_init['l0'] = [4 * 2 ** (1/2)]                                                                                         # μm
params_init['tau0'] = [params_init.get('l0')[0] ** 2]                                                                        # ps
params_init['N'] = [64]                                                                                                      # dimensionless!
params_init['dx'] = [4]                                                                                                      # dimensionless!
params_init['m'] = [8e-5]                                                                                                    # will multiply m_el in model_script.py
params_init['p'] = [2]                                                                                                       # dimensionless!
params_init['gamma0'] = [0.3125]                                                                                             # ps^-1
params_init['gamma2'] = [0.1]                                                                                                # μm^2 ps^-1
params_init['g'] = [0]                                                                                                       # μeV μm^-2
params_init['gr'] = [0]                                                                                                      # μeV μm^-2
params_init['ns'] = [3.75]                                                                                                   # μm^-2

dt = 5e-5                                                                                                                    # dimensionless!
di = 1                                                                                                                       # sample step
N_input = 100                                                                                                                # number of time steps
tf = N_input * dt * params_init.get('tau0')[0]
time = {}
time['dt'] = dt
time['di'] = di                                                                 # you might wanna implement a secondary sampling step for θ, but you need to unwrap in every time time step
time['N_input'] = N_input

def theta_data(i_batch, **args):
    mypath = args.get('misc_folder')
    for job in range(jobs_per_core):
        print('Running job = %.i at core = %.i' % (job, i_batch))
        gpe = model_script.gpe(**args)
        theta_unwound = gpe.time_evolution_theta(cutoff = 0.6, **time)
        np.savetxt(mypath + os.sep + 'trajectories_unwound'+ '_' + 'core' + str(i_batch + 1) + '_' + 'job' + str(job + 1) +'_' + 'iteration' + str(iteration) +'.dat', theta_unwound)
    return None

def call_avg(final_save_path, **args):
    keys = args.keys()
    values = (args[key] for key in keys)
    params = [dict(zip(keys, combination)) for combination in itertools.product(*values)]
    for parameters_current in params:
        p = parameters_current.get('p')
        gamma0 = parameters_current.get('gamma0')
        gamma2 = parameters_current.get('gamma2')
        g = parameters_current.get('g')
        gr = parameters_current.get('gr')
        ns = parameters_current.get('ns')
        m = parameters_current.get('m')
        
        name = 'm' + str(m) + '_' + 'p' + str(p) + '_' + 'gamma' + str(gamma0) + '_' + 'gammak' + str(gamma2) + '_' + 'g' + str(g) + '_' + 'gr' + str(gr) + '_'  + 'ns' + str(ns)
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

call_avg(r'/Users/delis/Desktop', **params_init)

# =============================================================================
# The times associated with sampling are the following:
# =============================================================================
'''
t=[]
for i in range(0, int(N_input) + 1, di):
    ti = i * dt * params_init.get('tau0')[0]
    if ti >= 0 and i % di == 0:
        t.append(ti)
'''