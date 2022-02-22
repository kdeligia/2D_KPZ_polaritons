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

initial_path = r'/scratch/konstantinos' + os.sep + 'g1 simulations'
if os.path.isdir(r'/scratch/konstantinos') == True and os.path.isdir(initial_path) == False:
    os.mkdir(initial_path)
else:
    pass

parallel_tasks = 4092
number_of_cores = 128
jobs_per_core = parallel_tasks // number_of_cores
qutip.settings.num_cpus = number_of_cores

iteration = 1
number_of_files = number_of_cores * iteration

params_init = {}
params_init['l0'] = [4 * 2 ** (1/2)]                                                                                         # μm
params_init['tau0'] = [params_init.get('l0')[0] ** 2]                                                                        # ps
params_init['N'] = [4 * 64]                                                                                                  # dimensionless!
params_init['dx'] = [0.5 / 4]                                                                                                # dimensionless!
params_init['m'] = [8e-5]                                                                                                    # will multiply m_el in model_script.py
params_init['p'] = [2]                                                                                                       # dimensionless!
params_init['gamma0'] = [0.3125]                                                                                             # ps^-1
params_init['gamma2'] = [0.1]                                                                                                # μm^2 ps^-1
params_init['g'] = [0]                                                                                                       # μeV μm^-2
params_init['gr'] = [0]                                                                                                      # μeV μm^-2
params_init['ns'] = [3.75]                                                                                                   # μm^-2

dt = 5e-5 / 16                                                                                                               # dimensionless!
di = 1                                                                                                                       # sample step
N_input = 1e6                                                                                                                # number of time steps
tf = N_input * dt * params_init.get('tau0')[0]
time = {}
time['dt'] = dt
time['di'] = di
time['N_input'] = N_input

def correlation(i_batch, **args):
    mypath = args.get('misc_folder')
    for job in range(jobs_per_core):
        print('Running job = %.i at core = %.i' % (job, i_batch))
        gpe = model_script.gpe(**args)
        psi_correlation, n_correlation, n_avg, exponential_avg = gpe.time_evolution_psi(**time)
        psi_correlation /= jobs_per_core
        n_correlation /= jobs_per_core
        n_avg /= jobs_per_core
        exponential_avg /= jobs_per_core
        if job == 0:
            np.save(mypath + os.sep + 'psi_correlation' + '_' + 'core' + str(i_batch + 1) + '_' + 'iteration' + str(iteration) + '.npy', psi_correlation)
            np.save(mypath + os.sep + 'n_correlation' + '_' + 'core' + str(i_batch + 1) + '_' + 'iteration' + str(iteration) + '.npy', n_correlation)
            np.save(mypath + os.sep + 'n_avg' + '_' + 'core' + str(i_batch + 1) + '_' + 'iteration' + str(iteration) + '.npy', n_avg)
            np.save(mypath + os.sep + 'exponential_avg' + '_' + 'core' + str(i_batch + 1) + '_' + 'iteration' + str(iteration) + '.npy', exponential_avg)
        elif job > 0:
            np.save(mypath + os.sep + 'psi_correlation' + '_' + 'core' + str(i_batch + 1) + '_' + 'iteration' + str(iteration) + '.npy', 
                    psi_correlation + np.load(mypath + os.sep + 'psi_correlation' + '_' + 'core' + str(i_batch + 1) + '_' + 'iteration' + str(iteration) + '.npy'))
            np.save(mypath + os.sep + 'n_correlation' + '_' + 'core' + str(i_batch + 1) + '_' + 'iteration' + str(iteration) + '.npy', 
                    n_correlation + np.load(mypath + os.sep + 'n_correlation_INTERM' + '_' + 'core' + str(i_batch + 1) + '_' + 'iteration' + str(iteration) + '.npy'))
            np.save(mypath + os.sep + 'n_avg' + '_' + 'core' + str(i_batch + 1) + '_' + 'iteration' + str(iteration) + '.npy', 
                    n_avg + np.load(mypath + os.sep + 'n_avg' + '_' + 'core' + str(i_batch + 1) + '_' + 'iteration' + str(iteration) + '.npy'))
            np.save(mypath + os.sep + 'exponential_avg' + '_' + 'core' + str(i_batch + 1) + '_' + 'iteration' + str(iteration) + '.npy', 
                    exponential_avg + np.load(mypath + os.sep + 'exponential_avg_INTERM' + '_' + 'core' + str(i_batch + 1) + '_' + 'iteration' + str(iteration) + '.npy'))
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
        parallel_map(correlation, range(number_of_cores), task_kwargs = parameters_current, progress_bar = True)
        psi_correlation = []
        n_correlation = []
        n_avg = []
        exponential_avg = []
        if os.path.isdir(initial_path) == True:
            for file in os.listdir(misc_folder):
                if 'psi_correlation' in file:
                    psi_correlation.append(np.load(misc_folder + os.sep + file))
                elif 'n_correlation' in file:
                    n_correlation.append(np.load(misc_folder + os.sep + file))
                elif 'n_avg' in file:
                    n_avg.append(np.load(misc_folder + os.sep + file))
                elif 'exponential_avg' in file:
                    exponential_avg.append(np.load(misc_folder + os.sep + file))
        np.save(final_save_path + os.sep + name + '_' + 'g1psi' + '.npy', np.real(np.abs(np.mean(psi_correlation, axis=0)) / np.sqrt(np.mean(n_avg, axis=0)[0, 0] * np.mean(n_avg, axis=0))))
        np.save(final_save_path + os.sep + name + '_' + 'g1n' + '.npy', np.real(np.abs(np.mean(n_correlation, axis=0)) / np.sqrt(np.mean(n_avg, axis=0)[0, 0] * np.mean(n_avg, axis=0))))
        np.save(final_save_path + os.sep + name + '_' + 'g1theta' + '.npy', np.real(np.abs(np.mean(exponential_avg, axis=0))))

#call_avg(r'/home6/konstantinos', **params_init)