#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 15:09:50 2020

@author: delis
"""

from qutip import *
import os
import numpy as np
import utils
import itertools
import model_script


parallel_tasks = 4
number_of_cores = 4
jobs_per_core = parallel_tasks // number_of_cores
qutip.settings.num_cpus = number_of_cores
iteration = 1

params_init = {}
params_init['factor'] = [1]
params_init['l0'] = [4 * 2 ** (1/2)]                                                                                         # μm
params_init['tau0'] = [params_init.get('l0')[0] ** 2]                                                                        # ps                                                                                              # dimensionless!
params_init['m'] = [8e-5]                                                                                                    # will multiply m_el in model_script.py
params_init['p'] = [2]                                                                                                       # dimensionless!
params_init['gamma0'] = [0.3125]                                                                                             # ps^-1
params_init['gamma2'] = [0.1]                                                                                                # μm^2 ps^-1
params_init['g'] = [0]                                                                                                       # μeV μm^-2
params_init['gr'] = [0]                                                                                                      # μeV μm^-2
params_init['ns'] = [3.75]                                                                                                   # μm^-2

dt = 5e-5 / 16                                                                                                               # dimensionless!
di = 1                                                                                                                       # sample step
N_input = 100                                                                                                                # number of time steps
tf = N_input * dt * params_init.get('tau0')[0]
time = {}
time['dt'] = dt
time['di'] = di
time['N_input'] = N_input


def correlation(i_batch, **args):
    mypath = args.get('misc_folder')
    factor = args.get('factor')
    N = 64 * factor
    dx = 0.5 / factor
    psi_correlation = []
    n_correlation = []
    n_avg = []
    exponential_avg = []
    for job in range(jobs_per_core):
        print('Running job = %.i at core = %.i' % (job, i_batch))
        gpe = model_script.gpe(N, dx, **args)
        psi_correlation_run, n_correlation_run, n_avg_run, exponential_avg_run = gpe.time_evolution_psi(**time)

        psi_correlation.append(psi_correlation_run / jobs_per_core)
        n_correlation.append(n_correlation_run / jobs_per_core)
        n_avg.append(n_avg_run / jobs_per_core)
        exponential_avg.append(exponential_avg_run / jobs_per_core)

    np.savetxt(mypath + os.sep + 'psi_correlation' + '_' + 'core' + str(i_batch + 1) + '_' + 'iteration' + str(iteration) + '.npy', np.sum(psi_correlation))
    np.savetxt(mypath + os.sep + 'n_correlation' + '_' + 'core' + str(i_batch + 1) + '_' + 'iteration' + str(iteration) + '.npy', np.sum(n_correlation))
    np.savetxt(mypath + os.sep + 'n_avg' + '_' + 'core' + str(i_batch + 1) + '_' + 'iteration' + str(iteration) + '.npy', np.sum(n_avg))
    np.savetxt(mypath + os.sep + 'exponential_avg' + '_' + 'core' + str(i_batch + 1) + '_' + 'iteration' + str(iteration) + '.npy', np.sum(exponential_avg))
    return None


root_path = r'/Users/konstantinosdeligiannis/Desktop'
obj_path = r'g1 simulations'
if os.path.isdir(root_path) is True and os.path.isdir(root_path + os.sep + obj_path) is False:
    os.mkdir(root_path + os.sep + obj_path)
else:
    pass


def average_g1(full_name, folder, save_path):
    psi_correlation = []
    n_correlation = []
    n_avg = []
    exponential_avg = []
    for file in os.listdir(folder):
        if 'psi_correlation' in file:
            psi_correlation.append(np.loadtxt(folder + os.sep + file))
        elif 'n_correlation' in file:
            n_correlation.append(np.loadtxt(folder + os.sep + file))
        elif 'n_avg' in file:
            n_avg.append(np.loadtxt(folder + os.sep + file))
        elif 'exponential_avg' in file:
            exponential_avg.append(np.loadtxt(folder + os.sep + file))
    np.savetxt(save_path + os.sep + full_name + '_' + 'g1psi' + '.npy', np.real(np.abs(np.mean(psi_correlation, axis=0)) / np.sqrt(np.mean(n_avg, axis=0)[0, 0] * np.mean(n_avg, axis=0))))
    np.savetxt(save_path + os.sep + full_name + '_' + 'g1n' + '.npy', np.real(np.abs(np.mean(n_correlation, axis=0)) / np.sqrt(np.mean(n_avg, axis=0)[0, 0] * np.mean(n_avg, axis=0))))
    np.savetxt(save_path + os.sep + full_name + '_' + 'g1theta' + '.npy', np.real(np.abs(np.mean(exponential_avg, axis=0))))


def call_avg(**args):
    keys = args.keys()
    values = (args[key] for key in keys)
    params = [dict(zip(keys, combination)) for combination in itertools.product(*values)]
    for parameters_current in params:
        full_name = utils.full_id(**parameters_current)
        folder = root_path + os.sep + obj_path + os.sep + full_name
        if os.path.isdir(folder) is False:
            os.mkdir(folder)
        parameters_current['misc_folder'] = folder
        parallel_map(correlation, range(number_of_cores), task_kwargs=parameters_current)
        average_g1(full_name, folder, root_path)
    return None


#call_avg(**params_init)


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