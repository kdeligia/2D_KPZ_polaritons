#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 15 16:19:12 2021

@author: delis
"""

from qutip import *
import os
import numpy as np
import utils
import model_script
import itertools


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


def theta_data(i_batch, **args):
    mypath = args.get('misc_folder')
    factor = args.get('factor')
    N = 64 * factor
    dx = 0.5 / factor
    for job in range(jobs_per_core):
        print('Running job = %.i at core = %.i' % (job, i_batch))
        gpe = model_script.gpe(N, dx, **args)
        theta_unwound = gpe.time_evolution_theta(cutoff=0.5, **time)
        np.savetxt(mypath + os.sep + 'trajectories_unwound' + '_' + 'core' + str(i_batch + 1) + '_' + 'job' + str(job + 1) + '_' + 'iteration' + str(iteration) + '.dat', theta_unwound)
    return None


root_path = r'/Users/konstantinosdeligiannis/Desktop'
obj_path = r'theta simulations'
if os.path.isdir(root_path) is True and os.path.isdir(root_path + os.sep + obj_path) is False:
    os.mkdir(root_path + os.sep + obj_path)
else:
    pass


def call_avg(final_save_path, **args):
    keys = args.keys()
    values = (args[key] for key in keys)
    params = [dict(zip(keys, combination)) for combination in itertools.product(*values)]
    for parameters_current in params:
        full_name = utils.full_id(**parameters_current)
        folder = root_path + os.sep + obj_path + os.sep + full_name
        if os.path.isdir(folder) is False:
            os.mkdir(folder)
        parameters_current['misc_folder'] = folder
        parallel_map(theta_data, range(number_of_cores), task_kwargs=parameters_current, progress_bar=True)

        unwound_trajectories = []
        for file in os.listdir(folder):
            if 'trajectories_unwound' in file:
                unwound_trajectories.append(np.loadtxt(folder + os.sep + file))
        np.savetxt(final_save_path + os.sep + full_name + '_' + 'trajectories' + '.dat', np.concatenate(unwound_trajectories, axis=0))
        return None


call_avg(root_path, **params_init)

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