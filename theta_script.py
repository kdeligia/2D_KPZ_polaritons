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
import itertools
import model_script

root_path = r'/Users/konstantinosdeligiannis/Desktop'
obj_path = r'theta simulations'
observable_path = os.path.join(root_path, obj_path)
utils.mkstring(observable_path)

total_tasks = 10
number_of_cores = 5
jobs_per_core = total_tasks // number_of_cores
qutip.settings.num_cpus = number_of_cores
run = 1

params_init = {}
params_init['factor'] = [1]                                                     # dimensionless factor, controls discretization
params_init['l0'] = [4 * 2 ** (1/2)]                                            # μm
params_init['tau0'] = [params_init.get('l0')[0] ** 2]                           # ps                                                                                              # dimensionless!
params_init['m'] = [8e-5]                                                       # will multiply electron mass
params_init['p'] = [2]                                                          # dimensionless P/Pth
params_init['gamma0'] = [0.3125]                                                # ps^-1
params_init['gamma2'] = [0.1]                                                   # μm^2 ps^-1
params_init['g'] = [0]                                                          # μeV μm^-2
params_init['gr'] = [0]                                                         # μeV μm^-2
params_init['ns'] = [3.75]                                                      # μm^-2

dt = 1                                                                          # dimensionless
di = 1                                                                          # sample step
N_input = 1                                                                     # number of time steps
tf = N_input * dt * params_init.get('tau0')[0]
time = {}
time['dt'] = dt
time['di'] = di
time['N_input'] = N_input


def extract_theta(core, **args):
    save_path = args.get('current_folder')
    factor = args.get('factor')
    N = 64 * factor
    dx = 0.5 / factor
    subfolder_theta = os.path.join(save_path, 'pure theta', "")
    run_core_identifier = os.path.join('run', str(run), '_', 'core', str(core + 1), '_').replace('/', '')
    utils.mkstring(subfolder_theta)
    for job in range(jobs_per_core):
        job_identifier = os.path.join('job', str(job + 1)).replace('/', '')
        gpe = model_script.gpe(N, dx, **args)
        theta_unwound_run = gpe.time_evolution_theta(cutoff=0.5, **time)
        np.save(subfolder_theta + run_core_identifier + job_identifier + '.npy', theta_unwound_run)
    return subfolder_theta


def call_avg(**args):
    keys = args.keys()
    values = (args[key] for key in keys)
    params = [dict(zip(keys, combination)) for combination in itertools.product(*values)]
    for parameters_current in params:
        full_name = utils.full_id(**parameters_current)
        current_folder = os.path.join(observable_path, full_name)
        parameters_current['current_folder'] = current_folder
        utils.mkstring(current_folder)
        path = parallel_map(extract_theta, range(number_of_cores), task_kwargs=parameters_current)[0]

        tosave = path.split("/")[-2]
        result = utils.append_theta_trajectories(path)
        np.savetxt(observable_path + os.sep + 'trajectories_' + tosave + '.dat', result)
    return None


call_avg(**params_init)

# =============================================================================
# The times associated with sampling are the following:
# =============================================================================
'''
t=[]
for i in range(int(N_input) + 1, di):
    ti = i * dt * params_init.get('tau0')[0]
    if ti >= 0 and i % di == 0:
        t.append(ti)
'''