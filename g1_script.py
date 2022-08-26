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

root_path = r'/scratch/konstantinos'
obj_path = r'g1 simulations'
observable_path = os.path.join(root_path, obj_path)
utils.mkstring(observable_path)

total_tasks = 1280
number_of_cores = 64
jobs_per_core = total_tasks // number_of_cores
qutip.settings.num_cpus = number_of_cores
run = 1

params_init = {}
params_init['factor'] = [1]                                                     # dimensionless factor, controls discretization
params_init['l0'] = [4 * 2 ** (1/2)]                                            # μm
params_init['tau0'] = [int(params_init.get('l0')[0] ** 2)]                      # ps                                                                                              # dimensionless!
params_init['m'] = [8e-5]                                                       # will multiply electron mass
params_init['p'] = [2]                                                          # dimensionless P/Pth
params_init['gamma0'] = [0.3125]                                                # ps^-1
params_init['gamma2'] = [0.1]                                                   # μm^2 ps^-1
params_init['g'] = [0]                                                          # μeV μm^-2
params_init['gr'] = [0]                                                         # μeV μm^-2
params_init['ns'] = [3.75]                                                      # μm^-2

dt = 1e-4                                                                       # dimensionless
di = 500                                                                         # sample step
Nsteps = int(3e6)                                                               # number of time steps

time = {}
time['dt'] = dt
time['di'] = di
time['Nsteps'] = Nsteps

tsteady = 200
tsample = np.linspace(0, Nsteps * dt, Nsteps // di + 1) * params_init.get('tau0')[0]
tss = tsample[np.where(tsample >= tsteady)[0][0]]
time['tss'] = tss
np.savetxt(root_path + os.sep + 'refreport_t.dat', tsample[tsample>=tss])

'''
check = []
for i in range(Nsteps+1):
    t = i * dt * params_init.get('tau0')[0]
    if t >= tss and i % di == 0:
        check.append(t)
print(np.array(check))
'''


def extract_correlation(core, **args):
    save_path = args.get('current_folder')
    factor = args.get('factor')
    N = 64 * factor
    dx = 0.5 / factor

    subfolder_psi_correlation = os.path.join(save_path, 'psi_correlation', "")
    subfolder_theta_correlation = os.path.join(save_path, 'theta_correlation', "")
    subfolder_n_correlation = os.path.join(save_path, 'n_correlation', "")
    subfolder_n_avg = os.path.join(save_path, 'n_avg', "")

    run_core_identifier = os.path.join('run', str(run), '_', 'core', str(core + 1), '_').replace('/', '')
    utils.mkstring(subfolder_psi_correlation)
    utils.mkstring(subfolder_theta_correlation)
    utils.mkstring(subfolder_n_correlation)
    utils.mkstring(subfolder_n_avg)
    for job in range(jobs_per_core):
        job_identifier = os.path.join('job', str(job + 1)).replace('/', '')
        gpe = model_script.gpe(N, dx, **args)
        psi_correlation_run, n_correlation_run, n_avg_run, exponential_avg_run = gpe.time_evolution_psi(**time)

        np.save(subfolder_psi_correlation + run_core_identifier + job_identifier + '.npy', psi_correlation_run)
        np.save(subfolder_n_correlation + run_core_identifier + job_identifier + '.npy', n_correlation_run)
        np.save(subfolder_n_avg + run_core_identifier + job_identifier + '.npy', n_avg_run)
        np.save(subfolder_theta_correlation + run_core_identifier + job_identifier + '.npy', exponential_avg_run)
    return subfolder_psi_correlation, subfolder_theta_correlation, subfolder_n_avg, subfolder_n_correlation


def call_avg(**args):
    keys = args.keys()
    values = (args[key] for key in keys)
    params = [dict(zip(keys, combination)) for combination in itertools.product(*values)]
    for parameters_current in params:
        full_name = utils.full_id(**parameters_current)
        current_folder = os.path.join(observable_path, full_name)
        parameters_current['current_folder'] = current_folder
        utils.mkstring(current_folder)
        paths = parallel_map(extract_correlation, range(number_of_cores), task_kwargs=parameters_current)[0]
        for path in paths:
            tosave = path.split("/")[-2]
            result = utils.ensemble_average(path)
            np.save(observable_path + os.sep + 'AVG_' + tosave + '.npy', result)        # Watch out, np.save because of .npy files. See also the comment in utils.
    return None


call_avg(**params_init)

'''
keys = params_init.keys()
values = (params_init[key] for key in keys)
params = [dict(zip(keys, combination)) for combination in itertools.product(*values)]
for parameters_current in params:
    gpe = model_script.gpe(64, 0.5, **parameters_current)
'''


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