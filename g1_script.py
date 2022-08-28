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
obj_path = r'g1_simulations'
observable_path = os.path.join(root_path, obj_path)
utils.mksubfolder(observable_path)

total_tasks = 1920
number_of_cores = 128
qutip.settings.num_cpus = number_of_cores
rerun = 1

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
di = 500                                                                        # sampling ratio: 1 out of every...
Nsteps = int(3e6)                                                               # number of time steps

time = {}
time['dt'] = dt
time['di'] = di
time['Nsteps'] = Nsteps

tsteady = 200
tsample = np.linspace(0, Nsteps * dt, Nsteps // di + 1) * params_init.get('tau0')[0]
tss = tsample[np.where(tsample >= tsteady)[0][0]]
time['tss'] = tss
np.savetxt(root_path + os.sep + 'refreport_t.dat', tsample[tsample >= tss])


def correlation(core, **args):
    run = args.get('run')

    run_core_identifier = os.path.join('run', str(run), '_', 'core', str(core + 1), '_').replace('/', '')
    rerun_identifier = os.path.join('job', str(rerun)).replace('/', '')

    factor = args.get('factor')
    N = 64 * factor
    dx = 0.5 / factor
    gpe = model_script.gpe(N, dx, **args)
    psi_correlation, density_correlation, density_avg, exponential_avg = gpe.time_evolution_psi(**time)

    subfolder_psi_correlation = args.get('subfolder_psi_correlation')
    subfolder_theta_correlation = args.get('subfolder_theta_correlation')
    subfolder_density_correlation = args.get('subfolder_density_correlation')
    subfolder_density_avg = args.get('subfolder_density_avg')
    np.save(subfolder_psi_correlation + run_core_identifier + rerun_identifier + '.npy', psi_correlation)
    np.save(subfolder_theta_correlation + run_core_identifier + rerun_identifier + '.npy', exponential_avg)
    np.save(subfolder_density_correlation + run_core_identifier + rerun_identifier + '.npy', density_correlation)
    np.save(subfolder_density_avg + run_core_identifier + rerun_identifier + '.npy', density_avg)
    return None


def call_avg(**args):
    keys = args.keys()
    values = (args[key] for key in keys)
    dictionaries_unique = [dict(zip(keys, combination)) for combination in itertools.product(*values)]
    for parameters in dictionaries_unique:
        full_name = utils.full_id(**parameters)
        folder = os.path.join(observable_path, full_name)
        utils.mksubfolder(folder)

        subfolder_psi_correlation = os.path.join(folder, 'psi_correlation', "")
        subfolder_theta_correlation = os.path.join(folder, 'theta_correlation', "")
        subfolder_density_correlation = os.path.join(folder, 'n_correlation', "")
        subfolder_density_avg = os.path.join(folder, 'n_avg', "")
        utils.mksubfolder(subfolder_psi_correlation)
        utils.mksubfolder(subfolder_theta_correlation)
        utils.mksubfolder(subfolder_density_correlation)
        utils.mksubfolder(subfolder_density_avg)

        parameters['subfolder_psi_correlation'] = subfolder_psi_correlation
        parameters['subfolder_theta_correlation'] = subfolder_theta_correlation
        parameters['subfolder_density_correlation'] = subfolder_density_correlation
        parameters['subfolder_density_avg'] = subfolder_density_avg
        iteration = 1
        while iteration <= total_tasks / number_of_cores:
            parameters['run'] = iteration
            print('Currently on iteration %.i of %.i ...' % (iteration, total_tasks // number_of_cores))
            parallel_map(correlation, range(number_of_cores), task_kwargs=parameters, progress_bar=True)
            iteration += 1
        for path in [subfolder_psi_correlation, subfolder_theta_correlation, subfolder_density_correlation, subfolder_density_avg]:
            tosave = path.split("/")[-2]
            result = utils.ensemble_average(path)
            np.save(observable_path + os.sep + 'AVG_' + tosave + '.npy', result)
    return None


call_avg(**params_init)

# =============================================================================
# Some tests (ignore)
# =============================================================================
'''
rho0 = 1 / params_init.get('l0')[0]**2
keys = params_init.keys()
values = (params_init[key] for key in keys)
params = [dict(zip(keys, combination)) for combination in itertools.product(*values)]
for parameters_current in params:
    gpe = model_script.gpe(64, 0.5, **parameters_current)
    #psi_correlation, density_correlation, density_avg, exponential_avg = gpe.time_evolution_psi(**time)
'''

'''
g1 = np.abs(psi_correlation)/np.sqrt(np.abs(density_avg)[0,0]*np.abs(density_avg))
g2 = np.abs(density_correlation)/np.sqrt(np.abs(density_avg)[0,0]*np.abs(density_avg))
'''


'''
import matplotlib.pyplot as pl
pl.plot(tsample, n)
pl.plot(tsample, np.ones(len(tsample))*params_init.get('ns')[0]/rho0 * (params_init.get('p')[0] - 1), c='black')
'''