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

parallel_tasks = 1536
n_batch = 128
n_internal = parallel_tasks//n_batch
qutip.settings.num_cpus = n_batch

p_array = np.array([2])
gamma2_array = np.array([0.1])
gamma0_array = np.array([0.3125])
sigma_array = np.array([0.02])
g_array = np.array([0])
m_array = np.array([1e-4])
gr = 0
ns = 1.
N = 2 ** 6

path = r'/scratch/konstantinos'
final_save_path = r'/home6/konstantinos'

init = path + os.sep + 'THETA_SIMULATIONS' + '_' + 'N' + str(N) + '_' + 'ns' + str(int(ns)) + '_' + 'm' + str(m_array[0])
if os.path.isdir(init) == False:
    os.mkdir(init)
ids = ext.ids(N, p_array, sigma_array, gamma0_array, gamma2_array, g_array)

N_input = 6000000
dt = 0.001
i_start = 0
di = 500
N_i = N_input + i_start + di
t = ext.time(dt, N_i, i_start, di)
np.savetxt(final_save_path + os.sep + 't_theta_64.dat', t)

def theta_data(i_batch, p, sigma, gamma0, gamma2, g, mypath):
    for i_n in range(n_internal):
        gpe = model_script.gpe(p, sigma, gamma0, gamma2, g, gr = gr, ns = ns, m = m_array[0], N = N, dx = 0.5)
        theta_unwound = gpe.time_evolution_theta(dt = dt, N_input = N_input, i_start = i_start, di = di)
        np.savetxt(mypath + os.sep + 'trajectories_unwound' + '_' + 'core' + str(i_batch + 1) + '_' + str(i_n + 1) + '.dat', theta_unwound)
        if (i_n + 1) % 2 == 0:
            print('Core %.i finished realisation %.i \n' % (i_batch, i_n + 1))
    return None

def call_avg(final_save_path):
    for p in p_array:
        sigma = sigma_array[np.where(p_array == p)[0][0]]
        for g in g_array:
            for gamma2 in gamma2_array:
                gamma0 = gamma0_array[np.where(gamma2_array == gamma2)[0][0]]
                print('--- Simulation parameters: p = %.1f, sigma = %.2f, g = %.2f, ns = %.i' % (p, sigma, g, ns))
                id_string = ids.get(('p=' + str(p), 'sigma=' + str(sigma), 'gamma0=' + str(gamma0), 'gamma2=' + str(gamma2), 'g=' + str(g)))
                save_folder = init + os.sep + id_string
                print(save_folder)
                if os.path.isdir(save_folder) == False:
                    os.mkdir(save_folder)
                unwound_trajectories = []
                print('--- Loss rates: gamma0 = %.4f, gamma2 = %.1f' % (gamma0, gamma2))
                parallel_map(theta_data, range(n_batch), task_kwargs=dict(p = p, sigma = sigma, gamma0 = gamma0, gamma2 = gamma2, g = g, mypath = save_folder))
                for file in os.listdir(save_folder):
                    if 'trajectories_unwound' in file:
                        unwound_trajectories.append(np.loadtxt(save_folder + os.sep + file))
                np.savetxt(final_save_path + os.sep + id_string + '_' + 'unwound_trajectories' + '.dat', np.concatenate(unwound_trajectories, axis = 0))
        return None

call_avg(final_save_path)