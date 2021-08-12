#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 19 13:53:23 2021

@author: delis
"""

from qutip import *
import os
import numpy as np
import external as ext
import model_class

n_batch = 1
qutip.settings.num_cpus = n_batch

p_array = np.array([2])
gamma2_array = np.array([0.1])
gamma0_array = np.array([10])
sigma_array = np.array([0.02])
g_array = np.array([0])
m_array = np.array([1e-4])
gr = 0
ns = 1.
N = 2 ** 7

path = r'/Users/delis/Desktop'
final_save_path = r'/Users/delis/Desktop'
init = path + os.sep + 'VORTICES_SIMULATIONS' + '_' + 'N' + str(N) + '_' + 'ns' + str(int(ns)) + '_' + 'm' + str(m_array[0])
if os.path.isdir(init) == False:
    os.mkdir(init)
ids = ext.ids(N, p_array, sigma_array, gamma0_array, gamma2_array, g_array)

def vortices(p, gamma0, gamma2, g):
    sigma = sigma_array[np.where(p_array == p)[0][0]]
    print('--- Currently running: p = %.1f, sigma = %.2f' % (p, sigma))
    id_string = ids.get(('p=' + str(p), 'sigma=' + str(sigma), 'gamma0=' + str(gamma0), 'gamma2=' + str(gamma2), 'g=' + str(g)))
    save_folder = init + os.sep + id_string
    if os.path.isdir(save_folder) == False:
        os.mkdir(save_folder)
    gpe = model_class.gpe(p, sigma, gamma0, gamma2, g, gr = gr, ns = ns, m = m_array[0], N = N, dx = 0.5)
    nvort, dens = gpe.time_evolution_vortices(save_folder, dt = 0.001, N_input = 4000, i_start = 0, di = 2000)
    np.savetxt(final_save_path + os.sep + id_string + '_' + 'nv' + '.dat', nvort)
    np.savetxt(final_save_path + os.sep + id_string + '_' + 'dens' + '.dat', dens)
    '''
    os.system(
        'ffmpeg -framerate 10 -i ' + 
        save_folder + os.sep + 
        'fig%d.jpg ' + 
        save_local + os.sep + 
        id_string + '.mp4')
    '''
    return None

def parallel(p):
    g = g_array[0]
    for gamma2 in gamma2_array:
        gamma0 = gamma0_array[np.where(gamma2_array == gamma2)[0][0]]
        print('--- Simulation parameters: gamma0 = %.f, gamma2 = %.2f, g = %.1f, ns = %.i' % (gamma0, gamma2, g, ns))
        parallel_map(vortices, p, task_kwargs=dict(gamma0 = gamma0, gamma2 = gamma2, g = g))
    return None

parallel(p_array)