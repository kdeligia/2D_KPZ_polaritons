#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 19 13:53:23 2021

@author: delis
"""

import matplotlib.pyplot as pl
from qutip import *
import os
import numpy as np
import external as ext
import model_script
import itertools

final_save_path = r'/Users/delis/Desktop'
initial_path = r'/Users/delis/Desktop' + os.sep + 'TEST_SIMULATIONS'
if os.path.isdir(initial_path) == False:
    os.mkdir(initial_path)

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

time_dict = {}
time_dict['dt'] = 0.0004
time_dict['i_start'] = 0
time_dict['di'] = 1
time_dict['N_input'] = 500000
t = ext.time(time_dict.get('dt'), time_dict.get('N_input'), time_dict.get('i_start'), time_dict.get('di'))
np.savetxt('/Users/delis/Desktop/t.dat', t)

keys = params_init.keys()
values = (params_init[key] for key in keys)
params = [dict(zip(keys, combination)) for combination in itertools.product(*values)]
for i in range(len(params)):
    params[i]['number'] = i + 1
qutip.settings.num_cpus = len(params)

'''
import matplotlib.pyplot as pl
for mydict in params:
    k, plus, minus = ext.bogoliubov(**mydict)
    k_phys = np.fft.fftshift(k)
    fig, ax = pl.subplots()
    ax.plot(k_phys, plus)
    ax.plot(k_phys, minus)
    fig.show()
'''

def evolution(i_dict, home):
    for i in range(len(params)):
        if params[i]['number'] == i_dict + 1:
            current_dict = params[i]
            break
    N = current_dict.get('N')
    p = current_dict.get('p')
    sigma = current_dict.get('sigma')
    gamma0 = current_dict.get('gamma0')
    gamma2 = current_dict.get('gamma2')
    g = current_dict.get('g')
    ns = current_dict.get('ns')
    m = current_dict.get('m')

    name = 'N' + str(N) + '_' + 'p' + str(p) + '_' + 'sigma' + str(sigma) + '_' + 'gamma' + str(gamma0) + '_' + 'gammak' + str(gamma2) + '_' + 'g' + str(g) + '_' + 'ns' + str(ns) + '_' + 'm' + str(m) 
    misc_folder = initial_path + os.sep + name
    if os.path.isdir(misc_folder) == False:
        os.mkdir(misc_folder)
    current_dict['misc_folder'] = misc_folder

    gpe = model_script.gpe(**current_dict)
    density, theta_wound = gpe.time_evolution_vortices(misc_folder, **time_dict)
    #np.savetxt(initial_path + os.sep + name + '__' + 'nvortices' + '.dat', vortex_number)
    np.save(initial_path + os.sep + name + '__' + 'density_spacetime' + '.npy', density)
    np.save(initial_path + os.sep + name + '__' + 'theta_spacetime' + '.npy', theta_wound)
    '''
    os.system(
        'ffmpeg -framerate 10 -i ' + 
        misc_folder + os.sep + 
        'fig%d.jpg ' + 
        final_save_path + os.sep + 
        name + '__' + 'movie' + '.mp4')
    '''
    return None

#parallel_map(evolution, range(len(params)), task_kwargs = dict(home = final_save_path))
#evolution(0, home=final_save_path)