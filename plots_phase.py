#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 14:16:40 2020

@author: delis
"""

import os
import numpy as np
from plots_functions import theta_compute_cumulants, trajectories_select, data_select, statistics, gaussian, gumbel, distributions
import matplotlib.pyplot as pl
from matplotlib import rc
from matplotlib.texmanager import TexManager
from scipy.optimize import curve_fit
from scipy import interpolate
from matplotlib import rcParams as rcP
import warnings
from scipy.stats import skew, kurtosis
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
warnings.filterwarnings("ignore")

'''
fig_width_pt = 246.0
inches_per_pt = 1.0 / 72.27
golden_mean = (np.sqrt(5) - 1.0) / 2.0
fig_width = fig_width_pt * inches_per_pt
fig_height = fig_width * golden_mean #* 1.25
'''

fig_width = 4.8
#fig_width = 2.4
fig_height = 3.3
#fig_height = 2.2

#fig_width = 1.5
#fig_height = 3

fig_size = [fig_width, fig_height]
params = {
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'axes.formatter.limits' : [-4, 4],
        'legend.columnspacing' : 1,
        'legend.fontsize' : 11,
        'legend.frameon': False,
        'axes.labelsize': 12,
        'font.size': 12,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'lines.linewidth': 1,
        'lines.markersize': 3,
        'ytick.major.pad' : 4,
        'xtick.major.pad' : 4,
        'text.usetex': True,
        'font.family' : 'sans-serif',
        'font.weight' : 'light',
        'figure.figsize': fig_size,
        'savefig.format': 'pdf',
        'savefig.bbox': 'tight'
}

rcP.update(params)
 
path_phase = r'/Users/delis/Documents/PhD/Data 2D/Phase'
TWgue, TWgoe, BR = distributions()

hatt = 32 # ps
hatx = 4 * 2 ** (1/2)
chi = 0.39
z = 2 - chi
beta = chi / z

ns = 120
m = 8e-5
sigma = 7.5
p = 2
gamma2 = 0.1
gamma0 = 0.3125
g = 0
gr = 0
file_id =  'p' + str(p) + '_' + 'sigma' + str(sigma) + '_' + 'gamma' + str(gamma0) + '_' + 'gammak' + str(gamma2) + '_' + 'g' + str(g) + '_' + 'ns' + str(ns) + '_' + 'm' + str(m)

t = np.loadtxt(path_phase + os.sep + file_id + '_' + 't_theta' + '.dat')
t *= hatt
dataL = [1 / (N * hatx * 0.5) for N in [32, 64, 128]]

def theta_treat(N, i1, i2, centralwidth, level):
    theta = np.loadtxt(path_phase + os.sep + 'N' + str(N) + '_' + file_id + '_' + 'trajectories.dat')
    if N == 64:
        every = 4
        treatcutoff = 0.1
    else:
        every = 8
        if N == 32:
            treatcutoff = 0.05
        else:
            treatcutoff = 0.15
    theta_x1 = theta[0 :: every]
    theta_x2 = theta[1 :: every]
    theta_x3 = theta[2 :: every]
    theta_x4 = theta[3 :: every]
    var_posttreat_mean = np.zeros(len(t))
    fig, ax = pl.subplots()
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.axvspan(t[i1], t[i2], color='green', alpha = 0.5)
    mylist = [theta_x1, theta_x2, theta_x3, theta_x4]
    for which in range(len(mylist)):
        theta_x0 = mylist[which]
        if which == 0:
            which_save = 'x1'
        elif which == 1:
            which_save = 'x2'
        elif which == 2:
            which_save = 'x3'
        elif which == 3:
            which_save = 'x4'
        theta_x0_select, count = trajectories_select(t, theta_x0, treatcutoff)
        print(which_save)
        print('Percentage of good trajectories = %.f' % (100 * count / len(theta_x0[:, 0])))
        var_x0_posttreat = theta_compute_cumulants(theta_x0_select)[1]
        var_posttreat_mean += var_x0_posttreat / 4
        sk_noint, ku_noint, hist_data_noint, sk_int, ku_int, hist_data_int = data_select(theta_x0, centralwidth, level, i1, i2, False)
        #np.savetxt(r'/Users/delis/Desktop' + os.sep + 'Results' + os.sep + str(N) + os.sep + 'noint' + os.sep + 'skewness' + '_' + which_save + '_' + str(centralwidth) + '_' + str(level) + '.dat', sk_noint)
        #np.savetxt(r'/Users/delis/Desktop' + os.sep + 'Results' + os.sep + str(N) + os.sep + 'noint' + os.sep + 'kurtosis' + '_' + which_save + '_' + str(centralwidth) + '_' + str(level) + '.dat', ku_noint)
        np.savetxt(r'/Users/delis/Desktop' + os.sep + 'Results' + os.sep + str(N) + os.sep + 'noint' + os.sep + 'hist_data' + '_' + which_save + '_' + str(centralwidth) + '_' + str(level) + '.dat', hist_data_noint)
        #np.savetxt(r'/Users/delis/Desktop' + os.sep + 'Results' + os.sep + str(N) + os.sep + 'int' + os.sep + 'skewness' + '_' + which_save + '_' + str(centralwidth) + '_' + str(level) + '.dat', sk_int)
        #np.savetxt(r'/Users/delis/Desktop' + os.sep + 'Results' + os.sep + str(N) + os.sep + 'int' + os.sep + 'kurtosis' + '_' + which_save + '_' + str(centralwidth) + '_' + str(level) + '.dat', ku_int)
        np.savetxt(r'/Users/delis/Desktop' + os.sep + 'Results' + os.sep + str(N) + os.sep + 'int' + os.sep + 'hist_data' + '_' + which_save + '_' + str(centralwidth) + '_' + str(level) + '.dat', hist_data_int)
    ax.plot(t, var_posttreat_mean / t ** (2 * beta))
    fig.show()
    return None

#theta_treat(32, 1550, 2000, 4, 1e-3)
#theta_treat(64, 1550, 2000, 4, 1e-3)
#theta_treat(128, 1550, 2000, 4, 1e-3)

# =============================================================================
# Plots 1-2
# =============================================================================

N = 64
theta = np.loadtxt(path_phase + os.sep + 'N' + str(N) + '_' + file_id + '_' + 'trajectories.dat')
theta1 = theta[0 :: 4]
'''
fig, ax = pl.subplots()
ax.plot(t, theta1[50])
ax.plot(t, theta1[0])
ax.plot(t, theta1[66])
ax.set_xlabel(r'$t[ps]$')
ax.set_ylabel(r'$\theta(t, \vec{r}_0)$')
ax.set_xticks([0, 3e3, 6e3])
ax.set_xticklabels([r'0', r'$3\times10^3$', r'$6\times10^3$'])
ax.set_ylim(-60, 5)
ax.text(5.8e3, -4, r'$(i)$', fontsize = 12)
axins = zoomed_inset_axes(ax, zoom = 3.4, loc = 'lower center')
axins.plot(t[(t>=900) & (t<=1042)], theta1[66, (t>=900) & (t<=1042)], 'go-', markevery = 18, markersize = 1.6, linewidth = 0.1)
axins.plot(t[(t>=1041) & (t<=1048)], theta1[66, (t>=1041) & (t<=1048)], 'go-', markevery = 1, markersize = 1.8, linewidth = 0.1)
axins.plot(t[(t>=1047) & (t<=1200)], theta1[66, (t>=1047) & (t<=1200)], 'go-', markevery = 18, markersize = 1.6, linewidth = 0.1)
x1, x2, y1, y2 = 900, 1200, -14.5, -6.5
axins.set_xlim(x1, x2)
axins.set_ylim(y1, y2)
mark_inset(ax, axins, loc1 = 2, loc2 = 3, fc="none", ec="0.5")
axins.yaxis.tick_right()
axins.xaxis.tick_top()
axins.tick_params(labelleft=False, labelright=True, labelsize = 9)
axins.set_xticks([x1, x2])
fig.show()
'''

#--for i in [1550, 2000] normally
#--for i in [1800, 1801] for the single figure of the paper --- t, theta1 only for the figure of the presentation, else REMOVE!
N = 128
#theta = np.loadtxt(path_phase + os.sep + 'N' + str(N) + '_' + file_id + '_' + 'trajectories.dat')
theta2 = theta[0 :: 8]
mylist = [theta2]
for which in range(len(mylist)):
    theta_x0 = mylist[which]
    sk_noint, ku_noint, hist_data_noint, sk_int, ku_int, hist_data_int = data_select(t, theta1, theta_x0, 4, 1e-3, 1800, 1801, True)


# =============================================================================
# Hist gather
# =============================================================================
'''
theta_x1_int_32 = np.loadtxt(r'/Users/delis/Desktop' + os.sep + 'Results' + os.sep + str(32) + os.sep + 'int' + os.sep + 'hist_data' + '_' + 'x1' + '_' + str(4) + '_' + str(1e-3) + '.dat')
theta_x2_int_32 = np.loadtxt(r'/Users/delis/Desktop' + os.sep + 'Results' + os.sep + str(32) + os.sep + 'int' + os.sep + 'hist_data' + '_' + 'x2' + '_' + str(4) + '_' + str(1e-3) + '.dat')
theta_x3_int_32 = np.loadtxt(r'/Users/delis/Desktop' + os.sep + 'Results' + os.sep + str(32) + os.sep + 'int' + os.sep + 'hist_data' + '_' + 'x3' + '_' + str(4) + '_' + str(1e-3) + '.dat')
theta_x4_int_32 = np.loadtxt(r'/Users/delis/Desktop' + os.sep + 'Results' + os.sep + str(32) + os.sep + 'int' + os.sep + 'hist_data' + '_' + 'x4' + '_' + str(4) + '_' + str(1e-3) + '.dat')
theta_x1_noint_32 = np.loadtxt(r'/Users/delis/Desktop' + os.sep + 'Results' + os.sep + str(32) + os.sep + 'noint' + os.sep + 'hist_data' + '_' + 'x1' + '_' + str(4) + '_' + str(1e-3) + '.dat')
theta_x2_noint_32 = np.loadtxt(r'/Users/delis/Desktop' + os.sep + 'Results' + os.sep + str(32) + os.sep + 'noint' + os.sep + 'hist_data' + '_' + 'x2' + '_' + str(4) + '_' + str(1e-3) + '.dat')
theta_x3_noint_32 = np.loadtxt(r'/Users/delis/Desktop' + os.sep + 'Results' + os.sep + str(32) + os.sep + 'noint' + os.sep + 'hist_data' + '_' + 'x3' + '_' + str(4) + '_' + str(1e-3) + '.dat')
theta_x4_noint_32 = np.loadtxt(r'/Users/delis/Desktop' + os.sep + 'Results' + os.sep + str(32) + os.sep + 'noint' + os.sep + 'hist_data' + '_' + 'x4' + '_' + str(4) + '_' + str(1e-3) + '.dat')

myhist = []
for theta in [theta_x1_int_32, theta_x2_int_32, theta_x3_int_32, theta_x4_int_32]:
    myhist.extend(theta)
np.savetxt('/Users/delis/Desktop/Results/histogram_data_int_32.dat', myhist)

myhist = []
for theta in [theta_x1_noint_32, theta_x2_noint_32, theta_x3_noint_32, theta_x4_noint_32]:
    myhist.extend(theta)
np.savetxt('/Users/delis/Desktop/Results/histogram_data_noint_32.dat', myhist)

theta_x1_int_64 = np.loadtxt(r'/Users/delis/Desktop' + os.sep + 'Results' + os.sep + str(64) + os.sep + 'int' + os.sep + 'hist_data' + '_' + 'x1' + '_' + str(4) + '_' + str(1e-3) + '.dat')
theta_x2_int_64 = np.loadtxt(r'/Users/delis/Desktop' + os.sep + 'Results' + os.sep + str(64) + os.sep + 'int' + os.sep + 'hist_data' + '_' + 'x2' + '_' + str(4) + '_' + str(1e-3) + '.dat')
theta_x3_int_64 = np.loadtxt(r'/Users/delis/Desktop' + os.sep + 'Results' + os.sep + str(64) + os.sep + 'int' + os.sep + 'hist_data' + '_' + 'x3' + '_' + str(4) + '_' + str(1e-3) + '.dat')
theta_x4_int_64 = np.loadtxt(r'/Users/delis/Desktop' + os.sep + 'Results' + os.sep + str(64) + os.sep + 'int' + os.sep + 'hist_data' + '_' + 'x4' + '_' + str(4) + '_' + str(1e-3) + '.dat')
theta_x1_noint_64 = np.loadtxt(r'/Users/delis/Desktop' + os.sep + 'Results' + os.sep + str(64) + os.sep + 'noint' + os.sep + 'hist_data' + '_' + 'x1' + '_' + str(4) + '_' + str(1e-3) + '.dat')
theta_x2_noint_64 = np.loadtxt(r'/Users/delis/Desktop' + os.sep + 'Results' + os.sep + str(64) + os.sep + 'noint' + os.sep + 'hist_data' + '_' + 'x2' + '_' + str(4) + '_' + str(1e-3) + '.dat')
theta_x3_noint_64 = np.loadtxt(r'/Users/delis/Desktop' + os.sep + 'Results' + os.sep + str(64) + os.sep + 'noint' + os.sep + 'hist_data' + '_' + 'x3' + '_' + str(4) + '_' + str(1e-3) + '.dat')
theta_x4_noint_64 = np.loadtxt(r'/Users/delis/Desktop' + os.sep + 'Results' + os.sep + str(64) + os.sep + 'noint' + os.sep + 'hist_data' + '_' + 'x4' + '_' + str(4) + '_' + str(1e-3) + '.dat')

myhist = []
for theta in [theta_x1_int_64, theta_x2_int_64, theta_x3_int_64, theta_x4_int_64]:
    myhist.extend(theta)
np.savetxt('/Users/delis/Desktop/Results/histogram_data_int_64.dat', myhist)

myhist = []
for theta in [theta_x1_noint_64, theta_x2_noint_64, theta_x3_noint_64, theta_x4_noint_64]:
    myhist.extend(theta)
np.savetxt('/Users/delis/Desktop/Results/histogram_data_noint_64.dat', myhist)

theta_x1_int_128 = np.loadtxt(r'/Users/delis/Desktop' + os.sep + 'Results' + os.sep + str(128) + os.sep + 'int' + os.sep + 'hist_data' + '_' + 'x1' + '_' + str(4) + '_' + str(1e-3) + '.dat')
theta_x2_int_128 = np.loadtxt(r'/Users/delis/Desktop' + os.sep + 'Results' + os.sep + str(128) + os.sep + 'int' + os.sep + 'hist_data' + '_' + 'x2' + '_' + str(4) + '_' + str(1e-3) + '.dat')
theta_x3_int_128 = np.loadtxt(r'/Users/delis/Desktop' + os.sep + 'Results' + os.sep + str(128) + os.sep + 'int' + os.sep + 'hist_data' + '_' + 'x3' + '_' + str(4) + '_' + str(1e-3) + '.dat')
theta_x4_int_128 = np.loadtxt(r'/Users/delis/Desktop' + os.sep + 'Results' + os.sep + str(128) + os.sep + 'int' + os.sep + 'hist_data' + '_' + 'x4' + '_' + str(4) + '_' + str(1e-3) + '.dat')
theta_x1_noint_128 = np.loadtxt(r'/Users/delis/Desktop' + os.sep + 'Results' + os.sep + str(128) + os.sep + 'noint' + os.sep + 'hist_data' + '_' + 'x1' + '_' + str(4) + '_' + str(1e-3) + '.dat')
theta_x2_noint_128 = np.loadtxt(r'/Users/delis/Desktop' + os.sep + 'Results' + os.sep + str(128) + os.sep + 'noint' + os.sep + 'hist_data' + '_' + 'x2' + '_' + str(4) + '_' + str(1e-3) + '.dat')
theta_x3_noint_128 = np.loadtxt(r'/Users/delis/Desktop' + os.sep + 'Results' + os.sep + str(128) + os.sep + 'noint' + os.sep + 'hist_data' + '_' + 'x3' + '_' + str(4) + '_' + str(1e-3) + '.dat')
theta_x4_noint_128 = np.loadtxt(r'/Users/delis/Desktop' + os.sep + 'Results' + os.sep + str(128) + os.sep + 'noint' + os.sep + 'hist_data' + '_' + 'x4' + '_' + str(4) + '_' + str(1e-3) + '.dat')

myhist = []
for theta in [theta_x1_int_128, theta_x2_int_128, theta_x3_int_128, theta_x4_int_128]:
    myhist.extend(theta)
np.savetxt('/Users/delis/Desktop/Results/histogram_data_int_128.dat', myhist)

myhist = []
for theta in [theta_x1_noint_128, theta_x2_noint_128, theta_x3_noint_128, theta_x4_noint_128]:
    myhist.extend(theta)
np.savetxt('/Users/delis/Desktop/Results/histogram_data_noint_128.dat', myhist)
'''
# =============================================================================
# Load data
# =============================================================================
'''
theta_list_noint_32 = np.loadtxt(path_phase + os.sep + 'Results' + os.sep + 'histogram_data_noint_32.dat')
theta_list_noint_64 = np.loadtxt(path_phase + os.sep + 'Results' + os.sep + 'histogram_data_noint_64.dat')
theta_list_noint_128 = np.loadtxt(path_phase + os.sep + 'Results' + os.sep + 'histogram_data_noint_128.dat')
'''

'''
size_list = [32, 64, 128]
for mode in ['int', 'noint']:
    skew_mean = np.zeros(3)
    kurt_mean = np.zeros(3)
    skew_std = np.zeros(3)
    kurt_std = np.zeros(3)
    for size in size_list:
        skew_interm = []
        kurt_interm = []
        for point in ['x1', 'x2', 'x3', 'x4']:
            skew_interm.append(np.mean(np.loadtxt(r'/Users/delis/Desktop' + os.sep + 'Results' + os.sep + str(size) + os.sep + mode + os.sep + 'skewness' + '_' + point + '_' + str(4) + '_' + str(1e-3) + '.dat')[1550-250:2000-250]))
            kurt_interm.append(np.mean(np.loadtxt(r'/Users/delis/Desktop' + os.sep + 'Results' + os.sep + str(size) + os.sep + mode + os.sep + 'kurtosis' + '_' + point + '_' + str(4) + '_' + str(1e-3) + '.dat')[1550-250:2000-250]))
        skew_mean[np.array(size_list) == size] = np.mean(skew_interm)
        kurt_mean[np.array(size_list) == size] = np.mean(kurt_interm)
        skew_std[np.array(size_list) == size] = np.std(skew_interm)
        kurt_std[np.array(size_list) == size] = np.std(kurt_interm)
    np.savetxt('/Users/delis/desktop/Results/skewness_mean' + '_' + mode +'.dat', skew_mean)
    np.savetxt('/Users/delis/desktop/Results/kurtosis_mean' + '_' + mode +'.dat', kurt_mean)
    np.savetxt('/Users/delis/desktop/Results/skewness_std' + '_' + mode +'.dat', skew_std)
    np.savetxt('/Users/delis/desktop/Results/kurtosis_std' + '_' + mode +'.dat', kurt_std)
'''

'''
P32_noint, q32_noint = np.histogram(theta_list_noint_32, bins = 'auto', density = True)
P64_noint, q64_noint = np.histogram(theta_list_noint_64, bins = 'auto', density = True)
P128_noint, q128_noint = np.histogram(theta_list_noint_128, bins = 'auto', density = True)
q32_noint = q32_noint[1:]
q64_noint = q64_noint[1:]
q128_noint = q128_noint[1:]

s32_noint, k32_noint = statistics(q32_noint, P32_noint)[3], statistics(q32_noint, P32_noint)[4]
s64_noint, k64_noint = statistics(q64_noint, P64_noint)[3], statistics(q64_noint, P64_noint)[4]
s128_noint, k128_noint = statistics(q128_noint, P128_noint)[3], statistics(q128_noint, P128_noint)[4]

sk_hist_noint = [-s32_noint, -s64_noint, -s128_noint]
ku_hist_noint = [k32_noint, k64_noint, k128_noint]
'''
# =============================================================================
# Cumulants
# =============================================================================

'''
sk_mean_int = np.loadtxt(path_phase + os.sep + 'Results' + os.sep + 'skewness_mean' + '_' + 'int' + '.dat')
sk_mean_noint = np.loadtxt(path_phase + os.sep + 'Results' + os.sep + 'skewness_mean' + '_' + 'noint' + '.dat')
sk_std_int = np.loadtxt(path_phase + os.sep + 'Results' + os.sep + 'skewness_std' + '_' + 'int' + '.dat')
sk_std_noint = np.loadtxt(path_phase + os.sep + 'Results' + os.sep + 'skewness_std' + '_' + 'noint' + '.dat')

fig, ax = pl.subplots()
ax.hlines(y = 0.4245, xmin = dataL[0], xmax=dataL[-1], color='black', label=r'$sk_{2d, flat}$')
ax.errorbar(dataL, -sk_mean_noint, yerr=sk_std_noint, fmt='o--', c='green', label=r'$-sk(\delta \theta)$')
ax.errorbar(dataL, -sk_mean_int, yerr=sk_std_int, fmt='o--', c='red', label=r'$-sk(\delta \theta_{int})$')
ax.set_xlabel(r'$1/L[\mu m^{-1}]$')
#ax.text(0.0065, 0.47, r'$(i)$', fontsize = 12)
#ax.legend(loc=(-0.075, 1.05), ncol=2)
ax.legend()
fig.show()


ku_mean_int = np.loadtxt(path_phase + os.sep + 'Results' + os.sep + 'kurtosis_mean' + '_' + 'int' + '.dat')
ku_mean_noint = np.loadtxt(path_phase + os.sep + 'Results' + os.sep + 'kurtosis_mean' + '_' + 'noint' + '.dat')
ku_std_int = np.loadtxt(path_phase + os.sep + 'Results' + os.sep + 'kurtosis_std' + '_' + 'int' + '.dat')
ku_std_noint = np.loadtxt(path_phase + os.sep + 'Results' + os.sep + 'kurtosis_std' + '_' + 'noint' + '.dat')

fig, ax = pl.subplots()
ax.hlines(y = 0.3445, xmin = dataL[0], xmax=dataL[-1], color='black', label=r'$ku_{2d, flat}$')
ax.errorbar(dataL, ku_mean_noint, yerr=ku_std_noint, fmt='o--', c='green', label=r'$ku(\delta \theta)$')
ax.errorbar(dataL, ku_mean_int, yerr=ku_std_int, fmt='o--', c='red', label=r'$ku(\delta \theta_{int})$')
ax.set_xlabel(r'$1/L[\mu m^{-1}]$')
#ax.legend(loc=(-0.025, 1.05), ncol=2)
#ax.text(0.0065, 0.41, r'$(ii)$', fontsize = 12)
ax.legend(loc = 'lower right')
fig.show()
'''

# =============================================================================
# Histogram plot
# =============================================================================
'''
theta_list_noint_32 = np.loadtxt(path_phase + os.sep + 'Results' + os.sep + 'histogram_data_noint_32.dat')
theta_list_noint_64 = np.loadtxt(path_phase + os.sep + 'Results' + os.sep + 'histogram_data_noint_64.dat')
theta_list_noint_128 = np.loadtxt(path_phase + os.sep + 'Results' + os.sep + 'histogram_data_noint_128.dat')

P32_noint, q32_noint = np.histogram(theta_list_noint_32, bins = 'auto', density = True)
P64_noint, q64_noint = np.histogram(theta_list_noint_64, bins = 'auto', density = True)
P128_noint, q128_noint = np.histogram(theta_list_noint_128, bins = 'auto', density = True)
q32_noint = q32_noint[1:]
q64_noint = q64_noint[1:]
q128_noint = q128_noint[1:]
'''

'''
xmin = -4.8
xmax = 3.8

x1_gaus = -10
x2_gaus = 10
qgaus = np.linspace(x1_gaus, x2_gaus, 1000)
Pgaus = gaussian(qgaus, 0, 1)

qGumbel = np.linspace(-100, 100, 100000)
PGumbel = gumbel(6, 1, 0.32, qGumbel)
qGumbel_rescaled = (qGumbel - 1) / 0.32 ** (1/2)
PGumbel_rescaled = PGumbel * 0.32 ** (1/2)

fig, ax = pl.subplots()
ax.set_yscale('log')
ax.plot(q32_noint, P32_noint, '+', c='#dd3497', markersize=3.5)
ax.plot(q64_noint, P64_noint, 'v', c='#78c679', markersize=1.8)
ax.plot(q128_noint, P128_noint, 'o', c='#525252', markersize=1)
ax.plot(qgaus[Pgaus>4e-4], Pgaus[Pgaus>4e-4], linewidth = 1.5, label=r'$\mathcal{N}(0, 1)$')
ax.plot(qGumbel_rescaled, PGumbel_rescaled, linewidth = 1.5, label=r'$\mathcal{G}_6(0, 1)$')
ax.set_xlabel(r'$\delta \theta$')
ax.set_ylabel(r'$P[\delta \theta]$')
ax.set_xlim(xmin, xmax)
ax.set_ylim(5e-4, 1e0)
lines = ax.get_lines()
legend1 = pl.legend([lines[i] for i in [0,1,2]], [r'$L=90.56 \mu \textrm{m}$', r'$L=181.12 \mu \textrm{m}$', r'$L=362.24 \mu \textrm{m}$'], ncol = 3, markerscale = 2.5, loc=(-0.18, 1.025))
legend2 = pl.legend([lines[i] for i in [3, 4]], [r'$\mathcal{N}(0, 1)$', r'$\mathcal{G}_6(0, 1)$'], loc='upper left')
ax.add_artist(legend1)
ax.add_artist(legend2)
#ax.legend(loc='upper left')

axins = ax.inset_axes([0.35, 0.15, 0.45, 0.2], zorder=5)
axins.plot(q32_noint, P32_noint, '+', c='#dd3497', markersize=3.5)
axins.plot(q64_noint, P64_noint, 'v', c='#78c679', markersize=1.8)
axins.plot(q128_noint, P128_noint, 'o', c='#525252', markersize=1)
axins.plot(qgaus[Pgaus>4e-3], Pgaus[Pgaus>4e-3], linewidth = 1.5, label=r'$\mathcal{N}(0, 1)$')
axins.plot(qGumbel_rescaled, PGumbel_rescaled, linewidth = 1.5, label=r'$\mathcal{G}_6(0, 1)$')
mark_inset(ax, axins, loc1 = 1, loc2 = 2, fc="none", ec="0.5")
x1, x2, y1, y2 = -0.9, 1.2, 2e-1, 4.8e-1
axins.set_xlim(x1, x2)
axins.set_ylim(y1, y2)
axins.tick_params(labelsize = 10)
axins.set_xticks([x1+0.1, x2-0.2])
fig.show()

print('Gaussian display', 'const = %.5f \n mean = %.5f \n variance = %.5f \n skewness = %.5f \n kurtosis = %.5f' % statistics(qgaus[(qgaus >= xmin) & (qgaus <= xmax)], Pgaus[(qgaus >= xmin) & (qgaus <= xmax)]))
print('Gaussian [-10, 10]', 'const = %.5f \n mean = %.5f \n variance = %.5f \n skewness = %.5f \n kurtosis = %.5f' % statistics(qgaus[(qgaus >= -10) & (qgaus <= 10)], Pgaus[(qgaus >= -10) & (qgaus <= 10)]))
print('Gumbel display', 'const = %.5f \n mean = %.5f \n variance = %.5f \n skewness = %.5f \n kurtosis = %.5f' % statistics(qGumbel_rescaled[(qGumbel_rescaled >= xmin) & (qGumbel_rescaled <= xmax)], PGumbel_rescaled[(qGumbel_rescaled >= xmin) & (qGumbel_rescaled <= xmax)]))
print('Gumbel [-100, 100]', 'const = %.5f \n mean = %.5f \n variance = %.5f \n skewness = %.5f \n kurtosis = %.5f' % statistics(qGumbel_rescaled[(qGumbel_rescaled >= -100) & (qGumbel_rescaled <= 100)], PGumbel_rescaled[(qGumbel_rescaled >= -100) & (qGumbel_rescaled <= 100)]))

print('Ν=32', 'const = %.5f \n mean = %.5f \n variance = %.5f \n skewness = %.5f \n kurtosis = %.5f' % statistics(q32_noint[(q32_noint >= xmin) & (q32_noint <= xmax)], P32_noint[(q32_noint >= xmin) & (q32_noint <= xmax)]))
print('Ν=64', 'const = %.5f \n mean = %.5f \n variance = %.5f \n skewness = %.5f \n kurtosis = %.5f' % statistics(q64_noint[(q64_noint >= xmin) & (q64_noint <= xmax)], P64_noint[(q64_noint >= xmin) & (q64_noint <= xmax)]))
print('Ν=128', 'const = %.5f \n mean = %.5f \n variance = %.5f \n skewness = %.5f \n kurtosis = %.5f' % statistics(q128_noint[(q128_noint >= xmin) & (q128_noint <= xmax)], P128_noint[(q128_noint >= xmin) & (q128_noint <= xmax)]))
'''
# =============================================================================
#  Gumbel tests
# =============================================================================

'''
qGumbel = np.linspace(-4, 4, 1000)
mrange = np.arange(1, 10, 0.1)
fig1, ax1 = pl.subplots()
fig2, ax2 = pl.subplots()
for m in mrange:
    PGumbel_braz = gumbel(m, 1, 0.32, qGumbel)
    #PGumbel_braz_resc = PGumbel_braz * 0.32 ** (1/2)
    #qGumbel_braz_resc = (qGumbel - 1) / 0.32 ** (1/2)
    PGumbel_ours = gumbel(m, 0, 1, qGumbel)
    s_braz, k_braz = statistics(qGumbel, PGumbel_braz)[3], statistics(qGumbel, PGumbel_braz)[4]
    s_ours, k_ours = statistics(qGumbel, PGumbel_ours)[3], statistics(qGumbel, PGumbel_ours)[4]
    #s_braz_resc, k_braz_resc = statistics(qGumbel_braz_resc, PGumbel_braz_resc)[3], statistics(qGumbel_braz_resc, PGumbel_braz_resc)[4]
    ax1.plot(m, s_braz, 'ro')
    ax2.plot(m, k_braz, 'ro')
    ax1.plot(m, s_ours, 'go')
    ax2.plot(m, k_ours, 'go')
    #ax1.plot(m, s_braz_resc, 'bo')
    #ax2.plot(m, k_braz_resc, 'bo')
ax1.hlines(y=-0.4247, xmin=mrange[0], xmax=mrange[-1], color='magenta')
ax1.hlines(y=-0.5518, xmin=mrange[0], xmax=mrange[-1], color='black')
ax1.vlines(x=6, ymin=-2, ymax=0, color='magenta')
ax2.hlines(y=0.3597, xmin=mrange[0], xmax=mrange[-1], color='magenta')
ax2.hlines(y=0.5041, xmin=mrange[0], xmax=mrange[-1], color='black')
ax2.vlines(x=6, ymin=0.07, ymax=0.9, color='magenta')
fig1.show()
fig2.show()
'''

'''
m = 6
PGumbel = gumbel(m, 1, 0.32, qGumbel)
qGumbel_resc = (qGumbel - 1) / 0.32 ** (1/2)
PGumbel_resc = PGumbel * 0.32 ** (1/2)

fig, ax = pl.subplots()
ax.set_yscale('log')
ax.plot(qGumbel, PGumbel, '+-')
ax.plot(qGumbel_resc, PGumbel_resc, 'v-')
ax.set_xlim(-4, 4)
ax.set_ylim(1e-10, 1e0)
fig.show()

print(statistics(qGumbel, PGumbel))
print(statistics(qGumbel_resc, PGumbel_resc))
'''