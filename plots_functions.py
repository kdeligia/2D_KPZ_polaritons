#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 22 10:06:50 2021

@author: delis
"""

import os 
import numpy as np
from scipy.special import gamma, polygamma
from scipy import interpolate
import matplotlib.pyplot as pl
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset


def trajectories_select(t, theta_point, cutoff):
    theta_point_select = []
    count = 0
    for i in range(len(theta_point[:, 0])):
        slope = (theta_point[i, -1] - theta_point[i, 0]) / (t[-1] - t[0])
        y = theta_point[i, 0] + slope * (t - t[0])
        if np.var((y - theta_point[i]) ** 2) < cutoff:
            count += 1
            theta_point_select.append(theta_point[i])
    return np.vstack(theta_point_select), count

def data_select(t, theta_x1_plot_defence, theta, centerwidth, level, i1, i2, toplot):
    skew_noint = []
    kurt_noint = []
    skew_int = []
    kurt_int = []
    hist_data_int = []
    hist_data_noint = []
    #for i in range(250, len(theta[0])):
    for i in [i1, i2]:
        #print(i)
        theta_t0 = theta[:, i]
        P_full, q_full = np.histogram(theta_t0, bins = 'auto', density = True)
        q_full = q_full[1:]
        maxindex =  P_full == np.max(P_full)
        qmax = q_full[maxindex][-1]
        q_centraltip = q_full[(q_full >= qmax - centerwidth) & (q_full <= qmax + centerwidth)]
        P_centraltip = P_full[(q_full >= qmax - centerwidth) & (q_full <= qmax + centerwidth)]
        Pnoint = P_centraltip[P_centraltip >= level]
        qnoint = q_centraltip[P_centraltip >= level]
        
        f = interpolate.interp1d(q_centraltip, P_centraltip)
        q_int = np.linspace(q_centraltip[0], q_centraltip[-1], 1000, endpoint = True)
        P_int = f(q_int)
        Pint = P_int[P_int>= level]
        qint = q_int[P_int >= level]

        skew_int.append(statistics(qint, Pint)[3])
        kurt_int.append(statistics(qint, Pint)[4])
        skew_noint.append(statistics(qnoint, Pnoint)[3])
        kurt_noint.append(statistics(qnoint, Pnoint)[4])

        theta_int = theta_t0[(theta_t0 >= qint[0]) & (theta_t0 <= qint[-1])]
        theta_noint = theta_t0[(theta_t0 >= qnoint[0]) & (theta_t0 <= qnoint[-1])]
        if i >= i1 and i <= i2:
            hist_data_int.extend((theta_int.tolist() - np.mean(theta_int)) / np.sqrt(np.var(theta_int)))
            hist_data_noint.extend((theta_noint.tolist() - np.mean(theta_noint)) / np.sqrt(np.var(theta_noint)))
        if toplot == True:
            fig,ax = pl.subplots()
            ax.set_yscale('log')
            ax.plot(q_centraltip[0] - qmax, P_centraltip[0], 'bD', markersize = 5)
            ax.plot(q_centraltip[-1] - qmax, P_centraltip[-1], 'bD', markersize = 5)
            ax.plot(qnoint - qmax, Pnoint,'go-', linewidth = 3.2, markersize = 4)
            #x.plot(qint - qmax, Pint, 'rv', markersize = 1.2)
            ax.plot(q_full - qmax, P_full, '+-', color = 'gray', markersize = 5)
            ax.set_xticks([-8 * np.pi, -6 * np.pi, -4 * np.pi, -2 * np.pi, 0, 2 * np.pi])
            ax.set_xticklabels([r'$-8\pi$', r'$-6\pi$', r'$-4\pi$', r'$-2\pi$', '0', r'$2 \pi$'])
            ax.set_xlabel(r'$\theta_{t_i} - \theta_{t_i, max}$')
            ax.set_ylabel(r'$P[\theta_{t_i}]$')
            #ax.text(-6*np.pi, 4e-1, r'$(ii)$', fontsize = 12)
            ax.axhline(y=level, xmin=-8*np.pi, xmax=8*np.pi, linestyle = '--', c='black')
            ax.axvspan(qmax - centerwidth - qmax , qmax + centerwidth - qmax, color='gray', alpha = 0.3)
            ax.set_ylim([5e-4, 1e0])
            ax.set_xlim([-7 * np.pi, 3 * np.pi])

            left, bottom, width, height = [0.195, 0.53, 0.19, 0.25]
            ax2 = fig.add_axes([left, bottom, width, height])
            ax2.plot(t[(t>=1000) & (t<=1100)], theta_x1_plot_defence[66, (t>=1000) & (t<=1100)], 'go-', markevery = 1, markersize = 1.8, linewidth = 0.1)
            x1, x2, y1, y2 = 1000, 1100, -14.5, -6.5
            ax2.set_xlim(x1, x2)
            ax2.set_ylim(y1, y2)
            ax2.yaxis.tick_right()
            ax2.xaxis.tick_top()
            ax2.tick_params(labelleft=False, labelright=True, labelsize = 11)
            ax2.set_xticks([x1, x2])
            ax2.set_xlabel(r'$t[ps]$')
            ax2.set_ylabel(r'$\theta(t, \vec{r}_0)$')
            fig.show()
        else:
            continue
    return skew_noint, kurt_noint, hist_data_noint, skew_int, kurt_int, hist_data_int

def theta_compute_cumulants(theta):
    mean = np.mean(theta, axis = 0)
    deltatheta = theta - mean
    deltatheta2 = np.mean(deltatheta ** 2, axis = 0)
    deltatheta3 = np.mean(deltatheta ** 3, axis = 0)
    deltatheta4 = np.mean(deltatheta ** 4, axis = 0)
    sk = deltatheta3 / deltatheta2 ** (3/2)
    ku = (deltatheta4 - 3 * deltatheta2 ** 2) / (deltatheta2 ** 2)
    return deltatheta, deltatheta2, sk, ku

def gumbel(param, mean, var, q):
    Gamma = gamma(param)
    psi0 = polygamma(0, param)
    psi1 = polygamma(1, param)
    b = np.sqrt(psi1 / var)
    s = (np.log(param) - psi0) / b
    z = b * (mean - q + s)
    G = (b * param ** param) / Gamma * np.exp(- param * (z + np.exp(-z)))
    return G

def gaussian(q, mean, var):
    return 1 / np.sqrt(2 * np.pi * var) * np.exp(- 0.5 * (q - mean) ** 2 / var)

def statistics(q, pdf):
    n = np.sum(pdf) * (q[1] - q[0])
    mu1 = np.sum(pdf * q) * (q[1] - q[0]) / n
    mu2 = np.sum(pdf * (q - mu1) ** 2) * (q[1] - q[0]) / n
    mu3 = np.sum(pdf * (q - mu1) ** 3) * (q[1] - q[0]) / n
    mu4 = np.sum(pdf * (q - mu1) ** 4) * (q[1] - q[0]) / n
    mean = mu1
    variance = mu2 
    skewness = mu3 / mu2 ** (3/2)
    kurtosis = (mu4 - 3 * mu2 ** 2) / (mu2 ** 2)
    return n, mean, variance, skewness, kurtosis

def distributions():
    TWgue_data = np.genfromtxt('/Users/delis/Python Workspace/Paper Figures/Figure 2 data/GUEps.txt')
    TWgoe_data = np.genfromtxt('/Users/delis/Python Workspace/Paper Figures/Figure 2 data/GOEps.txt')
    F0_data = np.genfromtxt('/Users/delis/Python Workspace/Paper Figures/Figure 2 data/FO.txt')
    
    TWgue_return = np.zeros((2, len(TWgue_data[:, 2])))
    Pq_gue = np.exp(TWgue_data[:, 2])
    q_gue = TWgue_data[:, 0]
    ignore, mean_gue, var_gue, skew_gue, kurt_gue = statistics(q_gue, Pq_gue)
    TWgue_return[0] = -(q_gue - mean_gue) / var_gue ** (1/2)
    TWgue_return[1] = Pq_gue * var_gue ** (1/2)

    TWgoe_return = np.zeros((2, len(TWgoe_data[:, 2])))
    Pq_goe = np.exp(TWgoe_data[:, 2])
    q_goe = TWgoe_data[:, 0]
    ignore, mean_goe, var_goe, skew_goe, kurt_goe = statistics(q_goe, Pq_goe)
    TWgoe_return[0] = -(q_goe - mean_goe) / var_goe ** (1/2)
    TWgoe_return[1] = Pq_goe * var_goe ** (1/2)

    BR_return = np.zeros((2, len(F0_data[:, 2])))
    Pq_BR = np.exp(F0_data[:, 2])
    q_BR = F0_data[:, 0]
    ignore, mean_BR, var_BR, skew_BR, kurt_BR = statistics(q_BR, Pq_BR)
    BR_return[0] = -(q_BR - mean_BR) / var_BR ** (1/2)
    BR_return[1] = Pq_BR * var_BR ** (1/2)
    return TWgue_return, TWgoe_return, BR_return