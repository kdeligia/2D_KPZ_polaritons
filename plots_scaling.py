#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 16:45:25 2020

@author: delis
"""

import os
import matplotlib.pyplot as pl
import numpy as np
import matplotlib.ticker
import warnings
from scipy.optimize import curve_fit
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
from matplotlib import rcParams as rcP
warnings.filterwarnings("ignore")

'''
fig_width_pt = 246.0
inches_per_pt = 1.0 / 72.27
golden_mean = (np.sqrt(5) - 1.0) / 2.0
fig_width = fig_width_pt * inches_per_pt
fig_height = fig_width * golden_mean * 1.25
'''

#fig_width = 4.8
fig_width = 1.2 * 2.4
#fig_height = 3.3
fig_height = 2.2

fig_size = [fig_width, fig_height]
params = {
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'axes.formatter.limits' : [-4, 4],
        'legend.columnspacing' : 1,
        'legend.fontsize' : 12,
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
        'savefig.format': 'eps',
        'savefig.bbox': 'tight'
}
rcP.update(params)
rcP.update(params)
matplotlib.rc(
    'text.latex', preamble=r"\usepackage{xcolor}")

chi = 0.39
z = 2 - chi
beta = chi / z

g0 = 1
path_correlation = r'/Users/delis/Documents/PhD/Data 2D//gy'
gyth_038 = np.loadtxt(path_correlation + os.sep + 'gy_theor_beta0.38.dat')
gyth_039 = np.loadtxt(path_correlation + os.sep + 'gy_theor_beta0.39.dat')

dx_tilde = 0.5
N = 2 ** 6
Dr = np.arange(N // 2) * dx_tilde
Dt = np.loadtxt(path_correlation + os.sep + 'Dt.dat')

def fit_th(x, a, b):
    return a * x ** (2 * chi) + b

popt, pcov = curve_fit(fit_th, gyth_039[800:, 0], gyth_039[800:, 1] / gyth_039[0, 1])
coeff_compare = popt[0]

def coeffs(temp_coeff, spat_coeff):
    C0 = temp_coeff / g0
    y0 = (spat_coeff / (C0 * coeff_compare)) ** (1 / (2 * chi))
    return C0, y0

np.seterr(divide='ignore')
def selection_Dr(C0, y0, dr, dt, g1, heatmap_g1x):
    gy_x = []
    y_x = []
    upper_lim = 16
    cutoff = 3e-2
    '''
    fig, ax = pl.subplots()
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_ylim(1e-2, 5)
    ax.plot(dr, C0 * y0 ** (2 * chi) * coeff_compare * dr ** (2 * chi))
    '''
    for i in range(1, 12):
        check = -np.gradient(- 2 * np.log(g1[i]) / dr ** (2 * chi), dr)
        #ax.plot(dr, g1[i])
        for j in range(0, upper_lim):
            if check[j] < cutoff:
                gy_x.append(g1[i, j] / (C0 * dt[i] ** (2 * beta)))
                y_x.append(y0 * dr[j] / dt[i] ** (1 / z))
                #ax.plot(dr[j], g1[i, j], 'ro-')
                heatmap_g1x[i, j] = 3
        #fig.show()
    return y_x, gy_x, heatmap_g1x

def selection_Dt(C0, y0, dt, dr, g1, heatmap_g1t):
    y_t = []
    gy_t = []
    begin = 10
    end = 350
    cutoff = 5e-1
    '''
    fig, ax = pl.subplots()
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_ylim(1e-2, 40)
    ax.plot(dt, C0 * dt ** (2 * beta))
    ax.plot(dt, g1[:, 0])
    ax.plot(dt[begin : end], g1[begin : end,0], 'red')
    '''
    y_t.extend(dr[0] / dt[begin:end] ** (1 / z))
    gy_t.extend(g1[begin:end, 0] / dt[begin:end] ** (2 * beta))
    for i in range(1, 18):
        diff = np.abs(g1[5:, i] - g1[5:, 0])
        '''
        ax.plot(dt, g1[:, i])
        ax.plot(dt[begin + np.where(diff < cutoff)[0][0]: end], g1[begin + np.where(diff < cutoff)[0][0]: end, i], 'red')
        fig.show()
        '''
        y_t.extend(y0 * dr[i] / dt[begin + np.where(diff < cutoff)[0][0]: end] ** (1 / z))
        gy_t.extend(g1[begin + np.where(diff < cutoff)[0][0]: end, i] / (C0 * dt[begin + np.where(diff < cutoff)[0][0]: end] ** (2 * beta)))
        heatmap_g1t[begin + np.where(diff < cutoff)[0][0]: end, i] = 1
    return y_t, gy_t, heatmap_g1t

# =============================================================================
# g1 from theta vs g1 from psi
# =============================================================================
hatt = 32 # ps
hatx = 4 * np.sqrt(2)
Dr *= hatx
Dt *= hatt
colors_blue = ['#c6dbef', '#9ecae1', '#6baed6', '#4292c6', '#2171b5', '#08519c', '#08306b']
colors_red = ['#fcbba1', '#fc9272', '#fb6a4a', '#ef3b2c', '#cb181d', '#a50f15', '#67000d']

name9 = 'N64_p1.05_sigma5.125_gamma0.3125_gammak0.1_g0_ns120_m8e-05'
name10 = 'N64_p1.1_sigma5.25_gamma0.3125_gammak0.1_g0_ns120_m8e-05'
name11 = 'N64_p1.2_sigma5.5_gamma0.3125_gammak0.1_g0_ns120_m8e-05'
name12 = 'N64_p1.3_sigma5.75_gamma0.3125_gammak0.1_g0_ns120_m8e-05'
name13 = 'N64_p1.5_sigma6.25_gamma0.3125_gammak0.1_g0_ns120_m8e-05'
name14 = 'N64_p1.6_sigma6.5_gamma0.3125_gammak0.1_g0_ns120_m8e-05'
name15 = 'N64_p1.7_sigma6.75_gamma0.3125_gammak0.1_g0_ns120_m8e-05'
name16 = 'N64_p1.8_sigma7.0_gamma0.3125_gammak0.1_g0_ns120_m8e-05'
name17 = 'N64_p2_sigma7.5_gamma0.3125_gammak0.1_g0_ns120_m8e-05'
name18 = 'N64_p2.5_sigma8.75_gamma0.3125_gammak0.1_g0_ns120_m8e-05'
name19 = 'N64_p3_sigma10.0_gamma0.3125_gammak0.1_g0_ns120_m8e-05'
name20 = 'N64_p4_sigma12.5_gamma0.3125_gammak0.1_g0_ns120_m8e-05'
name21 = 'N64_p5_sigma15.0_gamma0.3125_gammak0.1_g0_ns120_m8e-05'
name22 = 'N64_p10_sigma27.5_gamma0.3125_gammak0.1_g0_ns120_m8e-05'

g_psi_9 = np.load(path_correlation + os.sep + 'change p' + os.sep + name9 + '_full_g1' + '.npy')
g_psi_10 = np.load(path_correlation + os.sep + 'change p' + os.sep + name10 + '_full_g1' + '.npy')
g_psi_11 = np.load(path_correlation + os.sep + 'change p' + os.sep + name11 + '_full_g1' + '.npy')
g_psi_12 = np.load(path_correlation + os.sep + 'change p' + os.sep + name12 + '_full_g1' + '.npy')
g_psi_13 = np.load(path_correlation + os.sep + 'change p' + os.sep + name13 + '_full_g1' + '.npy')
g_psi_14 = np.load(path_correlation + os.sep + 'change p' + os.sep + name14 + '_full_g1' + '.npy')
g_psi_15 = np.load(path_correlation + os.sep + 'change p' + os.sep + name15 + '_full_g1' + '.npy')
g_psi_16 = np.load(path_correlation + os.sep + 'change p' + os.sep + name16 + '_full_g1' + '.npy')
g_psi_17 = np.load(path_correlation + os.sep + 'change p' + os.sep + name17 + '_full_g1' + '.npy')
g_psi_18 = np.load(path_correlation + os.sep + 'change p' + os.sep + name18 + '_full_g1' + '.npy')
g_theta_14 = np.load(path_correlation + os.sep + 'change p' + os.sep + name14 + '_exponential_g1' + '.npy')
g_theta_15 = np.load(path_correlation + os.sep + 'change p' + os.sep + name15 + '_exponential_g1' + '.npy')
g_theta_16 = np.load(path_correlation + os.sep + 'change p' + os.sep + name16 + '_exponential_g1' + '.npy')
g_theta_17 = np.load(path_correlation + os.sep + 'change p' + os.sep + name17 + '_exponential_g1' + '.npy')
g_theta_18 = np.load(path_correlation + os.sep + 'change p' + os.sep + name18 + '_exponential_g1' + '.npy')

g2_14 = np.load(path_correlation + os.sep + 'density density correlation' + os.sep + name14 + '_full_g2' + '.npy')
g2_15 = np.load(path_correlation + os.sep + 'density density correlation' + os.sep + name15 + '_full_g2' + '.npy')
g2_16 = np.load(path_correlation + os.sep + 'density density correlation' + os.sep + name16 + '_full_g2' + '.npy')
g2_17 = np.load(path_correlation + os.sep + 'density density correlation' + os.sep + name17 + '_full_g2' + '.npy')
g2_18 = np.load(path_correlation + os.sep + 'density density correlation' + os.sep + name18 + '_full_g2' + '.npy')

rhs_14 = g_theta_14 * g2_14
rhs_15 = g_theta_15 * g2_15
rhs_16 = g_theta_16 * g2_16
rhs_17 = g_theta_17 * g2_17
rhs_18 = g_theta_18 * g2_18

matrix_x = np.zeros((len(Dt), len(Dr)))
matrix_t = np.zeros((len(Dt), len(Dr)))
C0_17, y0_17 = coeffs(0.019, 0.039)
C0_14, y0_14 = coeffs(0.034, 0.062)
y_x, gy_x, heatmap_g1x = selection_Dr(C0_17, y0_17, Dr, Dt, -2 * np.log(g_psi_17), matrix_x)
y_t, gy_t, heatmap_g1t = selection_Dt(C0_17, y0_17, Dt, Dr, -2 * np.log(g_psi_17), matrix_t)
R_portion, T_portion = np.meshgrid(Dr[1:], Dt[1:])
R_full, T_full = np.meshgrid(Dr, Dt)
heatmap_total = (heatmap_g1x + heatmap_g1t) / 2


# =============================================================================
# Plot 1 try 1
# =============================================================================

fig, ax = pl.subplots()
ax.plot(Dt[1:800] ** (2 * beta), C0_17 * g0 * Dt[1:800] ** (2 * beta), '--', color = 'black')
ax.plot(Dt[1:] ** (2 * beta), -2 * np.log(g_psi_17[1:, 0]))
ax.set_xlabel(r'$t^{2\beta} [ps^{2\beta}]$')
ax.set_xlim(0.5, 160)
ax.set_ylim(0, 4)
#ax.text(13.2e1, 5e-1, r'$(i)$', fontsize = 12)
#ax.axvspan(3e2 ** (2*beta), 2e4 ** (2*beta), color = 'gray', alpha = 0.2)
#pl.suptitle(r'$-2 \ln g_{1, \psi}(\Delta t, \Delta r=0)$', y=1.015, fontsize = 12)
ax.set_ylabel(r'$C(t, r=0)$')
fig.suptitle(r'$C(t, r=0) \sim t^{\textcolor{purple}{2\beta}} \Leftrightarrow F_{2d}(y)=F_{2d}(0)$', fontsize = 13, y = 1.025)
fig.show()

fig, ax = pl.subplots()
ax.plot(Dr[2:16] ** (2 * chi), C0_17 * y0_17 ** (2 * chi) * coeff_compare * Dr[2:16] ** (2 * chi), '--', color = 'black')
ax.plot(Dr[1:] ** (2 * chi), -2 * np.log(g_psi_17[0, 1:]))
ax.set_xlabel(r'$\Delta r^{2 \chi} [\mu m^{2 \chi}]$')
ax.set_xlim(1.25, 20)
ax.set_ylim(0.12, 0.9)
#ax.text(2.3e0, 7.5e-1, r'$(ii)$', fontsize = 12)
#ax.axvspan(1e1 ** (2*chi), 4e1 ** (2*chi), color='gray', alpha = 0.2)
#pl.suptitle(r'$-2 \ln g_{1, \psi}(\Delta t=0, \Delta r)$', y=1.015, fontsize = 12)
ax.set_ylabel(r'$C(t=0, r)$')
fig.suptitle(r'$C(t=0, r) \sim r^{\textcolor{purple}{2\chi}} \Leftrightarrow F_{2d}(y) \sim y^{\textcolor{purple}{2\chi}}$', fontsize = 13, y= 1.025)
fig.show()


'''
y = [*y_x, *y_t]
gy = [*gy_x, *gy_t]
total = np.vstack((y, gy))
gynum = total.T[total.T[:, 0].argsort()]

fig,ax = pl.subplots()
ax.loglog(gynum[:, 0], gynum[:, 1], 'bo', markersize = 0.5)
ax.loglog(gyth_039[:, 0], gyth_039[:, 1] / gyth_039[0, 1], 'black')
ax.set_ylim(6e-1, 2e2)
#ax.set_xlabel(r'$y_0 \Delta r / \Delta t^{1/z}$')
#ax.set_ylabel(r'$ -2 \ln g_{1, \psi}(\Delta t, \Delta r) / C_0 \Delta t^{2\beta}$')
ax.set_xlabel(r'$y_0 r / t^{1/z}$')
ax.set_ylabel(r'$ C(t, r) / C_0 t^{2\beta}$')
#ax.text(2e1, 1e0, r'$(iii)$', fontsize = 10)
ax.text(1e-2, 1e2, r'$g_{1, \psi}$', fontsize=14)

axin = ax.inset_axes([0.23, 0.35, 0.55, 0.45])
axin.set_xscale('log')
axin.set_yscale('log')
axin_cbar = ax.inset_axes([0.15, 0.89, 0.7, 0.03])
im = axin.pcolormesh(R_portion, T_portion, g_psi_17[1:, 1:], shading = 'gouraud')
axin.contour(R_portion, T_portion, heatmap_total[1:, 1:], levels = 0, colors = 'red')
axin.set_xticks([1e1, 5e1])
axin.set_yticks([1e2, 1e3, 1e4, 1e5])
axin.set_yticklabels([r'$10^2$',r'$10^3$', r'$10^4$', r'$10^5$'])
axin.set_xticklabels([r'$10$', r'$50$'])
#axin.set_xlabel(r'$\Delta r [\mu m]$')
#axin.set_ylabel(r'$\Delta t [ps] $')
axin.set_xlabel(r'$r [\mu m]$', fontsize = 12)
axin.set_ylabel(r'$t [ps] $', fontsize = 12)
axin.tick_params(pad = 3.5, labelsize = 12)
axin_cbar.tick_params(pad = 3, labelsize = 12)
cbar = fig.colorbar(im, cax = axin_cbar, orientation = 'horizontal')
#axin_cbar.set_title(r'$g_{1, \psi}$', rotation=0)
fig.show()
'''
# =============================================================================
# Plot SupMat 1
# =============================================================================
'''
fig, ax = pl.subplots()
ax.set_xscale('log')
ax.set_yscale('log')
ax.plot(Dt[1:], -2 * np.log(g_psi_14[1:, 0]) / Dt[1:] ** (2 * beta), color = colors_red[2])
ax.plot(Dt[1:], -2 * np.log(g_psi_15[1:, 0]) / Dt[1:] ** (2 * beta), color = colors_red[3])
ax.plot(Dt[1:], -2 * np.log(g_psi_16[1:, 0]) / Dt[1:] ** (2 * beta), color = colors_red[4])
ax.plot(Dt[1:], -2 * np.log(g_psi_17[1:, 0]) / Dt[1:] ** (2 * beta), color = colors_red[5])
ax.plot(Dt[1:], -2 * np.log(g_psi_18[1:, 0]) / Dt[1:] ** (2 * beta), color = colors_red[6])
ax.axvspan(1e3, 2e4, color = 'gray', alpha = 0.2)
#ax.text(7.5e4, 1.25e-2, r'$(i)$', fontsize = 12)
ax.set_yticks((1e-1, 1e-2))
ax.tick_params(which='minor', labelleft=False)
#ax.set_xlabel(r'$\Delta t [ps]$')
#ax.set_ylabel(r'$-2 \ln g_{1, \psi}(\Delta t, \Delta r=0)/\Delta t^{2 \beta}$')
ax.set_xlabel(r'$t[ps]$')
ax.set_ylabel(r'$-2 \ln g_{1, \psi}(t, r=0)/t^{2 \beta}$')
fig.show()
'''

'''
fig, ax = pl.subplots()
ax.set_xscale('log')
ax.set_yscale('log')
ax.plot(Dr[1:], -2 * np.log(g_psi_14[0, 1:]) / Dr[1:] ** (2 * chi), color = colors_red[2])
ax.plot(Dr[1:], -2 * np.log(g_psi_15[0, 1:]) / Dr[1:] ** (2 * chi), color = colors_red[3])
ax.plot(Dr[1:], -2 * np.log(g_psi_16[0, 1:]) / Dr[1:] ** (2 * chi), color = colors_red[4])
ax.plot(Dr[1:], -2 * np.log(g_psi_17[0, 1:]) / Dr[1:] ** (2 * chi), color = colors_red[5])
ax.plot(Dr[1:], -2 * np.log(g_psi_18[0, 1:]) / Dr[1:] ** (2 * chi), color = colors_red[6])
ax.axvspan(10, 40, color = 'gray', alpha = 0.2)
#ax.text(6e1, 1.35e-2, r'$(ii)$', fontsize = 12)
ax.set_yticks((1e-1, 1e-2))
#ax.set_xlabel(r'$\Delta r [\mu m]$')
#ax.set_ylabel(r'$-2 \ln g_{1, \psi}(\Delta t=0, \Delta r)/\Delta r^{2 \chi}$')
ax.set_xlabel(r'$r[\mu m]$')
ax.set_ylabel(r'$-2 \ln g_{1, \psi}(t=0, r)/r^{2 \chi}$')
fig.show()
'''

# =============================================================================
# Plot SupMat 2
# =============================================================================

'''
fig, ax = pl.subplots()
ax.plot(Dt[1:560] ** (2 * beta), C0_14 * g0 * Dt[1:560] ** (2 * beta), '--', color = 'black', linewidth = 1.5)
ax.plot(Dt[1:800] ** (2 * beta), C0_17 * g0 * Dt[1:800] ** (2 * beta), '--', color = 'black', linewidth = 1.5)
ax.plot(Dt[1:] ** (2 * beta), -2 * np.log(g_psi_14[1:, 0]), color = colors_red[2])
ax.plot(Dt[1:] ** (2 * beta), -2 * np.log(g_psi_17[1:, 0]), color = colors_red[5])
ax.plot(Dt[1:] ** (2 * beta), -2 * np.log(g_theta_14[1:, 0]), color = colors_blue[2])
ax.plot(Dt[1:] ** (2 * beta), -2 * np.log(g_theta_17[1:, 0]), color = colors_blue[5])
#ax.set_xlabel(r'$\Delta t^{2\beta} [ps^{2\beta}]$')
#ax.set_ylabel(r'$-2 \ln g_{1, \psi}(\Delta t, \Delta r=0)$')
ax.set_xlabel(r'$t^{2\beta} [ps^{2\beta}]$')
ax.set_ylabel(r'$-2 \ln g_{1, \psi}(t, r=0)$')
ax.axvspan(1e3 ** (2*beta), 2e4 ** (2*beta), color = 'gray', alpha = 0.2)
ax.set_xlim(0, 200)
ax.set_ylim(0, 6)
#ax.text(1.7e2, 6e-1, r'$(i)$', fontsize = 12)
#ax.text(7e0, 5.3e0, r'$(i)$', fontsize = 12)

#axin = ax.inset_axes([0.03, 0.68, 0.35, 0.25], transform=None, zorder=5)
axin = ax.inset_axes([0.6, 0.04, 0.36, 0.2], transform=None, zorder=5)
axin.set_xscale('log')
axin.set_yscale('log')
axin.plot(Dt[1:], g2_14[1:, 0], color = '#a1d99b')
axin.plot(Dt[1:], g2_17[1:, 0], color = '#005a32')
#axin.yaxis.set_label_position("right")
#axin.yaxis.tick_right()
axin.xaxis.tick_top()
axin.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
axin.set_xticks([1e2, 1e3, 1e4, 1e5])
axin.set_yticks([1, 0.95, 0.9])
axin.set_yticklabels([r'$1$', r'$0.95$', r'$0.9$'])
axin.tick_params(which='both', labelsize = 9)
axin.set_ylabel(r'$g_{1, n}$')
fig.show()

fig, ax = pl.subplots()
ax.plot(Dr[2:16] ** (2 * chi), C0_17 * y0_17 ** (2 * chi) * coeff_compare * Dr[2:16] ** (2 * chi), '--', color = 'black', linewidth = 1.5) #color = colors_red[5], 
ax.plot(Dr[2:16] ** (2 * chi), 0.35 * C0_17 * y0_17 ** (2 * chi) * coeff_compare * Dr[2:16] ** (2 * chi), '--', color = 'black', linewidth = 1.5) #color = colors_blue[5]
ax.plot(Dr[8:16] ** (2 * chi), 1.25 * C0_14 * y0_14 ** (2 * chi) * coeff_compare * Dr[8:16] ** (2 * chi), '--', color = 'black', linewidth = 1.5) #color = colors_red[2]
ax.plot(Dr[8:17] ** (2 * chi), 0.75 * C0_14 * y0_14 ** (2 * chi) * coeff_compare * Dr[8:17] ** (2 * chi), '--', color = 'black', linewidth = 1.5) #color = colors_blue[2]
ax.plot(Dr[1:] ** (2 * chi),(-2 * np.log(g_psi_14[0, 1:])), color = colors_red[2])      #*1.25 NOTE 
ax.plot(Dr[1:] ** (2 * chi), -2 * np.log(g_psi_17[0, 1:]), color = colors_red[5])
ax.plot(Dr[1:] ** (2 * chi), (-2 * np.log(g_theta_14[0, 1:])), color = colors_blue[2])      # *1.25 NOTE
ax.plot(Dr[1:] ** (2 * chi), -2 * np.log(g_theta_17[0, 1:]), color = colors_blue[5])
ax.axvspan(10 ** (2*chi), 40 ** (2*chi), color='gray', alpha = 0.2)
ax.set_ylim(0, 2.5)
#ax.set_xlabel(r'$\Delta r^{2 \chi} [\mu m^{2 \chi}]$')
#ax.set_ylabel(r'$-2 \ln g_{1, \psi}(\Delta t=0, \Delta r)$')
ax.set_xlabel(r'$r^{2 \chi} [\mu m^{2 \chi}]$')
ax.set_ylabel(r'$-2 \ln g_{1, \psi}(t=0, r)$')
#ax.text(2.9e1, 2.15e0, r'$(ii)$', fontsize = 12)

axin = ax.inset_axes([0.05, 0.65, 0.36, 0.2], transform=None, zorder=5)
axin.set_xscale('log')
axin.set_yscale('log')
axin.plot(Dr[1:], g2_14[0, 1:], color = '#a1d99b')
axin.plot(Dr[1:], g2_17[0, 1:], color = '#005a32')
axin.xaxis.tick_top()
axin.xaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(10))
axin.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
axin.set_xticks([1e1, 1e2])
axin.set_yticks([1, 0.95, 0.9])
axin.set_yticklabels([r'$1$', r'$0.95$', r'$0.9$'])
axin.yaxis.set_label_position("right")
axin.yaxis.tick_right()
axin.tick_params(which='both', labelsize = 9)
axin.set_ylabel(r'$g_{1, n}$')
fig.show()
'''