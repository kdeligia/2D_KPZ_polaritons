#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  8 10:06:57 2022

@author: delis
"""

import math
import time as time
import os
import numpy as np
import matplotlib.pyplot as pl
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib import rcParams as rcP
import gc

import sys
sys.path.insert(0, '/Users/konstantinosdeligiannis/Python Workspace/2D_KPZ_polaritons')
import external as ext

pi = 3.14
fig_width_pt = 246.0
inches_per_pt = 1.0 / 72.27
golden_mean = (np.sqrt(5) - 1.0) / 2.0
fig_width = fig_width_pt * inches_per_pt
fig_height = fig_width * golden_mean * 1.25

fig_size = [fig_width, fig_height]
params = {
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'axes.formatter.limits' : [-4, 4],
        'legend.columnspacing' : 1,
        'legend.fontsize' : 10,
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

def grad3d(theta, dt, dx, dy):
    Ft = np.gradient(theta, dt, axis=0)
    Fx = np.gradient(np.unwrap(theta, axis=1), dx, axis=1)
    Fy = np.gradient(np.unwrap(theta, axis=2), dy, axis=2)
    return Ft, Fx, Fy

def curl(Ft, Fx, Fy, dt, dx, dy):
    curl_t = np.gradient(Fy, dx, axis=1) - np.gradient(Fx, dy, axis=2)
    curl_x = np.gradient(Ft, dy, axis=2) - np.gradient(Fy, dt, axis=0)
    curl_y = np.gradient(Fx, dt, axis=0) - np.gradient(Ft, dx, axis=1)
    return curl_t, curl_x, curl_y

l0 = 4 * 2 ** (1/2)
tau0 = l0 ** 2
rho0 = 1 / l0 ** 2
# =============================================================================
# Load data 
# =============================================================================
'''
offset = 12500
path = '/Users/delis/Desktop/simulations vortices/'
x = np.loadtxt(path + 'dt5e-05dx0.5/' + 'N64_dx0.5_unit5.656854249492381_xphys.dat')
y = np.loadtxt(path + 'dt5e-05dx0.5/' + 'N64_dx0.5_unit5.656854249492381_yphys.dat')
t = np.round(np.loadtxt(path + 'dt5e-05dx0.5/' + 'Ninput250000_dt5e-05_unit32.00000000000001_tphys.dat'), 3)
dx = 0.5 
dy = 0.5
dt = 5e-5 * 5

n = np.load(path + 'dt5e-05dx0.5/' + 'm8e-05_p2_gamma0.3125_gammak0.1_g0_gr0_ns3.75_density.npy')[offset:25001]
theta = np.load(path + 'dt5e-05dx0.5/' + 'm8e-05_p2_gamma0.3125_gammak0.1_g0_gr0_ns3.75_theta_unwrapped.npy')[offset:25001]
print(theta.shape)
#grad3d(theta, dt, dx, dy)

Ft = np.load(path + 'dt5e-05dx0.5/' + 'Ft.npy')
Fx = np.load(path + 'dt5e-05dx0.5/' + 'Fx.npy')
Fy = np.load(path + 'dt5e-05dx0.5/' + 'Fy.npy')

curl_t, curl_x, curl_y = curl(Ft, Fx, Fy, dt, dx, dy)
curl_t *= dx * dy
curl_x *= dt * dy
curl_y *= dt * dx
curl_t[abs(curl_t)<0.01] = 0
curl_x[abs(curl_x)<0.01] = 0
curl_y[abs(curl_y)<0.01] = 0
'''
# =============================================================================
# Convergence tests: steady state density
# =============================================================================
'''
t1 = np.loadtxt('/Users/delis/Desktop/convergence tests/f1' + os.sep + 'Ninput150000_dt5e-05_unit32.00000000000001_tphys.dat')
n1 = np.load('/Users/delis/Desktop/convergence tests/f1' + os.sep + 'm8e-05_p2_gamma0.3125_gammak0.1_g0_gr0_ns3.75_density.npy')
t2 = np.loadtxt('/Users/delis/Desktop/convergence tests/f2' + os.sep + 'Ninput600000_dt1.25e-05_unit32.00000000000001_tphys.dat')
n2 = np.load('/Users/delis/Desktop/convergence tests/f2' + os.sep + 'm8e-05_p2_gamma0.3125_gammak0.1_g0_gr0_ns3.75_density.npy')
t3 = np.loadtxt('/Users/delis/Desktop/convergence tests/f3' + os.sep + 'Ninput1350000_dt5.555555555555556e-06_unit32.00000000000001_tphys.dat')
n3 = np.load('/Users/delis/Desktop/convergence tests/f3' + os.sep + 'm8e-05_p2_gamma0.3125_gammak0.1_g0_gr0_ns3.75_density.npy')
t4 = np.loadtxt('/Users/delis/Desktop/convergence tests/f4' + os.sep + 'Ninput2000000_dt3.125e-06_unit32.00000000000001_tphys.dat')
n4 = np.load('/Users/delis/Desktop/convergence tests/f4' + os.sep + 'm8e-05_p2_gamma0.3125_gammak0.1_g0_gr0_ns3.75_density.npy')
t5 = np.loadtxt('/Users/delis/Desktop/convergence tests/f5' + os.sep + 'Ninput3750000_dt2e-06_unit32.00000000000001_tphys.dat')
n5 = np.load('/Users/delis/Desktop/convergence tests/f5' + os.sep + 'm8e-05_p2_gamma0.3125_gammak0.1_g0_gr0_ns3.75_density.npy')
t6 = np.loadtxt('/Users/delis/Desktop/convergence tests/f6' + os.sep + 'Ninput5400000_dt1.388888888888889e-06_unit32.00000000000001_tphys.dat')
n6 = np.load('/Users/delis/Desktop/convergence tests/f6' + os.sep + 'm8e-05_p2_gamma0.3125_gammak0.1_g0_gr0_ns3.75_density.npy')
t7 = np.loadtxt('/Users/delis/Desktop/convergence tests/f7' + os.sep + 'Ninput6125000_dt1.0204081632653063e-06_unit32.00000000000001_tphys.dat')
n7 = np.load('/Users/delis/Desktop/convergence tests/f7' + os.sep + 'm8e-05_p2_gamma0.3125_gammak0.1_g0_gr0_ns3.75_density.npy')

ns = 3.75
fig, ax = pl.subplots(figsize=(6, 4))
ax.plot(t1, n1 * rho0, label=r'$dx=%.4f \mu m, N=%.i$' % (l0 * 0.5, 64))
ax.plot(t2, n2 * rho0, label=r'$dx=%.4f \mu m, N=%.i$' % (l0 * 0.25, 128))
ax.plot(t3, n3 * rho0, label=r'$dx=%.4f \mu m, N=%.i$' % (l0 * 0.166, 192))
ax.plot(t4, n4 * rho0, label=r'$dx=%.4f \mu m, N=%.i$' % (l0 * 0.125, 256))
ax.plot(t5, n5 * rho0, label=r'$dx=%.4f \mu m, N=%.i$' % (l0 * 0.1, 320))
ax.plot(t6, n6 * rho0, label=r'$dx=%.4f \mu m, N=%.i$' % (l0 * 0.08333333333333333, 384))
ax.plot(t7, n7 * rho0, label=r'$dx=%.4f \mu m, N=%.i$' % (l0 * 0.07142857142857142, 448))
ax.hlines(y=ns, xmin=0, xmax=240, color = 'black', label=r'MF')
ax.legend()
fig.show()

density_ss = np.zeros((2, 7))
density_ss[1, 0] = np.mean(n1[12500:])
density_ss[1, 1] = np.mean(n2[12500:])
density_ss[1, 2] = np.mean(n3[2084:])
density_ss[1, 3] = np.mean(n4[2084:])
density_ss[1, 4] = np.mean(n5[2084:])
density_ss[1, 5] = np.mean(n6[2084:])
density_ss[1, 6] = np.mean(n7[2084:])
density_ss[0, 0] = l0 * 0.5
density_ss[0, 1] = l0 * 0.25
density_ss[0, 2] = l0 * 0.166
density_ss[0, 3] = l0 * 0.125
density_ss[0, 4] = l0 * 0.1
density_ss[0, 5] = l0 * 0.08333333333333333
density_ss[0, 6] = l0 * 0.07142857142857142

fig, ax = pl.subplots()
ax.plot(density_ss[0], density_ss[1] * rho0, 'o-')
ax.set_xlabel(r'$dx[\mu m]$')
ax.set_ylabel(r'$n_{ss} [\mu m^{-2}]$')
ax.set_xticks(np.linspace(0.5, 3, 6))
fig.show()
'''
# =============================================================================
# Treat data
# =============================================================================

def dict_fill(keyword, A):
    a, b, c = np.where(abs(A) >= pi/2)
    mydict = {}
    if keyword == 't projection':
        unique = list(set(a))
        for i in range(len(unique)):
            mydict[unique[i]] = []
        for i in range(len(a)):
            mydict[a[i]].append([b[i], c[i]])
        return mydict
    elif keyword == 'x projection':
        unique = list(set(b))
        for i in range(len(unique)):
            mydict[unique[i]] = []
        for i in range(len(b)):
            mydict[b[i]].append([a[i], c[i]])
        return mydict
    elif keyword == 'y projection':
        unique = list(set(c))
        for i in range(len(unique)):
            mydict[unique[i]] = []
        for i in range(len(c)):
            mydict[c[i]].append([a[i], b[i]])
        return mydict

def decluster_simple(keyword, key, lists, vort_x, vort_y, vort_t):
    incl_pos = True
    incl_neg = True
    l = []
    if keyword == 'x projection':
        x_index = key
        for i in range(len(lists)):
            t_index = lists[i][0]
            y_index = lists[i][1]
            if vort_t[i] == 0 and vort_x[i] > 0 and vort_y[i] == 0 and incl_pos == True:
                l.append([x_index, y_index, t_index, 2 * pi, 0, 0])
                incl_pos = False
            elif vort_t[i] == 0 and vort_x[i] < 0 and vort_y[i] == 0 and incl_neg == True:
                l.append([x_index, y_index, t_index, - 2 * pi, 0, 0])
                incl_neg = False
        return l
    if keyword == 'y projection':
        y_index = key
        for i in range(len(lists)):
            t_index = lists[i][0]
            x_index = lists[i][1]
            if vort_t[i] == 0 and vort_x[i] == 0 and vort_y[i] > 0 and incl_pos == True:
                l.append([x_index, y_index, t_index, 0, 2 * pi, 0])
                incl_pos = False
            elif vort_t[i] == 0 and vort_x[i] == 0 and vort_y[i] < 0 and incl_neg == True:
                l.append([x_index, y_index, t_index, 0, -2 * pi, 0])
                incl_neg = False
        return l
    if keyword == 't projection':
        t_index = key
        for i in range(len(lists)):
            x_index = lists[i][0]
            y_index = lists[i][1]
            if vort_t[i] > 0 and incl_pos == True:
                l.append([x_index, y_index, t_index, 0, 0, 2 * pi])
                incl_pos = False
            elif vort_t[i] < 0 and incl_neg == True:
                l.append([x_index, y_index, t_index, 0, 0, -2 * pi])
                incl_neg = False
        return l

def decluster_advanced(keyword, key, lists, curl_x, curl_y, curl_t):
    lists.sort()
    list_pairs = []
    mydict = {}
    index = 0
    mydict[index] = []
    ref = lists[0]
    if keyword == 'x projection':
        x_index = key
        for i in range(len(lists)):
            if abs(lists[i][0] - ref[0]) <= 10 and abs(lists[i][1] - ref[1]) <= 2:
                mydict[index].append(lists[i])
            else:
                index += 1
                ref = lists[i]
                mydict[index] = []
                mydict[index].append(lists[i])
        for i in range(len(mydict.keys())):
            mean_index_y = 0
            mean_index_t = 0
            cluster_vx = 0
            cluster = mydict.get(i)
            for pair in cluster:
                t_index = pair[0]
                y_index = pair[1]
                mean_index_t += t_index / len(cluster)
                mean_index_y += y_index / len(cluster)
                cluster_vx += curl_x[t_index, x_index, y_index]
            if np.round(abs(cluster_vx)) > 0.1:
                if np.sign(cluster_vx) > 0:
                    list_pairs.append([x_index, math.ceil(mean_index_y), math.ceil(mean_index_t), 2 * pi, 0, 0])
                elif np.sign(cluster_vx) < 0:
                    list_pairs.append([x_index, math.floor(mean_index_y), math.floor(mean_index_t), -2 * pi, 0, 0])
    if keyword == 'y projection':
        y_index = key
        for i in range(len(lists)):
            if abs(lists[i][0] - ref[0]) <= 10 and abs(lists[i][1] - ref[1]) <= 2:
                mydict[index].append(lists[i])
            else:
                index += 1
                ref = lists[i]
                mydict[index] = []
                mydict[index].append(lists[i])
        for i in range(len(mydict.keys())):
            mean_index_x = 0
            mean_index_t = 0
            cluster_vy = 0
            cluster = mydict.get(i)
            for pair in cluster:
                t_index = pair[0]
                x_index = pair[1]
                mean_index_t += t_index / len(cluster)
                mean_index_x += x_index / len(cluster)
                cluster_vy += curl_y[t_index, x_index, y_index]
            if np.round(abs(cluster_vy)) > 0.1:
                if np.sign(cluster_vy) > 0:
                    list_pairs.append([math.ceil(mean_index_x), y_index, math.ceil(mean_index_t), 0, 2 * pi, 0])
                elif np.sign(cluster_vy) < 0:
                    list_pairs.append([math.floor(mean_index_x), y_index, math.floor(mean_index_t), 0, -2 * pi, 0])
    if keyword == 't projection':
        t_index = key
        for i in range(len(lists)):
            if abs(lists[i][0] - ref[0]) <= 4 and abs(lists[i][1] - ref[1]) <= 4:
                mydict[index].append(lists[i])
            else:
                index += 1
                ref = lists[i]
                mydict[index] = []
                mydict[index].append(lists[i])
        for i in range(len(mydict.keys())):
            mean_index_x = 0
            mean_index_y = 0
            cluster_vt = 0
            cluster = mydict.get(i)
            for pair in cluster:
                x_index = pair[0]
                y_index = pair[1]
                mean_index_x += x_index / len(cluster)
                mean_index_y += y_index / len(cluster)
                cluster_vt += curl_t[t_index, x_index, y_index]
            if np.round(abs(cluster_vt)) > 0.1:
                if np.sign(cluster_vt) > 0:
                    list_pairs.append([math.ceil(mean_index_x), math.ceil(mean_index_y), t_index, 0, 0, 2 * pi])
                elif np.sign(cluster_vt) < 0:
                    list_pairs.append([math.floor(mean_index_x), math.floor(mean_index_y), t_index, 0, 0, -2 * pi])
    return list_pairs

def scan_t(arg, curl_x, curl_y, curl_t):
    keys = arg.keys()
    list_pairs_final = []
    mean_density = 0
    count = 0
    for key in keys:
        t_index = key
        vort = []
        if len(arg.get(key)) == 4 or len(arg.get(key)) == 6:
            for i in range(len(arg.get(key))):
                x_index = arg.get(key)[i][0]
                y_index = arg.get(key)[i][1]
                vort.append(np.round(curl_t[t_index, x_index, y_index], 2))
            if np.sum(vort) < 0.01:
                list_pairs = decluster_simple('t projection', key, arg.get(key), None, None, vort)
                count += 1
                mean_density += len(list_pairs) / (179.6051224213831 * 179.6051224213831)
                for item in list_pairs:
                    list_pairs_final.append(item)
        else:
            list_pairs = decluster_advanced('t projection', key, arg.get(key), curl_x, curl_y, curl_t)
            if len(list_pairs) != 0:
                count += 1
                mean_density += len(list_pairs) / (179.6051224213831 * 179.6051224213831)
                for item in list_pairs:
                    list_pairs_final.append(item)
    return  np.array(list_pairs_final), mean_density / count

def scan_x(arg, curl_x, curl_y, curl_t):
    keys = arg.keys()
    list_pairs_final = []
    mean_density = 0
    count = 0
    for key in keys:
        x_index = key
        vort_t = []
        vort_x = []
        vort_y = []
        if len(arg.get(key)) == 4 or len(arg.get(key)) == 6:
            for i in range(len(arg.get(key))):
                t_index = arg.get(key)[i][0]
                y_index = arg.get(key)[i][1]
                vort_t.append(np.round(curl_t[t_index, x_index, y_index], 2))
                vort_x.append(np.round(curl_x[t_index, x_index, y_index], 2))
                vort_y.append(np.round(curl_y[t_index, x_index, y_index], 2))
            if abs(np.sum(vort_x)) < 0.01:
                list_pairs = decluster_simple('x projection', key, arg.get(key), vort_x, vort_y, vort_t)
                count += 1
                mean_density += len(list_pairs) / (100 * 179.6051224213831)
                for item in list_pairs:
                    list_pairs_final.append(item)
        else:
            list_pairs = decluster_advanced('x projection', key, arg.get(key), curl_x, curl_y, curl_t)
            if len(list_pairs) != 0:
                count += 1
                mean_density += len(list_pairs) / (100 * 179.6051224213831)
                for item in list_pairs:
                    list_pairs_final.append(item)
    return np.array(list_pairs_final), mean_density / count

def scan_y(arg, curl_x, curl_y, curl_t):
    keys = arg.keys()
    list_pairs_final = []
    mean_density = 0
    count = 0
    for key in keys:
        y_index = key
        vort_t = []
        vort_x = []
        vort_y = []
        if len(arg.get(key)) == 4 or len(arg.get(key)) == 6:
            for i in range(len(arg.get(key))):
                t_index = arg.get(key)[i][0]
                x_index = arg.get(key)[i][1]
                vort_t.append(np.round(curl_t[t_index, x_index, y_index], 2))
                vort_x.append(np.round(curl_x[t_index, x_index, y_index], 2))
                vort_y.append(np.round(curl_y[t_index, x_index, y_index], 2))
            if abs(np.sum(vort_y)) < 0.01:
                list_pairs = decluster_simple('y projection', key, arg.get(key), vort_x, vort_y, vort_t)
                count += 1
                mean_density += len(list_pairs) / (100 * 179.6051224213831)
                for item in list_pairs:
                    list_pairs_final.append(item)
        else:
            list_pairs = decluster_advanced('y projection', key, arg.get(key), curl_x, curl_y, curl_t)
            if len(list_pairs) != 0:
                count += 1
                mean_density += len(list_pairs) / (100 * 179.6051224213831)
                for item in list_pairs:
                    list_pairs_final.append(item)
    return np.array(list_pairs_final), mean_density / count

def spatiotemporal_vortices(f):
    dx = 0.5 / f
    dy = dx
    path = '/Users/konstantinosdeligiannis/Documents/PhD/Data 2D' +  os.sep + 'convergence tests' + os.sep + 'f' + str(f)
    for element in os.listdir(path):
        if 'tphys' in element:
            t = np.loadtxt(path + os.sep + element)
            dt = (t[1] - t[0]) / tau0
            i1 = np.where(t>=100)[0][0]
            try:
                i2 = np.where(t>=200)[0][0]
            except IndexError:
                i2 = len(t)-1
    print('---> N = %.i, dx = %.5f' % (64 * f, dx * l0))
    theta = np.load(path + os.sep + 'm8e-05_p2_gamma0.3125_gammak0.1_g0_gr0_ns3.75_theta_unwrapped.npy')[i1 : i2+1]

    Ft, Fx, Fy = grad3d(theta, dt, dx, dy)
    curl_t, curl_x, curl_y = curl(Ft, Fx, Fy, dt, dx, dy)
    curl_t *= dx * dy
    curl_x *= dt * dy
    curl_y *= dt * dx
    curl_t[abs(curl_t)<0.01] = 0
    curl_x[abs(curl_x)<0.01] = 0
    curl_y[abs(curl_y)<0.01] = 0

    mydict_t = dict_fill('t projection', curl_t)
    pos_pairs_xy, mean_density_xy = scan_t(mydict_t, curl_x, curl_y, curl_t)
    print('---> Total pairs in XY plane = %.i' %(pos_pairs_xy.shape[0]//2))
    print('---> Mean density in XY plane = %.5f' %mean_density_xy)
    
    mydict_x = dict_fill('x projection', curl_x)
    pos_pairs_yt, mean_density_yt = scan_x(mydict_x, curl_x, curl_y, curl_t)
    print('---> Total pairs in YT plane = %.i' %(pos_pairs_yt.shape[0]//2))
    print('---> Mean density in YT plane = %.5f' %mean_density_yt)

    mydict_y = dict_fill('y projection', curl_y)
    pos_pairs_xt, mean_density_xt = scan_y(mydict_y, curl_x, curl_y, curl_t)
    print('---> Total pairs in XT plane = %.i' %(pos_pairs_xt.shape[0]//2))
    print('---> Mean density in XT plane = %.5f' %mean_density_xt)

    np.savetxt(path + os.sep + 'pos_pairs_xy.dat', pos_pairs_xy)
    del pos_pairs_xy
    np.savetxt(path + os.sep + 'pos_pairs_xt.dat', pos_pairs_xt)
    del pos_pairs_xt
    np.savetxt(path + os.sep + 'pos_pairs_yt.dat', pos_pairs_yt)
    del pos_pairs_yt

    return dx, (mean_density_yt + mean_density_xt) / 2, mean_density_xy

def spatial_vortices(p):
    dx = 0.5
    dy = dx
    path = '/Users/konstantinosdeligiannis/Documents/PhD/Data 2D' +  os.sep + 'vortices' + os.sep + 'pump tests' + os.sep
    for element in os.listdir(path):
        if 'tphys' in element:
            t = np.loadtxt(path + os.sep + element)
            dt = (t[1] - t[0]) / tau0
            i1 = np.where(t>=100)[0][0]
            try:
                i2 = np.where(t>=200)[0][0]
            except IndexError:
                i2 = len(t)-1
    theta = np.load(path + os.sep + 'm8e-05' + '_' + 'p' + str(p) + '_gamma0.3125_gammak0.1_g0_gr0_ns3.75_theta_unwrapped.npy')[i1 : i2+1]
    Ft, Fx, Fy = grad3d(theta, dt, dx, dy)
    curl_t, curl_x, curl_y = curl(Ft, Fx, Fy, dt, dx, dy)
    curl_t *= dx * dy
    curl_x *= dt * dy
    curl_y *= dt * dx
    curl_t[abs(curl_t)<0.01] = 0
    curl_x[abs(curl_x)<0.01] = 0
    curl_y[abs(curl_y)<0.01] = 0
    
    mydict_t = dict_fill('t projection', curl_t)
    pos_pairs_xy, mean_density_xy = scan_t(mydict_t, curl_x, curl_y, curl_t)
    print('---> p = %.2f', p)
    print('---> Total pairs in XY plane = %.i' %(pos_pairs_xy.shape[0]//2))
    print('---> Mean density in XY plane = %.5f' %mean_density_xy)
    np.savetxt(path + os.sep + 'p' + str(p) + '_' + 'pos_pairs_xy.dat', pos_pairs_xy)
    return mean_density_xy

'''
d = np.zeros((4, 7))
d[0] = [1, 2, 3, 4, 5, 6, 7]
for i in range(len(d[0])):
    f = int(d[0, i])
    d[1, i], d[2, i], d[3, i] = spatiotemporal_vortices(f)
np.savetxt('/Users/konstantinosdeligiannis/Documents/PhD/Data 2D' +  os.sep + 'vortices' + os.sep + 'mean_density_fig2.dat', d)
'''

'''
d = np.zeros((2, 8))
d[0] = [1.02, 1.05, 1.1, 1.2, 1.4, 1.6, 1.8, 2]
for i in range(len(d[0])):
    p = d[0, i]
    if p == 2:
        p = int(p)
    d[1, i] = spatial_vortices(p)
np.savetxt('/Users/konstantinosdeligiannis/Documents/PhD/Data 2D' +  os.sep + 'vortices' + os.sep + 'mean_density_fig1.dat', d)
'''
# =============================================================================
# Fig 1
# =============================================================================
'''
mean_density_fig1 = np.array(np.loadtxt('/Users/delis/Desktop/pump tests/mean_density_fig1.dat'))
fig, ax = pl.subplots()
ax.set_ylabel(r'$\overline{d}_{vort, xy}\,[\mu m^{-2}]$')
ax.set_xlabel(r'$p$')
ax.set_yscale('log')
ax.plot(mean_density_fig1[0], 2 * mean_density_fig1[1], 'o-')

p = 1.2
path = '/Users/delis/Desktop/pump tests/'
for element in os.listdir(path):
    if 'tphys' in element:
        t = np.loadtxt(path + os.sep + element)
        i1 = np.where(t>=100)[0][0]
        i = np.where(t>=150)[0][0]
    if 'xphys' in element:
        x = np.loadtxt(path + os.sep + element)
    if 'yphys' in element:
        y = np.loadtxt(path + os.sep + element)
X, Y = np.meshgrid(x, y, indexing='ij')
print('---> p = %.2f' % p)
theta = np.load(path + os.sep + 'm8e-05' + '_' + 'p' + str(p) + '_gamma0.3125_gammak0.1_g0_gr0_ns3.75_theta_unwrapped.npy')[i]

positions = np.loadtxt(path + os.sep + 'p' + str(p) + '_' + 'pos_pairs_xy.dat')
axins1 = ax.inset_axes([1.35e-1, 1.85e-1, 0.3, 0.32])
axins1.set_ylabel(r'$y[\mu m]$', fontsize = 9, labelpad = -8)
axins1.set_xlabel(r'$x[\mu m]$', fontsize = 9, labelpad = -1)
axins1.tick_params(which='both', labelsize = 9)
axins1.set_xticks([-75, 0, 75])
axins1.set_yticks([-75, 0, 75])
im1 = axins1.pcolormesh(X, Y, np.angle(np.exp(1j * theta[:, :])), cmap='twilight', shading='gouraud')
cax1 = inset_axes(axins1,
                 width='100%',
                 height='5%',
                 loc='lower left',
                 bbox_to_anchor=(0., 1.05, 1, 1),
                 bbox_transform=axins1.transAxes,
                 borderpad=0)
cbar1 = fig.colorbar(im1, cax=cax1, orientation='horizontal')
cax1.set_title(r'$\theta_{t \simeq 150ps}(x, y)$', pad=-0.2, fontsize = 9)
cbar1.ax.set_xticks([-np.pi, np.pi])
cbar1.ax.set_xticklabels([r'$-\pi$', r'$\pi$'], fontsize = 9)
cax1.tick_params(pad = -0.02)
cax1.xaxis.set_ticks_position('top')
rows = np.where(positions[:, 2] + i1 == i)[0]
for row in rows:
    x_index = int(positions[row][0])
    y_index = int(positions[row][1])
    if positions[row, 5] > 0:
        axins1.plot(x[x_index], y[y_index], marker=(5, 2), color='green', markersize=1)
    elif positions[row, 5] < 0:
        axins1.plot(x[x_index], y[y_index], marker=(5, 2), color='red', markersize=1)

p = 1.8
path = '/Users/delis/Desktop/pump tests/'
for element in os.listdir(path):
    if 'tphys' in element:
        t = np.loadtxt(path + os.sep + element)
        i1 = np.where(t>=100)[0][0]
        i = np.where(t>=150)[0][0]
    if 'xphys' in element:
        x = np.loadtxt(path + os.sep + element)
    if 'yphys' in element:
        y = np.loadtxt(path + os.sep + element)
X, Y = np.meshgrid(x, y, indexing='ij')
print('---> p = %.2f' % p)
theta = np.load(path + os.sep + 'm8e-05' + '_' + 'p' + str(p) + '_gamma0.3125_gammak0.1_g0_gr0_ns3.75_theta_unwrapped.npy')[i+8]

positions = np.loadtxt(path + os.sep + 'p' + str(p) + '_' + 'pos_pairs_xy.dat')
axins2 = ax.inset_axes([6.5e-1, 4.5e-1, 0.3, 0.32])
axins2.set_ylabel(r'$y[\mu m]$', fontsize = 9, labelpad = -8)
axins2.set_xlabel(r'$x[\mu m]$', fontsize = 9, labelpad = -1)
axins2.tick_params(which='both', labelsize = 9)
axins2.set_xticks([-75, 0, 75])
axins2.set_yticks([-75, 0, 75])
im2 = axins2.pcolormesh(X, Y, np.angle(np.exp(1j * theta[:, :])), cmap='twilight', shading='gouraud')
cax2 = inset_axes(axins2,
                 width='100%',
                 height='5%',
                 loc='lower left',
                 bbox_to_anchor=(0., 1.05, 1, 1),
                 bbox_transform=axins2.transAxes,
                 borderpad=0)
cbar2 = fig.colorbar(im2, cax=cax2, orientation='horizontal')
cax2.set_title(r'$\theta_{t \simeq 150ps}(x, y)$', pad=-0.2, fontsize = 9)
cbar2.ax.set_xticks([-np.pi, np.pi])
cbar2.ax.set_xticklabels([r'$-\pi$', r'$\pi$'], fontsize = 9)
cax2.tick_params(pad = -0.02)
cax2.xaxis.set_ticks_position('top')
rows = np.where(positions[:, 2] + i1 == i+8)[0]
for row in rows:
    x_index = int(positions[row][0])
    y_index = int(positions[row][1])
    if positions[row, 5] > 0:
        axins2.plot(x[x_index], y[y_index], marker=(5, 2), color='green', markersize=2)
    elif positions[row, 5] < 0:
        axins2.plot(x[x_index], y[y_index], marker=(5, 2), color='red', markersize=2)
fig.show()
'''
# =============================================================================
# Fig 2
# =============================================================================
'''
from matplotlib.ticker import FixedLocator
mean_density_fig2 = np.array(np.loadtxt('/Users/delis/Desktop/convergence tests/mean_density_fig2.dat'))
fig, ax = pl.subplots()
ax.set_ylabel(r'$\overline{d}_{vort}[\mu m^{-2}, \mu m^{-1} ps^{-1}]$')
ax.set_xlabel(r'$dx[\mu m]$')
ax.set_yscale('log')
ax.plot(mean_density_fig2[1] * l0, mean_density_fig2[3], 'o-', label=r'$xy$')
ax.plot(mean_density_fig2[1] * l0, mean_density_fig2[2], 'o-', label=r'$xt+yt$')
ax.set_xticks(np.linspace(0, 3, 7))
ax.set_xticklabels(['$0$', '$0.5$', '$1$', '$1.5$', '$2$', '$2.5$', '$3$'])
ax.yaxis.set_major_locator(FixedLocator(locs = np.logspace(-8, 1, 10)))
ax.set_ylim([1e-8, 1e1])
ax.set_xlim([0, 3.5])
ax.legend(loc='lower right')

f = 2
dx = 0.5 / f
N = 64 * f
path = '/Users/delis/Desktop/convergence tests/' + 'f' + str(f)
for element in os.listdir(path):
    if 'tphys' in element:
        t = np.loadtxt(path + os.sep + element)
        i1 = np.where(t>=100)[0][0]
        try:
            i2 = np.where(t>=200)[0][0]
        except IndexError:
            i2 = -1
    if 'xphys' in element:
        x = np.loadtxt(path + os.sep + element)
    if 'yphys' in element:
        y = np.loadtxt(path + os.sep + element)
T, Y = np.meshgrid(t[i1 : i2+1 : 50], x, indexing='ij')

print('---> N = %.i, dx = %.5f' % (N, dx * l0))
theta = np.load(path + os.sep + 'm8e-05_p2_gamma0.3125_gammak0.1_g0_gr0_ns3.75_theta_unwrapped.npy')[i1 : i2+1]
positions = np.loadtxt(path + os.sep + 'pos_pairs_yt.dat')
axins1 = ax.inset_axes([5.815e-1, 6.5e-1, 0.28, 0.32])
axins1.set_ylabel(r'$y[\mu m]$', fontsize = 9, labelpad = -0.25)
axins1.set_xlabel(r'$t[ps]$', fontsize = 9, labelpad = -8)
axins1.set_xticks([100, 200])
axins1.tick_params(which='both', labelsize = 9)
im1 = axins1.pcolormesh(T, Y, np.angle(np.exp(1j * theta[::50, N//2, :])), cmap='twilight', shading='gouraud')
cax1 = inset_axes(axins1,
                 width="5%",
                 height="100%",
                 loc='lower left',
                 bbox_to_anchor=(1.05, 0., 1, 1),
                 bbox_transform=axins1.transAxes,
                 borderpad=0)
cbar1 = fig.colorbar(im1, cax=cax1)
cbar1.set_label(r'$\theta_{x=0}(t, y)$', fontsize = 9, labelpad = -4)
cax1.set_yticks([-np.pi, np.pi])
cax1.set_yticklabels([r'-$\pi$', r'$\pi$'])
rows = np.where(positions[:, 0] == N//2)[0]
for row in rows:
    y_index = int(positions[row][1])
    t_index = int(positions[row][2]) + i1
    if positions[row, 3] > 0:
        axins1.plot(t[t_index], y[y_index], marker=(5, 2), color='green', markersize=0.3)
    elif positions[row, 3] < 0:
        axins1.plot(t[t_index], y[y_index], marker=(5, 2), color='red', markersize=0.3)

f = 4
dx = 0.5 / f
N = 64 * f
path = '/Users/delis/Desktop/convergence tests/' + 'f' + str(f)
for element in os.listdir(path):
    if 'tphys' in element:
        t = np.loadtxt(path + os.sep + element)
        i1 = np.where(t>=100)[0][0]
        try:
            i2 = np.where(t>=200)[0][0]
        except IndexError:
            i2 = -1
    if 'xphys' in element:
        x = np.loadtxt(path + os.sep + element)
    if 'yphys' in element:
        y = np.loadtxt(path + os.sep + element)
T, Y = np.meshgrid(t[i1 : i2 : 6], x, indexing='ij')
print('---> N = %.i, dx = %.5f' % (N, dx * l0))
theta = np.load(path + os.sep + 'm8e-05_p2_gamma0.3125_gammak0.1_g0_gr0_ns3.75_theta_unwrapped.npy')[i1 : i2]
positions = np.loadtxt(path + os.sep + 'pos_pairs_yt.dat')
axins2 = ax.inset_axes([1.75e-1, 1.2e-1, 0.3, 0.32])
axins2.set_ylabel(r'$y[\mu m]$', fontsize = 9, labelpad=-0.25)
axins2.set_xlabel(r'$t[ps]$', fontsize = 9, labelpad=-8)
axins2.set_xticks([100.032, 199.68])
axins2.set_xticklabels([r'$100$', r'$200$'])
axins2.tick_params(which='both', labelsize = 9)
axins2.set_xlim([t[i1], t[i2]])
im2 = axins2.pcolormesh(T, Y, np.angle(np.exp(1j * theta[::6, N//2, :])), cmap='twilight', shading='gouraud')
cax2 = inset_axes(axins2,
                 width="5%",
                 height="100%",
                 loc='lower left',
                 bbox_to_anchor=(1.05, 0., 1, 1),
                 bbox_transform=axins2.transAxes,
                 borderpad=0)
cbar2 = fig.colorbar(im2, cax=cax2)
cbar2.set_label(r'$\theta_{x=0}(t, y)$', fontsize = 9, labelpad = -4)
cax2.set_yticks([-np.pi, np.pi])
cax2.set_yticklabels([r'-$\pi$', r'$\pi$'])
rows = np.where(positions[:, 0] == N//2)[0]
for row in rows:
    y_index = int(positions[row][1])
    t_index = int(positions[row][2]) + i1
    if positions[row, 3] > 0:
        axins2.plot(t[t_index], y[y_index], marker=(5, 2), color='green', markersize=0.0001)
    elif positions[row, 3] < 0:
        axins2.plot(t[t_index], y[y_index], marker=(5, 2), color='red', markersize=0.0001)
fig.show()
'''
# =============================================================================
# Trajectories vs time
# =============================================================================
'''
fig, ax = pl.subplots()
for i in range(len(pos_vortices_xt[:, 0])):
    ax.plot(t, theta[:, np.where(x==pos_vortices_xt[i, 0])[0][0], np.where(y==pos_vortices_xt[i, 1])[0][0]], c='black')
    ax.plot(t[np.where(t==pos_vortices_xt[i, 2])[0][0]], 
            theta[np.where(t==pos_vortices_xt[i, 2])[0][0], np.where(x==pos_vortices_xt[i, 0])[0][0], np.where(y==pos_vortices_xt[i, 1])[0][0]], 'ro')
for i in range(len(pos_vortices_yt[:, 0])):
    ax.plot(t, theta[:, np.where(x==pos_vortices_xt[i, 0])[0][0], np.where(y==pos_vortices_xt[i, 1])[0][0]], c='black')
    ax.plot(t[np.where(t==pos_vortices_xt[i, 2])[0][0]], 
            theta[np.where(t==pos_vortices_xt[i, 2])[0][0], np.where(x==pos_vortices_xt[i, 0])[0][0], np.where(y==pos_vortices_xt[i, 1])[0][0]], 'ro')
fig.show()
'''


# =============================================================================
# test vortex at (N//2, N//2)
# =============================================================================

'''
import math as m
path = '/Users/delis/Desktop/simulations vortices/'
x = np.loadtxt(path + 'dt5e-05dx0.5/' + 'N64_dx0.5_unit5.656854249492381_xphys.dat')
y = np.loadtxt(path + 'dt5e-05dx0.5/' + 'N64_dx0.5_unit5.656854249492381_yphys.dat')
Nx = len(x)
Ny = len(y)

data = np.zeros((Nx, Ny))
for i in range(Nx):
    for j in range(Ny):
        data[i, j] = m.atan2(y[j], x[i])
X, Y = np.meshgrid(x, y)

Fx = np.gradient(np.unwrap(data,axis=0), axis=0)
Fy = np.gradient(np.unwrap(data,axis=1), axis=1)
c = np.gradient(Fy, axis=0)-np.gradient(Fx, axis=1)
c[abs(c)<0.01]=0
circult = np.zeros_like(data)
for it in range(1):
    for ix in range(1, Nx - 1):
        for iy in range(1, Ny - 1):
            circult[ix, iy] = np.unwrap([data[ix + 1, iy], 
                                         data[ix + 1, iy + 1], 
                                         data[ix, iy + 1], 
                                         data[ix - 1, iy + 1], 
                                         data[ix - 1, iy], 
                                         data[ix - 1,iy - 1], 
                                         data[ix, iy - 1], 
                                         data[ix + 1, iy]])[7] - data[ix + 1, iy]
circult[abs(circult)<1e-15] = 0

print(c[c!=0])
print(circult[circult!=0])

fig,ax = pl.subplots()
im = ax.pcolormesh(X, Y, data)
ax.plot(x[np.where(circult!=0)[0]], y[np.where(circult!=0)[1]],'ro', markersize=10)
ax.plot(x[np.where(c!=0)[0]], y[np.where(c!=0)[1]], 'bo')
fig.colorbar(im, ax=ax)
fig.show()
'''