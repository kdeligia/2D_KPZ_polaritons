#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 11:03:50 2019

@author: delis
"""

import numpy as np

def noise(shape):
    np.random.seed()
    mu = 0
    sigma = 1
    re = np.random.normal(mu, sigma, shape)
    im = np.random.normal(mu, sigma, shape)
    s = re + 1j * im
    return s

def confining(array, V_0, l):
    V = (V_0/l) * (np.exp(-(array - len(array)/2) ** 2 / (l**2)) + np.exp(-(array + len(array)/2) ** 2 / (l**2)))
    return V

def time(dt, t_units, N_steps, i1, i2, secondarystep):
    lengthindex = i2-i1
    length = lengthindex//secondarystep + 1
    t = np.zeros(length)
    for i in range(N_steps+1):
        if i>=i1 and i<=i2 and i%secondarystep==0:
            t[(i-i1)//secondarystep] = i*dt
    return t