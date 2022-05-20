#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 18:26:31 2021

@author: delis
"""

import sympy as sp

k = sp.Symbol('k')
Kc = sp.Symbol('Kc')
Kd = sp.Symbol('Kd')
mu = sp.Symbol('mu')
Gamma = sp.Symbol('Gamma')

M = sp.Matrix([[(Kc - sp.I * Kd) * k**2 + mu - sp.I * Gamma, mu - sp.I * Gamma], [- mu - sp.I * Gamma, -(Kc + sp.I * Kd) * k**2 - mu - sp.I * Gamma]])

print(M.eigenvals())