#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 13:45:00 2019

ODE solvers for lab 4

@author: chenggong
"""

import numpy as np


def EulerForward(func, t_span, y0, n):
    """ Explicit Euler method
    Parameters
    ----------
    func : function, function handler of the right hand side of the ODE(s);
    t_span : list, t_span[0] is the initial time, t_span[1] is the final time;
    y0 : float, list, initial condition(s) of the ODE(s);
    n : number of time steps, dt = (t_span[1]-t_span[0])/n.
        
    Returns
    -------
    t : list, the resulting time steps;
    y : list, the solutions of the ODE(s)
    """
    t = np.linspace(t_span[0], t_span[1], n)
    dt = t[1]-t[0]
    y = [y0]
    for i in range(n-1):
        ynp1 = y[i] + dt*func(t[i], y[i])
        y.append(ynp1)
    return t, y
    
    
def Heun(func, t_span, y0, n):
    """ Heun's method
    Parameters
    ----------
    func : function, function handler of the right hand side of the ODE(s);
    t_span : list, t_span[0] is the initial time, t_span[1] is the final time;
    y0 : float, list, initial condition(s) of the ODE(s);
    n : number of time steps, dt = (t_span[1]-t_span[0])/n.
        
    Returns
    -------
    t : list, the resulting time steps;
    y : list, the solutions of the ODE(s)
    """
    t = np.linspace(t_span[0], t_span[1], n)
    dt = t[1]-t[0]
    y = [y0]
    for i in range(n-1):
        k1 = func(t[i], y[i])
        k2 = func(t[i+1], y[i]+dt*k1)
        ynp1 = y[i] + 0.5*dt*(k1+k2)
        y.append(ynp1)
    return t, y