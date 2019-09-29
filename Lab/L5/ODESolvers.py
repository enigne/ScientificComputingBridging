#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 13:45:00 2019

ODE solvers for lab 4

@author: chenggong
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve


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
    y = np.array([y0])
    for i in range(n-1):
        ynp1 = y[i] + dt*func(t[i], y[i])
        y = np.append(y, [ynp1])
    return t, y

def EulerBackward(func, t_span, y0, n):
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
    y = np.array([y0])
    for i in range(n-1):
        ynp1 = fsolve(lambda ynp1 : ynp1-y[i]-dt*func(t[i+1], ynp1), y[i]) 
        y = np.append(y, [ynp1])
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
    y = np.array([y0])
    for i in range(n-1):
        k1 = func(t[i], y[i])
        k2 = func(t[i+1], y[i]+dt*k1)
        ynp1 = y[i] + 0.5*dt*(k1+k2)
        y = np.append(y, [ynp1])
    return t, y


def testAccuracy(odeFunc, f_exact, y0, N=10, t_span=(0, 1)):
    """ Test accuracy for given function 
    Parameters
    ----------
    odeFunc : function, function handler of the right hand side of the ODE(s);
    f_exact : function, exact solution of odeFunc
    y0 : float, list, initial condition(s) of the ODE(s);
    N : number of time steps, dt = (t_span[1]-t_span[0])/N.
    t_span : list, t_span[0] is the initial time, t_span[1] is the final time;
        
    Returns
    -------
    err_fe : float, relative error in Euler Forward;
    err_he : float, relative error in Heun's method;
    err_rk : float, relative error in Runge-Kutta;
    """
    # Time steps
    t_eval = np.linspace(t_span[0], t_span[1], N)

    # Exact solution
    sol_ex = f_exact(t_eval, y0[0])

    # Euler Forward
    sol_fe = EulerForward(odeFunc, t_span, y0, N)
    err_abs_fe = sol_ex - sol_fe[1]
    err_fe = np.linalg.norm(err_abs_fe)/np.linalg.norm(sol_ex)

    # Heun's method
    sol_he = Heun(odeFunc, t_span, y0, N)
    err_abs_he = sol_ex - sol_he[1]
    err_he = np.linalg.norm(err_abs_he)/np.linalg.norm(sol_ex)

    # Solve for the ODE with R-K method
    h = t_eval[1] - t_eval[0]
    sol_rk = solve_ivp(odeFunc, t_span, y0, method='RK45', t_eval=t_eval, max_step=h)
    err_abs_rk = sol_ex - sol_rk.y[0]
    err_rk = np.linalg.norm(err_abs_rk)/np.linalg.norm(sol_ex)

    return err_fe, err_he, err_rk