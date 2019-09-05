#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 09:37:11 2019

Functions for lab 2

@author: chenggong
"""

import numpy as np


def absrelerror(corr=None, approx=None):
    """
    absrelerror (rewritten from Matlab script)
    Script that illustrates the relative and absolute error
    To tun it, call  absrelfel(corr, approx)
    The prograM calculates the absolute and relative error. 
    """
    print('*------------------------------------------------------------*')
    print('This program illustrates the absolute and relative error.')
    print('*------------------------------------------------------------*')

    if corr is None:
        corr = float(input('Give the correct, exact number: '))
    if approx is None:
        approx = float(input('Give the approximative, calculated number: '))

    abserror = np.linalg.norm(corr - approx)
    relerror = abserror/np.linalg.norm(corr)

    print('Absolute error: ' + str(abserror))
    print('Relative error: ' + str(relerror) + ', or in percent: ' +
          str(relerror*100))


def ForwardDiff(fx, x, h=0.001):
    """
    ForwardDiff(@fx, x, h);
    Use forward difference to approximatee the derivative of function fx
    in points x, and with step length h
    The function fx must be defined as a function handle with input
    parameter x and the derivative as output parameter
    """
    return (fx(x+h) - fx(x))/h


def CentralDiff(fx, x, h=0.001):
    """
    CentralDiff(@fx, x, h);
    Use Central difference to approximatee the derivative of function fx
    in points x, and with step length h
    The function fx must be defined as a function handle with input
    parameter x and the derivative as output parameter
    """
    return (fx(x+h) - fx(x-h))/h*0.5


def FivePointsDiff(fx, x, h=0.001):
    """
    FivePointsDiff(@fx, x, h);
    Use five points difference to approximatee the derivative of function fx
    in points x, and with step length h
    The function fx must be defined as a function handle with input
    parameter x and the derivative as output parameter
    """
    return (-fx(x+2*h) + 8*fx(x+h) - 8*fx(x-h) + fx(x-2*h)) / (12.0*h)
