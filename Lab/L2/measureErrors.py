#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 09:37:11 2019

Functions for lab 2

@revised date: 2020-03-27
@author: chenggong
@email: enigne@gmail.com
"""

import numpy as np


def absrelerror(corr=None, approx=None):
    """ 
    Illustrates the relative and absolute error.
    The program calculates the absolute and relative error. 
    For vectors and matrices, numpy.linalg.norm() is used.
    
    Parameters
    ----------
    corr : float, list, numpy.ndarray, optional
        The exact value(s)
    approx : float, list, numpy.ndarray, optional
        The approximated value(s)
        
    Returns
    -------
    None
    """
    
    print('*----------------------------------------------------------*')
    print('This program illustrates the absolute and relative error.')
    print('*----------------------------------------------------------*')

    # Check if the values are given, if not ask to input
    if corr is None:
        corr = float(input('Give the correct, exact number: '))
    if approx is None:
        approx = float(input('Give the approximated, calculated number: '))

    # be default 2-norm/Frobenius-norm is used
    abserror = np.linalg.norm(corr - approx)
    relerror = abserror/np.linalg.norm(corr)

    # Output
    print(f'Absolute error: {abserror}')
    print(f'Relative error: {relerror}')

def ForwardDiff(fx, x, h=0.001):
    """
    ForwardDiff(@fx, x, h);
    Use forward difference to approximatee the derivative of function fx
    in points x, and with step length h
    The function fx must be defined as a function handle with input
    parameter x and the derivative as output parameter
    
    Parameters
    ----------
    fx : function
        A function defined as fx(x)
    x : float, list, numpy.ndarray
        The point(s) of function fx to compute the derivatives
    h : float, optional
        The step size
    
    Returns
    -------
    float, list, numpy.ndarray: The numerical derivatives of fx at x with 
        the same size as x and the type if from fx()
    """
    return (fx(x+h) - fx(x))/h


def CentralDiff(fx, x, h=0.001):
    """
    CentralDiff(@fx, x, h);
    Use Central difference to approximatee the derivative of function fx
    in points x, and with step length h
    The function fx must be defined as a function handle with input
    parameter x and the derivative as output parameter

    Parameters
    ----------
    fx : function
        A function defined as fx(x)
    x : float, list, numpy.ndarray
        The point(s) of function fx to compute the derivatives
    h : float, optional
        The step size
    
    Returns
    -------
    float, list, numpy.ndarray: The numerical derivatives of fx at x with 
        the same size as x and the type if from fx()
    """
    return (fx(x+h) - fx(x-h))/h*0.5


def FivePointsDiff(fx, x, h=0.001):
    """
    FivePointsDiff(@fx, x, h);
    Use five points difference to approximatee the derivative of function fx
    in points x, and with step length h
    The function fx must be defined as a function handle with input
    parameter x and the derivative as output parameter
    
    Parameters
    ----------
    fx : function
        A function defined as fx(x)
    x : float, list, numpy.ndarray
        The point(s) of function fx to compute the derivatives
    h : float, optional
        The step size
    
    Returns
    -------
    float, list, numpy.ndarray: The numerical derivatives of fx at x with 
        the same size as x and the type if from fx()
    """
    return (-fx(x+2*h) + 8*fx(x+h) - 8*fx(x-h) + fx(x-2*h)) / (12.0*h)
