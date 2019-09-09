#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  6 14:57:09 2019

Functions for lab 3: System of linear equations, diving problem

@author: chenggong
"""

import numpy as np
import scipy
import scipy.sparse   # SciPy Linear Algebra Library


def trampolinmatris(n, L):
    """ Generate matrix to diving board problem.
    
    Parameters
    ----------
    n : int, number of points 
    L : float, length of the board
        
    Returns
    -------
    scipy.sparse : matrix for the problem
    """
    # The stencil is [1, -4, 6, -4, 1]
    A = scipy.sparse.diags([1, -4, 6, -4, 1], [-2, -1, 0, 1, 2], shape=(n, n))
    A = A.tocsc()
    A[0, 0] = 7
    A[n-2, n-2] = 5
    A[n-2, n-1] = -2
    A[n-1, n-1] = 2
    A[n-1, n-2] = -4
    A[n-1, n-3] = 2
    
    hinv = 1.0*n/L
    A = A*(hinv**4)
    
    return A

def belastningsvektor(n, L, p, v, pos):
    """ Generate rhs to diving board problem.
    
    Parameters
    ----------
    n : int, number of points 
    L : float, length of the board
    p : int, number of people
    v : float, average weight of one person
    pos : float, position to put the load
    Returns
    -------
    numpy.array : rhs for the problem
    """
    x = np.linspace(0, L, n)
    weight = 20
    area = 0.2
    
    material = 2.7e-4
    f = -v*p*np.ones(n)
    a = -weight*np.ones(n)
    var = (x >= (pos-area)) & (x <= pos)
    b = (a + var*f)*material
    return b