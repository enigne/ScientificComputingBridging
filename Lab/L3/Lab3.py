#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import scipy
import scipy.linalg   # SciPy Linear Algebra Library
import timeit
import matplotlib.pyplot as plt


# In[ ]:


saveFigure = True


# # Solve linear system of equations

# Use the ipython build-in magic command `%timeit`

# Options of [%timeit](https://ipython.readthedocs.io/en/stable/interactive/magics.html#magic-timeit):
# 
# 
# <!-- * `-c`: use time.clock to measure the time, which is the default on Windows and measures wall time. On Unix, resource.getrusage is used instead and returns the CPU user time. -->
# * `-n <N>`: execute the given statement N times in a loop. If N is not provided, N is determined so as to get sufficient accuracy.
# * `-o`: return a `TimeitResult` that can be stored in a variable to inspect the result in more details.
# * `-p <P>`: use a precision of P digits to display the timing result. Default: 3
# * `-q`: Quiet, do not print result.
# * `-r <R>`: number of repeats R, each consisting of N loops, and take the best result. Default: 7
# * `-t`: use time.time to measure the time, which is the default on Unix. This function measures wall time.
# 

# ## LU factorization

# This program measure the CPU-time when solving a system of equations with `numpy.linalg.solve`, see https://docs.scipy.org/doc/numpy/reference/generated/numpy.linalg.solve.html
# 
# The solutions are computed with `numpy.linalg.solve` using LAPACK routine `_gesv`, which is designed for solving linear systems with multiple right hand sides.
# The LU decomposition with partial pivoting and row interchanges is used, see https://software.intel.com/en-us/node/520973.

# In[ ]:


# Set the range of the problem sizes N
# NList = [ 500, 1000, 2000, 4000, 8000]

NList = np.arange(1,17)*500


# In[ ]:


# Loop for all N
invTime = []

for i in NList:
    A = scipy.random.rand(i,i)
    b = scipy.random.rand(i,1)

    invTimeTemp = get_ipython().run_line_magic('timeit', '-r 5 -n 1 -o x = scipy.linalg.solve(A, b)')
    invTime.append(invTimeTemp)


# In[ ]:


# Plot
fig, ax = plt.subplots(1)
xp = []
yp = []
for (i, t) in zip(xbar,invTime):
    xp.append(i)
    yp.append(t.average)
    
p3 = np.poly1d(np.polyfit(xp, yp, 3))

xbar = np.arange(len(NList))
ax.plot(xbar, p3(xbar),color='C2')

for (i, t) in zip(xbar,invTime):
    ax.bar(i, t.average,width=0.5,color='C0')
    ax.errorbar(i, t.average, yerr=t.stdev,color='C1',linewidth=2)

ax.set_xlabel(r'$N$')
ax.set_ylabel('CPU time(s)')
ax.set_xticks(xbar)
ax.set_xticklabels(NList)
ax.autoscale(enable=True, axis='y', tight=True)
ax.tick_params(axis ='x', rotation = -45) 
ax.legend(['Cubic fit','Average time','Standard Deviations'])

if saveFigure:
    filename = 'Multiple_LU.pdf'
    fig.savefig(filename, format='pdf', dpi=1000, bbox_inches='tight')


# ## Direct inverse vs. LU factorization

# In[ ]:


# Set the problem size
N = 2000


# In[ ]:


# Create the system and right-hand-side for comparison
A = scipy.random.rand(N, N)
b = scipy.random.rand(N, 1)


# #### Invert $A$ matrix first

# In[ ]:


get_ipython().run_cell_magic('timeit', '-r 5 -n 1 -o', 'A_inv = scipy.linalg.inv(A)\nx_inv = scipy.dot(A_inv,b)')


# In[ ]:


# save the previous cell output
time_inv = _


# #### Use LU factorization

# In[ ]:


time_LU = get_ipython().run_line_magic('timeit', '-r 5 -n 1 -o x_LU = scipy.linalg.solve(A, b)')


# In[ ]:


print('Execution time for solving a equation system with ' + str(N) + ' equations: ')
print('    with x = scipy.linalg.solve(A,b): ' + '{:.3f}'.format(time_LU.average) + '±' + '{:.3f}'.format(time_LU.stdev) + 's')
print('    with matrix invers, x=inv(A)*b:   ' + '{:.3f}'.format(time_inv.average) + '±' + '{:.3f}'.format(time_inv.stdev) + 's')
print('Matrix invers is a factor ' + '{:.3f}'.format(time_inv.average/time_LU.average) + ' slower')


# ## Tests LU factorization for multiple right-hand-sdie

# In[ ]:


# Size of matrix
N_LU = 3000
repeat = 10


# ### Solve for different rhs each time

# In[ ]:


# Solve once
A = scipy.random.rand(N_LU, N_LU)
b = scipy.random.rand(N_LU, 1)
timeLUonce = get_ipython().run_line_magic('timeit', '-r 5 -n 1 -o x_LU = scipy.linalg.solve(A, b)')


# Solve `repeat` times with different right hand sides.

# In[ ]:


get_ipython().run_cell_magic('timeit', '-r 5 -n 1 -o ', 'for i in range(repeat):\n    b = scipy.random.rand(N_LU, 1)\n    x_solve = scipy.linalg.solve(A, b)')


# In[ ]:


timeLURepeat = _


# ### Solve for different rhs with L and U saved

# In[ ]:


get_ipython().run_cell_magic('timeit', '-r 5 -n 1 -o ', 'P, L, U = scipy.linalg.lu(A)\nfor i in range(repeat):\n    b = scipy.random.rand(N_LU, 1)\n    dtemp = scipy.dot(P,b)\n    d = scipy.linalg.solve_triangular(L, dtemp)\n    x_LU = scipy.linalg.solve_triangular(U, d)')


# In[ ]:


timeLUSaved = _


# ### Solve all at once

# In[ ]:


get_ipython().run_cell_magic('timeit', '-r 5 -n 1 -o ', 'b = scipy.random.rand(N_LU, repeat)\nx_LUo = scipy.linalg.solve(A, b)')


# In[ ]:


timeLURepeatOnce = _


# ### Invert $A$ matrix first

# In[ ]:


get_ipython().run_cell_magic('timeit', '-r 5 -n 1 -o', 'A_inv = scipy.linalg.inv(A)\nx_inv = scipy.dot(A_inv,b)')


# In[ ]:


# save the previous cell output
timeInv = _


# ### Visualize the results

# In[ ]:


# Plot
ind = np.arange(5)
timeList = [timeInv,timeLUonce,timeLURepeat,timeLUSaved,timeLURepeatOnce]
fig, ax = plt.subplots(1, figsize=(10, 6))

for (i,t) in zip(ind, timeList):
    ax.bar(i, t.average, width=0.5, color='C0')
    ax.errorbar(i, t.average, t.stdev, color='C1')

ax.set_title('Execution time for solving one system with ' + str(N_LU) + ' equations')
ax.set_ylabel('CPU time(s)')
ax.set_xticks(ind)
ax.set_xticklabels(['Invert one system', 
                    'Solve one system', 
                    'Solve ' + str(repeat) + ' systems', 
                    'LU ' + str(repeat) + ' systems', 
                    'Solve ' + str(repeat) + ' systems,\n with multiple rhs'])
ax.autoscale(enable=True, axis='y', tight=True)


ax.legend(['Average time','Standard Deviations'])

if saveFigure:
    filename = 'LU_test.pdf'
    fig.savefig(filename, format='pdf', dpi=1000, bbox_inches='tight')

