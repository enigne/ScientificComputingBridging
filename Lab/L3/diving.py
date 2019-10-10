#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import scipy
import scipy.sparse.linalg  
import matplotlib.pyplot as plt


# In[ ]:


get_ipython().run_line_magic('run', 'linearSystems.py')


# In[ ]:


# Setup
nx = 1500
L = 4
# number of people
p = 5
# average weight per person
v = 75
# position of the load
pos = L
# save figures
saveFigure = True


# In[ ]:


# Construct matrix
A = trampolinmatris(nx, L)
# right hand side
b = belastningsvektor(nx, L, p, v, pos)


# In[ ]:


# solve the linear systems


# In[ ]:


get_ipython().run_cell_magic('timeit', '-r 5 -n 1 -o', 'y = scipy.sparse.linalg.spsolve(A,b)')


# In[ ]:


# plot
y = scipy.sparse.linalg.spsolve(A,b)
fig, ax = plt.subplots(1, figsize=(8, 6))
x = np.linspace(0, L, nx)
ax.plot(x, y, linewidth=8, color='C1')

# create waves
xwaves = np.linspace(0, L, 50)
waves = -1+0.05*np.random.rand(50);  
ax.fill_between(xwaves, -2, waves)

# setups
ax.set_xlim([0, L])
ax.set_ylim([-2, 1])
ax.set_title('Diving board deflection')

if saveFigure:
    filename = 'DivingBoard.pdf'
    fig.savefig(filename, format='pdf', dpi=1000, bbox_inches='tight')


# In[ ]:


Asp = trampolinmatris(20, L)

fig, ax = plt.subplots(1, figsize=(8, 6))
ax.spy(Asp)
if saveFigure:
    filename = 'sparsityPattern.pdf'
    fig.savefig(filename, format='pdf', dpi=1000, bbox_inches='tight')

