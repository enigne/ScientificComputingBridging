#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import ipywidgets as widgets
import matplotlib.pyplot as plt


# In[ ]:


get_ipython().run_line_magic('run', './measureErrors.py')
saveFigure = True


# # Lab Exercise 2 for SCB

# ## Errors

# ### Ex 1-3

# * Try to read `absrelerror()` in `measureErrors.py` and use it for the exercises

# ### Ex. 4 Round-off errors

# In[ ]:


# Generate a random nxn matrix and compute A^{-1}*A which should be I analytically
def testErrA(n = 10):
    A = np.random.rand(n,n)
    Icomp = np.matmul(np.linalg.inv(A),A)
    Iexact = np.eye(n)
    absrelerror(Iexact, Icomp)


# #### Random matrix $A$ with size $n=10$

# In[ ]:


testErrA()


# #### $n=100$

# In[ ]:


testErrA(100)


# #### $n=1000$

# In[ ]:


testErrA(1000)


# <span style="color:red">**Note**:</span> The execution time changes with the size of $n$ almost linearly, but for $n=10000$, it will take much longer time.

# ### Ex. 5 Discretization Errors

# Program that illustrate the concept discretization.
# 
# Replacing continuous with discrete, i.e. represent a continuous function on a interval with a finite number of points.
# 
# The density, the number of points is determined by the choice of the discretization parameter $h$.

# #### The step size
# 
# **TRY**  to change $h$ and see what will happen.  
# 
# **NOTE**: $h$ should not be too large or too small. A good range is in $[10^{-5},1]$.

# In[ ]:


h = 0.1


# #### Discretize and compute the numerical derivatives.
# 
# Here, the derivative `f'(x)` is computed in a finite number of points on a interval. 
# 

# In[ ]:


# The exact solution
N = 400
l = 0
u = 2
x = np.linspace(l, u, N)
f_exa = np.exp(x)

# check if h is too large or too small
if h > 1 or h < 1e-5:
    h = 0.5

# compute the numerical derivatives
xh = np.linspace(l, u, int(abs(u-l)/h))
fprimF = ForwardDiff(np.exp, xh, h);


# #### Use `matplotlib` to visuallize the results. 
# 
# Try to check on [https://matplotlib.org/](https://matplotlib.org/) for mor features, it is really powerful!

# In[ ]:


# Plot
fig, ax = plt.subplots(1)
ax.plot(x, f_exa, color='blue')
ax.plot(xh, fprimF, 'ro', clip_on=False)
ax.set_xlim([0,2])
ax.set_ylim([1,max(fprimF)])
ax.set_xlabel(r'$x$')
ax.set_ylabel('Derivatives')
ax.set_title('Discretization Errors')
ax.legend(['Exact Derivatives','Calculated Derivatives'])

if saveFigure:
    filename = 'DiscretizationError_h' + str(h) + '.pdf'
    fig.savefig(filename, format='pdf', dpi=1000, bbox_inches='tight')


# ## Computer Arithmetic

# Machine limits for floating point types use `np.finfo(float)`

# In[ ]:


print('machhine epsilon in python is: ' + str(np.finfo(float).eps))


# The overflow in python is shown by `np.finfo(float).max` and the underflow by `np.finfo(float).tiny`

# In[ ]:


print('The largest real number in python is: ' + str(np.finfo(float).max))
print('The smallest positive real number in python is: ' + str(np.finfo(float).tiny))


# Other attributes of `finfo` can be found [here](https://docs.scipy.org/doc/numpy-1.15.1/reference/generated/numpy.finfo.html)

# ## Computation of the derivative

# The function $f(x) = e^x$ in $x=1$ is used as test function.
# 
# * forward difference: $\displaystyle{f'(x)\approx\frac{f(x+h)-f(x)}{h}}$
# * central difference: $\displaystyle{f'(x)\approx\frac{f(x+h)-f(x-h)}{2h}}$
# * five points difference: $\displaystyles are $

# #### Determine the range you would like to experiment with

# In[ ]:


# choose h from 0.1 to 10^-t, t>=2
t = 15
hx = 10**np.linspace(-1,-t, 30)


# #### Compute the numerical derivatives using the three different schemes

# In[ ]:


# The exact derivative at x=1
x0 = 1
fprimExact = np.exp(1)

# Numeric derivative using the three methods
fprimF = ForwardDiff(np.exp, x0, hx);
fprimC = CentralDiff(np.exp, x0, hx);
fprim5 = FivePointsDiff(np.exp, x0, hx);

# Relative error
felF = abs(fprimExact - fprimF)/abs(fprimExact);
felC = abs(fprimExact - fprimC)/abs(fprimExact);
fel5 = abs(fprimExact - fprim5)/abs(fprimExact);


# #### Visualize the results

# In[ ]:


# Plot
fig, ax = plt.subplots(1)
ax.loglog(hx, felF)
ax.loglog(hx, felC)
ax.loglog(hx, fel5)
ax.autoscale(enable=True, axis='x', tight=True)
ax.set_xlabel(r'Step length $h$')
ax.set_ylabel('Relative error')
ax.legend(['Forward difference','Central difference', 'Five points difference'])

if saveFigure:
    filename = 'NumericalDerivative.pdf'
    fig.savefig(filename, format='pdf', dpi=1000, bbox_inches='tight')

