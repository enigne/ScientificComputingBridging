#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt


# # Lab 6 Monte Carlo Methods

# ## Introduction: Random numbers and statistics

# ### Normal distribution

# A nomral distribution can be generated by `numpy.eandom.normal`
# See details [here](https://docs.scipy.org/doc/numpy-1.15.0/reference/generated/numpy.random.normal.html).
# 
# The probability density function of the normal distribution follows a Gaussian function
# \begin{equation}
#     f(x|\mu,\sigma^2)=\frac{1}{\sqrt{2\pi\sigma^2}}e^{-\frac{(x-\mu)^2}{2\sigma^2}}
# \end{equation}

# In[ ]:


# mean
mu = 0
# Standard deviation
sigma = 1
# size of variables of the normal distribution, n can beint or tuple 
# depending on if x is a vector, or a matrix, etc.
n = 10000
# n = (100, 100)
# Normal distribution
x_normal = np.random.normal(mu, sigma, n)
# plot
fig, axs = plt.subplots(2, 1, figsize=(8, 8))
axs[0].plot(x_normal, 'ro')
axs[0].set_title(r'Normal Distribution with $\mu=$'+str(mu)+' and $\sigma=$'+str(sigma))

count, bins, ignored = axs[1].hist(x_normal, 100, density=True)
axs[1].plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) * 
            np.exp( - (bins - mu)**2 / (2 * sigma**2) ), 
            linewidth=2, color='r')
axs[1].set_title('Probability Density')

for i in range(2): axs[i].autoscale(enable=True, axis='both', tight=True)


# ### Uniform distribution

# Similarly, a uniform distribution can be constructed by `numpy.random.uniform`.
# See [here](https://docs.scipy.org/doc/numpy-1.15.0/reference/generated/numpy.random.uniform.html).

# In[ ]:


# the half-open interval [low, high)
low, high = 0, 1
# size of variables of the normal distribution, n can beint or tuple 
# depending on if x is a vector, or a matrix, etc.
n = 500
# uniform distribution
x_uni = np.random.uniform(low, high, n)

# plot
fig, axs = plt.subplots(2, 1, figsize=(8, 8))
axs[0].plot(x_uni, 'ro')
axs[0].set_title(r'Uniform Distribution in $[$'+str(low)+','+str(high)+r'$)$')

count, bins, ignored = axs[1].hist(x_uni, 100, density=True)
axs[1].plot(bins, np.ones_like(bins), linewidth=2, color='r')
axs[1].set_title('Probability Density')
for i in range(2): axs[i].autoscale(enable=True, axis='both', tight=True)


# ## A simulation: Throw dice

# The uniform distribution random function `numpy.random.uniform` generate real numbers. 
# In order to create only integers, one need to round the random value to the nearest integer with `numpy.floor` or `numpy.ceil`.

# Define the following function to simulate dice throw

# In[ ]:


def throwDice(N):
    '''
    To simulate throwing dice N times
    '''
    return np.floor(1 + 6*np.random.uniform(0, 1, size=N))


# In[ ]:


N = 1000
Nrepeat = 10000
r = [np.mean(throwDice(N)) for i in range(Nrepeat)] 

# plot
fig, ax = plt.subplots(1, 1, figsize=(8, 4))
ax.hist(r, 100, density=True)
ax.autoscale(enable=True, axis='both', tight=True)


# __NOTE__: There is a random integers generator in `numpy.random.randint` which create discrete uniform random integers from _low (inclusive)_ to __high (exclusive)__.
# See [here](https://docs.scipy.org/doc/numpy-1.15.0/reference/generated/numpy.random.randint.html).

# In[ ]:


# Range of the dice
# NOTE!! high value is not included, for a dice with six sides, high=7
low, high = 1, 7
# Number of throws
n = 1000
x_dice = np.random.randint(low, high, n)


# Then, we can simulate the same event with the following code

# In[ ]:


N = 1000
Nrepeat = 10000
r = [np.mean(np.random.randint(low, high, n)) for i in range(Nrepeat)] 

# plot
fig, ax = plt.subplots(1, 1, figsize=(8, 4))
ax.hist(r, 100, density=True)
ax.autoscale(enable=True, axis='both', tight=True)


# ## Application: Computing an integral with Monte Carlo

# An example use Monte Carlo method to compute the integration of multivariate normal distribution funciton.

# In[ ]:


def mcint(func, domain, N, M=30):
    """ Numerical integration using Monte Carlo method
    Parameters
    ----------
    func : function, function handler of the integrand;
    domain : numpy.ndarray, the domain of computation, 
            domain = array([[-5, 5],
                            [-5, 5],
                            [-5, 5]])
            The dimensions of the domain is given by domain.shape[0];
    N : integer, the number of points in each realization;
    M : integer, the number of repetitions used for error estimation,
        (Recommendation, M = 30+).
        Total number of points used is thus M*N
    Returns
    -------
    r.mean() : the integral value of func in the domain
    r.std() : the error in the result (the standard deviation)
    """
    # Get the dimensions
    dim = domain.shape[0]
    # volume of the domain
    V = abs(domain.T[0] - domain.T[1]).prod()
    
    r = np.zeros(M)
    for i in range(M):
        # generate uniform distributed random numbers within the domain
        x = np.random.uniform(domain.T[0], domain.T[1], (N, dim))
        r[i] = V * np.mean(func(x), axis=0)
        
    return r.mean(), r.std()

def fnorm(x):
    """ Normal distribution function in d-dimensions
    Parameters
    ----------
    x : numpy.ndarray, of shape (N, d), where d is the dimension and 
        N is the number of realizations
    Returns
    -------
    y : numpy.ndarray, of the shape (N, 1) 
    """ 
    d = x.shape[1]
    y = 1/((2*np.pi)**(d/2))*np.exp(-0.5*np.sum(x**2, axis=1))
    return y


# Take the domain in $[-4,4]\times[-4,4]$, compute the integral using funtion `fnorm`

# In[ ]:


# numbers of samples
N = 1000
M = 50
# domain
domain = np.array([[-4,4],[-4,4]])
# integrate
intF, err = mcint(fnorm, domain, N, M)
print('The result of the integral with N=', str(N), 'is', '{:.5f}'.format(intF),',')
print('with standard deviation', '{:.5f}'.format(err), 'for', str(M), 'realizations.')


# ### Check the order of accuracy $p$ for the Monte Carlo method.

# In[ ]:


# Change the dimension to see different results
dim = 3

# domain [-5, 5]^dim
domain = np.array([[-5, 5] for i in range(dim)])

# take an array of N
n = 8
NList = 500 * 2**np.array(range(n))
M = 50

# Save the error
errList = np.zeros_like(NList, dtype=float)
for i in range(n):
    intF, err = mcint(fnorm, domain, NList[i], M)
    errList[i] = err
    

# Plot
fig, ax = plt.subplots(1, 1, figsize=(8, 4))
ax.loglog(NList, errList)
ax.autoscale(enable=True, axis='both', tight=True)
ax.set_xlabel('N')
ax.set_ylabel('Error')

# Compute the order
a = np.polyfit(np.log(NList), np.log(errList),1)
p = np.round(a[0], 1)
print('Order of accuracy is N^p, with p=', p)


# ## Programming: Brownian motion

# In[ ]:


def brownian(x0, tEnd, dt):
    """
    Generate an instance of Brownian motion
    Parameters
    ----------
    x0 : float or numpy array (or something that can be converted to a numpy array
         using numpy.asarray(x0)).
         The initial condition(s) (i.e. position(s)) of the Brownian motion.
    tEnd : float, the final time.
    dt : float, the time step.
    Returns
    -------
    x: A numpy array of floats with shape `x0.shape + (n,)`.
    """
    x0 = np.asarray(x0)
    n = int(tEnd/dt)

    # For each element of x0, generate a sample of n numbers from a
    # normal distribution.
    r = np.random.normal(size=x0.shape + (n,), scale=(dt**0.5))

    # This computes the Brownian motion by forming the cumulative sum of
    # the random samples. 
    x = np.cumsum(r, axis=-1)

    # Add the initial condition.
    x += np.expand_dims(x0, axis=-1)
    
    return x


# In[ ]:


# Total time.
T = 10.0
# Number of steps.
N = 500
# Time step size
dt = T/N
# Initial values of x.
x = np.empty((2,N+1))
x[:, 0] = 0.0

# Brownian motion
x[:, 1:] = brownian(x[:,0], T, dt)

# Plot the 2D trajectory.
fig, ax = plt.subplots(1, 1, figsize=(8, 8))
ax.plot(x[0],x[1])
ax.plot(x[0,0],x[1,0], 'go')
ax.plot(x[0,-1], x[1,-1], 'ro')
ax.set_title('2D Brownian Motion')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.axis('equal')
ax.grid(True)

