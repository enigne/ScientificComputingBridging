#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import numpy as np


# In[ ]:


saveFigure = False


# # SIR example

# ## Deterministic model

# In[ ]:


def SIR(t, y, b, d, beta, u, v):
    N = y[0]+y[1]+y[2]
    return [b*N - d*y[0] - beta*y[1]/N*y[0] - v*y[0],
           beta*y[1]/N*y[0] - u*y[1] - d*y[1],
           u*y[1] - d*y[2] + v*y[0]]


# In[ ]:


# Time interval for the simulation
t0 = 0
t1 = 120
t_span = (t0, t1)
t_eval = np.linspace(t_span[0], t_span[1], 10000)

# Initial conditions, 
N = 1000
I = 5
R = 0
S = N - I
y0 = [S, I , R]

# Parameters
b = 0.002/365
d = 0.0016/365
beta = 0.3
u = 1.0/7.0
v = 0.0

# Solve for SIR equation
sol = solve_ivp(lambda t,y: SIR(t, y, b, d, beta, u, v),
                t_span, y0, method='RK45',t_eval=t_eval)

# plot
fig, ax = plt.subplots(1, figsize=(6, 4.5))
# plot y1 and y2 together
ax.plot(sol.t.T,sol.y.T )
ax.set_ylabel('Number of predator and prey')
ax.set_xlabel('time')
ax.legend(['Susceptible', 'Infectious', 'Recover'])

if saveFigure:
    filename = 'SIR_deterministic.pdf'
    fig.savefig(filename, format='pdf', dpi=1000, bbox_inches='tight')


# ## Stochastic model

# In[ ]:


import numpy as np
# Plotting modules
import matplotlib.pyplot as plt

def sample_discrete(probs):
    """
    Randomly sample an index with probability given by probs.
    """
    # Generate random number
    q = np.random.rand()
    
    # Find index
    i = 0
    p_sum = 0.0
    while p_sum < q:
        p_sum += probs[i]
        i += 1
    return i - 1

# Function to draw time interval and choice of reaction
def gillespie_draw(params, propensity_func, population):
    """
    Draws a reaction and the time it took to do that reaction.
    """
    # Compute propensities
    props = propensity_func(params, population)
    
    # Sum of propensities
    props_sum = props.sum()
    
    # Compute time
    time = np.random.exponential(1.0 / props_sum)
    
    # Compute discrete probabilities of each reaction
    rxn_probs = props / props_sum
    
    # Draw reaction from this distribution
    rxn = sample_discrete(rxn_probs)
    
    return rxn, time


def gillespie_ssa(params, propensity_func, update, population_0, 
                  time_points):
    """
    Uses the Gillespie stochastic simulation algorithm to sample
    from proability distribution of particle counts over time.
    
    Parameters
    ----------
    params : arbitrary
        The set of parameters to be passed to propensity_func.
    propensity_func : function
        Function of the form f(params, population) that takes the current
        population of particle counts and return an array of propensities
        for each reaction.
    update : ndarray, shape (num_reactions, num_chemical_species)
        Entry i, j gives the change in particle counts of species j
        for chemical reaction i.
    population_0 : array_like, shape (num_chemical_species)
        Array of initial populations of all chemical species.
    time_points : array_like, shape (num_time_points,)
        Array of points in time for which to sample the probability
        distribution.
        
    Returns
    -------
    sample : ndarray, shape (num_time_points, num_chemical_species)
        Entry i, j is the count of chemical species j at time
        time_points[i].
    """

    # Initialize output
    pop_out = np.empty((len(time_points), update.shape[1]), dtype=np.int)

    # Initialize and perform simulation
    i_time = 1
    i = 0
    t = time_points[0]
    population = population_0.copy()
    pop_out[0,:] = population
    while i < len(time_points):
        while t < time_points[i_time]:
            # draw the event and time step
            event, dt = gillespie_draw(params, propensity_func, population)
                
            # Update the population
            population_previous = population.copy()
            population += update[event,:]
                
            # Increment time
            t += dt

        # Update the index
        i = np.searchsorted(time_points > t, True)
        
        # Update the population
        pop_out[i_time:min(i,len(time_points))] = population_previous
        
        # Increment index
        i_time = i
                           
    return pop_out

def simple_propensity(params, population):
    """
    Returns an array of propensities given a set of parameters
    and an array of populations.
    """
    # Unpack parameters
    beta, b, d, u, v = params
    
    # Unpack population
    S, I, R = population
    
    N = S + I + R
    
    return np.array([beta*I*S/N, 
                     u*I,
                     v*S,
                     d*S,
                     d*I,
                     d*R,
                     b*N])


# #### Solve SIR in stochastic method

# In[ ]:


# Column changes S, I, R
simple_update = np.array([[-1, 1, 0],
                          [0, -1, 1],
                          [-1, 0, 1],
                          [-1, 0, 0],
                          [0, -1, 0],
                          [0, 0, -1],
                          [1, 0, 0]], dtype=np.int)

# Specify parameters for calculation
params = np.array([0.3, 0.002/365, 0.0016/365, 1/7.0, 0])
time_points = np.linspace(0, 120, 500)
population_0 = np.array([995, 5, 0])
n_simulations = 100

# Seed random number generator for reproducibility
np.random.seed(42)

# Initialize output array
pops = np.empty((n_simulations, len(time_points), 3))

# Run the calculations
for i in range(n_simulations):
    pops[i,:,:] = gillespie_ssa(params, simple_propensity, simple_update,
                                population_0, time_points)


# In[ ]:


# Set up subplots
fig, ax = plt.subplots(1, 1, figsize=(6, 4.5))

for j in range(3):
    ax.plot(time_points, pops[4,:,j], '-',
               color='C'+str(j))
ax.set_ylabel('Number of predator and prey')
ax.set_xlabel('time')
ax.legend(['Susceptible', 'Infectious', 'Recover'])

if saveFigure:
    filename = 'SIR_stochastic1.pdf'
    fig.savefig(filename, format='pdf', dpi=1000, bbox_inches='tight')


# In[ ]:


# Set up subplots
fig, ax = plt.subplots(1, 1, figsize=(6, 4.5))
ax.plot(time_points, pops[:,:,0].mean(axis=0), lw=6)
ax.plot(time_points, pops[:,:,1].mean(axis=0), lw=6)
ax.plot(time_points, pops[:,:,2].mean(axis=0), lw=6)

ax.set_ylabel('Number of predator and prey')
ax.set_xlabel('time')
ax.legend(['Susceptible', 'Infectious', 'Recover'])

for j in range(3):
    for i in range(n_simulations):
        ax.plot(time_points, pops[i,:,j], '-', lw=0.3, alpha=0.2, 
                   color='C'+str(j))

if saveFigure:
    filename = 'SIR_stochasticAll.pdf'
    fig.savefig(filename, format='pdf', dpi=1000, bbox_inches='tight')


# # Simulate interest rate path by the CIR model 

# In[ ]:


import math
import numpy as np
import matplotlib.pyplot as plt

def cir(r0, K, theta, sigma, T=1., N=10,seed=777):
    np.random.seed(seed)
    dt = T/float(N)    
    rates = [r0]
    for i in range(N):
        dr = K*(theta-rates[-1])*dt +             sigma*math.sqrt(abs(rates[-1]))*np.random.normal()
        rates.append(rates[-1] + dr)
    return range(N+1), rates

fig, ax = plt.subplots(1, 1, figsize=(6, 4.5))

for i in range(30):
    x, y = cir(72, 0.001, 0.01, 0.012, 10., N=200, seed=100+i)
    ax.plot(x, y)
ax.set_ylabel('Assets price')
ax.set_xlabel('time')
ax.autoscale(enable=True, axis='both', tight=True)

if saveFigure:
    filename = 'CIR.pdf'
    fig.savefig(filename, format='pdf', dpi=1000, bbox_inches='tight')

