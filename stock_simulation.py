'''
1. Creating a Realistic Loading Matrix $B$

A common structure is to have: 
* Factor 1: A "Market" factor (all stocks have positive loadings).
* Factors 2-5: "Sector" or "Style" factors (e.g., Growth vs. Value, Tech vs. Energy).

We will create $B$ ($50 \times 5$) such that different blocks of stocks have higher weights 
on different factors.

2. Python Simulation 

This code simulates the factor process $\mathbf{f}_t$ first, then uses $B$ to generate 
stock returns and prices.
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 1. Setup Dimensions
n_factors = 5
n_stocks = 50
T = 252  # One trading year
dt = 1

# 2. Create Realistic Loading Matrix B (50 x 5)
B = np.zeros((n_stocks, n_factors))
B[:, 0] = np.random.uniform(0.5, 1.5, n_stocks) # All stocks load on Market factor

# Assign blocks of stocks to specific factors (Sectors)
for i in range(1, n_factors):
    start_idx = (i-1) * 12
    end_idx = i * 12 if i < n_factors-1 else n_stocks
    B[start_idx:end_idx, i] = np.random.uniform(0.8, 2.0, end_idx - start_idx)

# write the matrix B to a file
np.save('B.npy', B)
# 3. Simulate Factor Process (Mean Reverting)
Phi = np.load('Phi_stable.npy') # -0.05 * np.eye(n_factors) # Slow mean reversion
f = np.zeros((T, n_factors))
f_sigma = 0.05 * np.eye(n_factors)
for t in range(1, T):
    f[t] = f[t-1] + Phi @ f[t-1] + np.random.multivariate_normal(np.zeros(n_factors), f_sigma)

# 4. Generate Stock Returns: r_{t+1} = B f_t + epsilon, and calculate the sample 
# covariance matrix of residuals epsilon_sigma
epsilon_sigma = 0.02 # Idiosyncratic risk
residuals = np.random.normal(0, epsilon_sigma, (T, n_stocks))
returns = f @ B.T + residuals

# calculate the sample covariance matrix of residuals
Sigma = np.cov(residuals, rowvar=False)
np.save('Sigma.npy', Sigma)
# print the matrix Sigma
print("Sigma = ", Sigma)
# 5. Convert Returns to Price Trajectories
# Price_t = Price_0 * exp(cumsum(r))
initial_prices = np.random.uniform(50, 150, n_stocks)
price_trajectories = initial_prices * np.exp(np.cumsum(returns, axis=0))

# 6. Plotting
#plt.figure(figsize=(12, 6))
#plt.plot(price_trajectories, alpha=0.7)
#plt.title(f"Simulated Trajectories for {n_stocks} Stocks (Factor Model)")
#plt.xlabel("Days")
#plt.ylabel("Price ($)")
#plt.grid(True, alpha=0.3)
#plt.show()
# save the price trajectories to a file
np.save('price_trajectories.npy', price_trajectories)
# save the returns to a file
np.save('returns.npy', returns)
# plot the store return series
plt.figure(figsize=(12, 6))
plt.plot(returns, alpha=0.7)
plt.title(f"Simulated Returns for {n_stocks} Stocks (Factor Model)")
plt.xlabel("Days")
plt.ylabel("Return ($)")
plt.grid(True, alpha=0.3)
plt.show()