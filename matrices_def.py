import numpy as np

S = 50 # number of securities
K = 5 # number of factors

lambda = 0.001 # tcost multiplier on the covariance matrix

rho = 0.05 # discount factor for the dynamic programming
rho_bar = 1 - rho

gamma = 0.9 # risk aversion parameter

Phi = np.load('Phi_stable.npy')
# invert the sign of the matrix Phi
Phi = -Phi
# load the matrix B
B = np.load('B.npy')
# load the matrix Sigma
Sigma = np.load('Sigma.npy')
# print the matrix Phi
# print("Phi = ", Phi)

# check if Phi is K x K matrix
if Phi.shape[0] != K or Phi.shape[1] != K:
    raise ValueError("Phi is not a K x K matrix")

# check if B is S x K matrix
if B.shape[0] != S or B.shape[1] != K:
    raise ValueError("B is not a S x K matrix")

# check if Sigma is S x S matrix
if Sigma.shape[0] != S or Sigma.shape[1] != S:
    raise ValueError("Sigma is not a S x S matrix")

# set the tcost matrix Lambda to be lambda times the Sigma matrix
Lambda = lambda * Sigma
Lambda_bar = (1.0/rho_bar) * Lambda

