import numpy as np
from scipy.linalg import sqrtm, inv

S = 50 # number of securities
K = 5 # number of factors

lmbda = 0.001 # tcost multiplier on the covariance matrix

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
Lambda = lmbda * Sigma
Lambda_bar = (1.0/rho_bar) * Lambda

# square root of Lambda_bar
Lambda_bar_sqrt = sqrtm(Lambda_bar)

# inverse of Lambda_bar
Lambda_bar_inv = inv(Lambda_bar)

# calculate rho_bar time gamma times square root of Lambda_bar times Sigma times square root of Lambda_bar
Axx_term_1 = rho_bar * gamma * Lambda_bar_sqrt @ Sigma @ Lambda_bar_sqrt

Axx_term_2 = 1./4. * (rho * rho * Lambda_bar * Lambda_bar + 2 * rho * gamma * Lambda_bar_sqrt @ Sigma @ Lambda_bar_sqrt + gamma * gamma * Lambda_bar_sqrt @ Sigma @ Lambda_bar_inv @ Sigma @ Lambda_bar_sqrt)

Axx = sqrtm(Axx_term_1 + Axx_term_2) - 1./2. * (rho * Lambda_bar + gamma *  Sigma)

# print the size of matrix Axx
print("Axx.shape = ", Axx.shape)

vec_Axf_term_1 = (np.eye(K) - Phi).T
vec_Axf_term_2 = (np.eye(S) - Axx @ inv(Lambda))

vec_Axf_pre = np.kron(vec_Axf_term_1, vec_Axf_term_2)
# get the size of vec_Axf_pre and save it in a variable
vec_Axf_pre_size = vec_Axf_pre.shape[0]

Axf_term_3 = (np.eye(S) - Axx @ inv(Lambda)) @ B
vec_Axf_term_3 = Axf_term_3.flatten(order='F')

vec_Axf = rho_bar * inv(np.eye(vec_Axf_pre_size) - rho_bar * vec_Axf_pre) @ vec_Axf_term_3

Axf = vec_Axf.reshape((S, K), order='F')

# print the size of matrix Axf
print("Axf.shape = ", Axf.shape)
