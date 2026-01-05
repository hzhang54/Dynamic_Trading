import numpy as np
import matplotlib.pyplot as plt

'''
This script calculates the eigenvalues of M, checks their magnitudes, 
and visualizes them against the "Unit Circle" (the boundary of stability).

$$\mathbf{f}_t = \mathbf{f}_{t-1} + \Phi \mathbf{f}_{t-1} + \epsilon_t 
= (I + \Phi)\mathbf{f}_{t-1} + \epsilon_t$$
The "effective" transition matrix is $M = (I + \Phi)$. 
For the process to be stable (stationary), all eigenvalues of $M$ must have a magnitude 
strictly less than 1.
'''

def check_stability(Phi):
    """
    Tests the stability of the process f_t = (I + Phi)f_{t-1} + eps
    """
    n = Phi.shape[0]
    # The system matrix is M = I + Phi
    M = np.eye(n) + Phi
    
    # Calculate eigenvalues
    eigenvalues = np.linalg.eigvals(M)
    magnitudes = np.abs(eigenvalues)
    
    is_stable = np.all(magnitudes < 1)
    max_eig = np.max(magnitudes)
    
    print(f"Stability Check: {'PASSED' if is_stable else 'FAILED'}")
    print(f"Max Eigenvalue Magnitude: {max_eig:.4f}")
    
    return is_stable, eigenvalues

# --- Example Usage ---
# Create a stable 5D Phi matrix
Phi_stable = -0.1 * np.eye(5) + 0.02 * np.random.randn(5, 5)

is_stable, evals = check_stability(Phi_stable)

# if the process is stable, write the Matrix Phi_stable to a file for furture loading
if is_stable:
    np.save('Phi_stable.npy', Phi_stable)
    # print the matrix Phi_stable
    print("Phi_stable = ", Phi_stable)
else:
    print("The process is not stable.")

# if the process is stable, load the Matrix Phi_stable from a file
# if is_stable:
#    Phi_stable = np.load('Phi_stable.npy')

# --- Visualization ---
fig, ax = plt.subplots(figsize=(6, 6))
# Draw unit circle
theta = np.linspace(0, 2*np.pi, 100)
ax.plot(np.cos(theta), np.sin(theta), color='gray', linestyle='--', label='Unit Circle')

# Plot eigenvalues in the complex plane
ax.scatter(evals.real, evals.imag, color='red', marker='x', s=100, label='Eigenvalues of (I + Phi)')

ax.axhline(0, color='black', lw=1)
ax.axvline(0, color='black', lw=1)
ax.set_title("Stability Analysis (Complex Plane)")
ax.set_xlabel("Real Part")
ax.set_ylabel("Imaginary Part")
ax.legend()
ax.grid(True, alpha=0.3)
plt.show()