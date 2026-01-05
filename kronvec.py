import numpy as np

# Define dimensions
m, n, p, q = 2, 3, 2, 2

# Create random matrices
A = np.random.rand(m, n)
X = np.random.rand(n, p)
B = np.random.rand(p, q)

# Left Side: vec(AXB)
AXB = A @ X @ B
left_side = AXB.flatten(order='F')

AXB_recovered = left_side.reshape((m, q), order='F')

print("axb recovered?")
# Verify
print(np.allclose(AXB, AXB_recovered)) # Should return True

# Right Side: (B.T ⊗ A) vec(X)
vec_X = X.flatten(order='F')
K = np.kron(B.T, A)
right_side = K @ vec_X

# Check if they are equal
print(f"Are they equal? {np.allclose(left_side, right_side)}")

# calculate transpose of A 
A_T = A.T

# calculate identity matrix minus A then transpose it
I_minus_A_T = (np.eye(m) - A_T).T
