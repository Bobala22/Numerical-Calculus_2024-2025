import numpy as np

A = [
    [4, 2, 3],
    [2, 7, 5.5],
    [6, 3, 12.5]
]

A_init = A

D = [2, 3, 4] 

B = [21.6, 33.6, 51.6]

#Decomposition
def dividing_matrix_LU(A, D):
    n = len(A)

    for row in range(n):
        for col in range(row + 1):
            s = 0
            for k in range(col):
                s += A[row][k] * A[k][col]
            A[row][col] = (A[row][col] - s) / D[col]
        
        for col in range(row + 1, n):
            s = 0
            for k in range(row):
                s += A[row][k] * A[k][col]
            A[row][col] = (A[row][col] - s) / A[row][row]

            
    return A

L = dividing_matrix_LU(A, D)

#Determinant
def determinant(A):
    n = len(A)
    det = 1;
    for i in range(n):
        det = det * A[i][i] * D[i]
    return det


#Ecuations solver
def ecuations_solver_L(A, B):
    n = len(A)
    Y = [] * n
    for i in range(n):
        s = 0
        for j in range(i):
            s += A[i][j] * Y[j]
        Y.append((B[i] - s) / A[i][i])
    return Y


def ecuations_solver_U(A, Y):
    n = len(A)
    X = [0] * n
    for i in range(n - 1, -1, -1):
        s = 0
        for j in range(i, n):
            if(i == j):
                s += D[j] * X[j]
            else:
                s += A[i][j] * X[j]
        X[i] = (Y[i] - s) / D[i]
    return X


def calculate_residual_norm(A_init, X, B):
    # Convert to NumPy arrays for easier matrix operations
    A_np = np.array(A_init)
    X_np = np.array(X)
    B_np = np.array(B)
    
    # Calculate A_init * X
    AX = A_np @ X_np
    
    # Calculate residual vector
    residual = AX - B_np
    
    # Calculate 2-norm
    norm = np.linalg.norm(residual)
    
    return norm

# Calculate and print the 2-norm
residual_norm = calculate_residual_norm(A_init, X, B)
print(f"||A_init * X - b||_2 = {residual_norm}")


#1------------------------------------
# for i, row in enumerate(A):  
#     for j in range(i + 1):  
#         print(A[i][j], end=" ")
#     print() 
# print()
# a_len = len(A)
# for i, row in enumerate(A):  
#     for j in range(i + 1, len(A)):
#         print(A[i][j], end=" ")
#     print() 

#2------------------------------------
# print(determinant(A))

#3------------------------------------
Y = ecuations_solver_L(A, B)
X = ecuations_solver_U(A, Y)
print(X)
