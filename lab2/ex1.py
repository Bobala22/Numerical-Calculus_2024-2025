import copy

import numpy as np

eps = 1e-8

A = [[4.0, 2.0, 3.0],
    [2.0, 7.0, 5.5],
    [6.0, 3.0, 12.5]]

A_init = copy.deepcopy(A)

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
            if abs(D[col]) < eps:
                raise ZeroDivisionError(f"Division by zero encountered: D[{col}] = 0")
            A[row][col] = (A[row][col] - s) / D[col]
        
        for col in range(row + 1, n):
            s = 0
            for k in range(row):
                s += A[row][k] * A[k][col]
            if abs(A[row][row]) < eps:
                raise ZeroDivisionError(f"Division by zero encountered: A[{row}][{row}]] = 0")
            A[row][col] = (A[row][col] - s) / A[row][row]

            
    return A

L = dividing_matrix_LU(A, D)

#Determinant
def determinant(A):
    n = len(A)
    det = 1
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
    A_np = np.array(A_init)
    X_np = np.array(X)
    B_np = np.array(B)
    
    # @ = inmultire de matrici in numpy
    AX = A_np @ X_np
    
    residual = AX - B_np
    
    norm = np.linalg.norm(residual)
    
    return norm

#1------------------------------------
for i, row in enumerate(A):  
    for j in range(i + 1):  
        print(A[i][j], end=" ")
    print() 

print()

a_len = len(A)
for i, row in enumerate(A):  
    print(D[i], end=" ")
    for j in range(i + 1, len(A)):
        print(A[i][j], end=" ")
    print()
#2------------------------------------
print("Determinantul matricei A este: ", determinant(A))

#3------------------------------------
Y = ecuations_solver_L(A, B)
print("Y: ", Y)
X = ecuations_solver_U(A, Y)
print("X: ", X)

#4------------------------------------
print("Residual norm: ", calculate_residual_norm(A_init, X, B))
A_inv = np.linalg.inv(A)
def solve_ecuation_with_lib(A, B):
    A_inv = np.linalg.inv(A)
    X = A_inv @ B
    return X
X_lib = solve_ecuation_with_lib(A_init, B)
print("Solving ecuation with python lib", X_lib)
print("First norm with lib:", calculate_residual_norm(X, X_lib, np.array(1)))
