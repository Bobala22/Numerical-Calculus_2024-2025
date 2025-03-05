
A = [
    [1,2,3],
    [4,5,6],
    [7,8,9]]



for i, row in enumerate(A):  
    for j in range(i + 1):  
        print(A[i][j], end=" ")
    print() 

a_len = len(A)
for i, row in enumerate(A):  
    for j in range(i, len(A)):
        print(A[i][j], end=" ")
    print() 
