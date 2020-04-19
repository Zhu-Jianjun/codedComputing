#!/bin/env python

import numpy as np

class GaussianElimination:
    def forward_elimination(self,A,b):
        m,n = A.shape
        augmented_A = np.c_[A,b]   # Translates slice objects to concatenation along the second axis.
        j = 0
        for i in range(m - 1):     # A is a m*n matrix
        # to find the pivot
            pivot = augmented_A[i][j]
            
            #######################################################
            if pivot == 0:
                found = False
                for k in range(i+1,m):
                    if augmented_A[k][j] != 0:
                        temp = augmented_A[i].copy()
                        augmented_A[i] = augmented_A[k].copy()
                        augmented_A[k] = temp.copy()
                        found = True
                        break 
                if found == False:
                    raise Exception("The Matrix A is singular. There is no unique Solution")
                else:
                    pivot = augmented_A[i][j]
            #######################################################
            
            for k in range(i+1,m):
                target = augmented_A[k][j]
                multiplier = target / pivot
                augmented_A[k] = augmented_A[k] - multiplier * augmented_A[i]
            j += 1

        #new_A is a triangular matrix
        new_A = augmented_A[:,0:n]

        new_b = augmented_A[:,-1]
        return new_A, new_b
    

    def backward_substitution(self,new_A, new_b):
        m,n = new_A.shape
        x = [None]*m  # container
        for i in range(m-1,-1,-1):
        #temp = list(range(m))
        #for i in temp[::-1]:
            t = 0
            for k in range(m -1 , i, -1):
                t += new_A[i][k] * x[k]
            x[i] = ( new_b[i] - t ) / new_A[i][i]
        return x 
                 

    def solve(self,A,b):
        A = np.array(A)
        b = np.array(b)
        m,n = A.shape
        if m != n:
            raise Exception("Matrix A should be square.")
        if m != len(b):
            raise Exception("Number of unknown variables should be equal to the length of b")
        new_A, new_b = GaussianElimination.forward_elimination(self,A,b)
        x = GaussianElimination.backward_substitution(self,new_A, new_b)

        return x
   

#test for a specific linear system


if __name__ == "__main__":
   
    A = np.array([[25,5,1],[64,8,1],[144,12,1]])
    b = np.array([[106.8],[177.2],[279.2]])
    tem=GaussianElimination()
    new_A, new_b = tem.forward_elimination(A,b)
    x = tem.solve(A,b)
    print('new_A:')
    print(new_A)
    print('new_b:')
    print(new_b)
    print('x:')
    print(x)


    
# # test generally

# if __name__ == "__main__": 

    # n = 100
    # A = np.random.rand(n,n)
    # b = np.random.rand(n,1)
    # #A = np.random.randint(0, 100+1, (n, n))
    # #b = np.random.randint(0, 100+1, (n, 1))
    # tem=GaussianElimination()
    # x = tem.solve(A,b)
    # print(x)

