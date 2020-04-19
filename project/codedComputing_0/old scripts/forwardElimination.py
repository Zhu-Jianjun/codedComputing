#!/bin/env python

import numpy as np


def forwardElimination(A,b):
    A=np.asarray(A)
    m,n = A.shape
    augmented_A = np.c_[A,b]   # Translates slice objects to concatenation along the second axis.
    j = 0
    for i in range(m - 1):     # A is a m*n matrix
    # to find the pivot
        pivot = augmented_A[i][j]
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
        for k in range(i+1,m):
            target = augmented_A[k][j]
            multiplier = target / pivot
            augmented_A[k] = augmented_A[k] - multiplier * augmented_A[i]
        j += 1

     #new_A is a triangular matrix
##    new_A = augmented_A[:,0:n]
##    with open('new_A.txt','w') as f:
##        print(new_A, file=f)
##    new_b = augmented_A[:, n:]   # b is a n*n matrix
##    #new_b = augmented_A[:, -1]    # b is a vector
##    with open('new_b.txt','w') as f:
##        print(new_b, file=f)
##    return new_A, new_b
##    print(type(augmented_A))

    return augmented_A
    #return augmented_A[:,n:]  #take out b
    #return augmented_A[:,:n] #take out A
    
    #A = augmented_A[:,:n]
    #return A



##A=[[68, 68, 67, 72],[54, 28,  4, 85]]
##
##b=[[87],[10]]
##
##print(forwardElimination(A,b))


