import numpy as np

N = 12 
K = 4 #block of matrix
M = int(N/K) #rows of each block
 
H = np.zeros((N//2,N)) #redun
print(H)
print('====================================')
for i in range(0,len(H),M):
    print('i is', i)
    for j in range(0,len(H[0]),M):
        print('j now is', j)
        for k in range(M):
            print('k then is', k)
            H[i+k,j+k] = (i/(M)+1)**(j/M)
         

print('H is')
print(H)

