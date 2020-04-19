import numpy as np

N = 8
K = 2
M = 2  #identity matrix dimention

#imtrx = [[1 if i==j else 0 for i in range(N)] for j in range(N)]
imtrx = np.identity(N, dtype=int)

x = np.array(list(range(1, K+1)))  
vander_matrix = np.vander(x, N // M, increasing=True)
vander_matrix = vander_matrix.tolist()

vander_matrix1 = np.vander(x, N // M, increasing=True)
vander_matrix1 = vander_matrix1.tolist()

t = [i for i in range(N) if i%2!=0]
t1 = [i for i in range(N) if i%2==0]

for i in range(len(vander_matrix)):
    for j in t:
        vander_matrix[i].insert(j,0)

        
for i in range(len(vander_matrix1)):
    for j in t1:
        vander_matrix1[i].insert(j,0)


##temp = [i for i in range(len(vander_matrix)*2) if i%2!=0]
##for j in temp:
##    vander_matrix.insert(j,vander_matrix1[0])
##print(vander_matrix)

temp = []    
for i in range(len(vander_matrix)):
    temp.append(vander_matrix[i])
    temp.append(vander_matrix1[i])

redun = np.vstack((imtrx,temp))
print(redun)
