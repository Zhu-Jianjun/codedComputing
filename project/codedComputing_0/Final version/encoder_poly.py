import numpy as np

N = 12 #mtrix size
G = 2  #corresponding to number of rows in each block, maximize the number of workers used when G=2


################## prepare data ######################

W = N//G #corresponding to number of blocks of matrix1
a = list(range(1, W+1))
A=[]
c=0
while c <= len(a)-1:
    A+=a
    a.append(a.pop(0))
    c+=1
A = np.asarray(A).reshape(len(A)//len(a), len(a))

Q = W + W #origin(W) + redundancy(Q-W), corresponding to number of workers totally
X=list(range(1,Q+1)) 
poly=[]
for x in X:
    for i in range(len(A)):
        t=1
        for j in range(1,len(A[0])):
            t*=(x-A[i][j])/(A[i][0]-A[i][j])
        poly.append(int(t))  #poly.append(t)       
poly=np.asarray(poly).reshape(len(poly)//len(a), len(a))
print(poly)

#################################################################

imtrx = np.identity(G, dtype=int)
encoder = np.zeros((len(poly)*G, len(poly[0])*G))
T = []
for i in range(len(poly)):
    for j in range(len(poly[0])):
        T.append(np.dot(poly[i][j], imtrx))

c=0
for i in range(0,len(encoder), G):
    for j in range(0,len(encoder[0]), G):
        encoder[i:i+G, j:j+G] = T[c]
        c+=1
#print(encoder)
print('encoder shape:', np.asarray(encoder).shape)

#################################################################

mtrx1 = np.random.randint(2, size=(N, N))
mtrx1Coded = np.dot(encoder, mtrx1)
print('mtrx1Coded shape:', np.asarray(mtrx1Coded).shape)

mtrx2 = np.random.randint(2, size=(N, N))
print('mtrx2 shape:', np.asarray(mtrx2).shape)
        


nodes = 12
rows = []
n = len(mtrx1Coded) // nodes
r = len(mtrx1Coded) % nodes
b, e = 0, n + min(1, r)
for i in range(nodes):
    rows.append(mtrx1Coded[b:e])
    r = max(0, r - 1)
    b, e = e, e + n + min(1, r)
print(len(rows))
print(rows[0])
print(mtrx1Coded[0:2])