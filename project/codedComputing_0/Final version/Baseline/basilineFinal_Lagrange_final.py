
from mpi4py import MPI
import time
import sys
import numpy as np


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
workers = comm.Get_size() - 1

stat = MPI.Status()
barrier = True

N = 1200 #mtrix size
G = 300  #identity matrix size, corresponding to number of rows in each block, maximize the number of workers used when G=2

#N = int(sys.argv[1])
#G = int(sys.argv[2])



def init_matrix():
    
    print('\n')
    global W
    W = N//G #corresponding to number of blocks of mtrx1
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
    global poly
    poly=[]
    for x in X:
        for i in range(len(A)):
            t=1
            for j in range(1,len(A[0])):
                t*=(x-A[i][j])/(A[i][0]-A[i][j])
            poly.append(int(t))  #poly.append(t)       
    poly=np.asarray(poly).reshape(len(poly)//len(a), len(a))
    #print(poly)
    print('poly shape:', np.asarray(poly).shape)
    
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
    
    global mtrx1
    global mtrx1Coded
    mtrx1 = np.random.randint(2, size=(N, N))
    mtrx1Coded = np.dot(encoder, mtrx1)
    print('mtrx1Coded shape:', np.asarray(mtrx1Coded).shape)

    global mtrx2
    mtrx2 = np.random.randint(2, size=(N, N))
    print('mtrx2 shape:', np.asarray(mtrx2).shape)
    print('\n')    


def split_matrix(matrixCoded, nodes):
    
    rows = []
    n = len(matrixCoded) // nodes
    r = len(matrixCoded) % nodes
    b, e = 0, n + min(1, r)
    for i in range(nodes):
        rows.append(matrixCoded[b:e])
        r = max(0, r - 1)
        b, e = e, e + n + min(1, r)    
    return rows 



def master():

    init_matrix()
    rows = split_matrix(mtrx1Coded, workers)
    
    mtrx3 = [] #data needed to send
    for i in range(len(rows)):
        data = np.vstack((rows[i], mtrx2))
        temp = [len(mtrx2)]*(len(data))
        data1 = np.c_[data,temp] #provide the length in the last column
        mtrx3.append(data1)    
        
    for pid in range(workers, 0, -1):
        comm.send(mtrx3[pid-1], dest=pid, tag=pid)
    
    if barrier:
        comm.Barrier()        
    
    '''receive data from workers'''
    
    redun = poly[W:] #prepare poly-coefficients for decoding
    print('redun coefficients is')
    print(redun)
    print('\n')

    c = 0
    mtrx4 = [] #group A(redundency-based results)
    mtrx5 = [] #group B
    mtrx6 = [] #store the rank who sent back the redundency-based results
    mtrx7 = [] #store the rank who just send back the results
    
    while c < workers:
        
        start_time = time.time()
        
        row = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=stat)
        source = stat.Get_source()
        print('chenking...')
        
        if c < W:  
            if source > W:
                mtrx6.append(source)
                mtrx4.append(row)
                print('Back from rank {}'.format(source))
            
            #if source <= W:
            else:
                mtrx7.append(source)
                mtrx5.append(row)
                print('Back from rank {}'.format(source))
             
        if c == W-1:
            print('mtrx4(redundency-based results):', np.asarray(mtrx4).shape)
            print('mtrx6(ranks who have contributions to the redundency-based results):', mtrx6)
            print('mtrx5:', np.asarray(mtrx5).shape)
            print('mtrx7(ranks who have contributions to the results):', mtrx7)    

            print('\n')
            print('starting decoding...')
            print('\n')
            
            #start_time = time.time()
            
            '''remove all the terms that are not necessary to decode and then update'''
            for i in range(len(mtrx4)):        
                for j in range(len(mtrx5)):    
                    alpha = redun[mtrx6[i]-W-1][mtrx7[j]-1]  
                    #print('alpha is', alpha)           
                    mtrx4[i] = np.asarray(mtrx4[i]) - alpha*(np.asarray(mtrx5[j]))    

            '''forward elimination'''
            mtrx8 = [] #to store the redun coefficients that has been back
            for i in range(len(mtrx6)):
                mtrx8.append(redun[mtrx6[i]-W-1])   
            #print('mtrx8 is')
            #print(np.asarray(mtrx8))

            mtrx9 = list(range(1,workers+1))
            for x in list(range(workers,workers-W, -1)): #need to remove all the redundency-based ranks
                if x in mtrx9:
                    mtrx9.remove(x)
            for y in mtrx7:
                if y in mtrx9:
                    mtrx9.remove(y)
            #print('rank that did not send back partial results(mtrx9):', mtrx9)

            mtrx10 = [[temp[k-1] for k in mtrx9] for temp in mtrx8]
            #print('mtrx10 is')
            #print(np.asarray(mtrx10))

            j = 0
            for i in range(len(mtrx4)-1):
                pivot = mtrx10[i][j]  
                #print('pivot is', pivot)
                for k in range(i+1, len(mtrx4)):
                    target = mtrx10[k][j]
                    #print('target is', target)
                    beta = target/pivot
                    #print('beta is', beta)
                    mtrx4[k] = np.asarray(mtrx4[k]) - beta*np.asarray(mtrx4[i])
                    mtrx10[k] = np.asarray(mtrx10[k]) - beta*np.asarray(mtrx10[i])
                    #print(np.asarray(mtrx10))
                j+=1
            #print('mtrx10 now(triangular matrix) is')
            #print(np.asarray(mtrx10))
            
            result = [[None]]*(len(mtrx10))
            for i in range(len(mtrx10)-1,-1,-1):
                t = 0
                for k in range(len(mtrx10)-1,i,-1):
                    t+=mtrx10[i][k]*result[k]    
                result[i] = (mtrx4[i]-t)/(mtrx10[i][i])
            print('results for workers of mtrx9', mtrx9)
            print(np.asarray(result))
            print('--------------------------------------------------------------')
            print('mtrx5:')
            print(np.asarray(mtrx5))
            
            end_time = time.time()
    
            print('\n')
            print('Time taken in seconds', end_time - start_time)
            print('\n')
            print('------------------------------------------------------------------')
            
            print('just for comparing...')
            print('mtrx1 * mtrx2:')
            print(np.dot(mtrx1, mtrx2))    
            
            #for validation
            for i in range(len(mtrx7)):
                index = mtrx7[i]-1
                value = mtrx5[i]
                result.insert(index, value)
            result = np.asarray(result)
            print('final result after reorder:')
            print(result)
            
            #take out the blow part
            # observedResults = np.dot(mtrx1, mtrx2)
            # a=len(result)
            # b=len(result[0])
            # c=a*b
            # observedResults = np.reshape(observedResults,(a,b,c))
            # print('comparison:')
            # print(np.subtract(result, observedResults))
                
        
        c+=1

    #validation
    observedResults = np.dot(mtrx1, mtrx2)
    a=len(result)
    b=len(result[0])
    c=a*b
    observedResults = np.reshape(observedResults,(a,b,c))
    print('comparison:')
    print(np.subtract(result, observedResults))
        
        
        
        
def slave():

    x = comm.recv(source=0, tag=rank)
    
    if barrier:
        comm.Barrier()    
    
    x = np.asarray(x)
    y = x[:,:-1]         #exclude the last column
    n = int(x[:,-1][0])  #extract the length of mtrx2
    
    r1 = y[:(len(y)-n)]  #r1 is actually the rows[i] above
    r2 = y[(len(y)-n):]  #r2 is the mtrx2
    
    z = np.dot(r1,r2)
    
    comm.send(z, dest=0, tag=rank)
    
        
        
        
        
#######################################################        
        
if __name__ == '__main__':
    
    if rank == 0:
        master()
        
    else:
        slave()
