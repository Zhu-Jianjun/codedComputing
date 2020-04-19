
from mpi4py import MPI
import time
import sys
import numpy as np


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
workers = comm.Get_size() - 1

stat = MPI.Status()

mtrx1 = []
mtrx2 = []

'''
Important: the number of the rows of each redundency block should be equal to 
the number of the rows of each worker. (relationship)

e.g, if N=12, then 24 rows(mtrx1_coded), 12 workers, then N//K has to be 24//12, 
which choose K=6. 
Or, if N=12, then 24 rows(mtrx1_coded), I want K=4, then N//K=3, 24//3=8, 
8 is the number of workers.
'''

N = 120
K = 6 #number of identity blocks
M = int(N/K)

#N = int(sys.argv[1])
#K = int(sys.argv[2])


def init_matrix():

    def encode(org):
        
        imtrx = np.identity(N, dtype=int)
        print('\n')
        #print('------------------------------------------')
        #print('imtrx:')
        #print(imtrx)
        #print(np.asarray(imtrx).shape)
        #print('------------------------------------------')
        
        global redun

        redun = np.zeros((N,N), dtype=int)  #first N is the number of rows
        #M = int(N//K)
        for i in range(0,len(redun),M):
            for j in range(0,len(redun[0]),M):
                for k in range(M):
                    redun[i+k,j+k] = (i/M+1)**(j/M)

        print('redun:')
        print(redun)
        print(np.asarray(redun).shape)
        #print('------------------------------------------')
        
        encoder = np.vstack((imtrx, redun))
        print('encoder:')
        print(encoder)
        print(np.asarray(encoder).shape)
        print('------------------------------------------')
           
        
        return encoder@org
    ######################################################################
    
    
    global mtrx1
    global mtrx1_coded
    
    #mtrx1 = [[np.random.randint(0, 9) for i in range(N)] for j in range(N)]
    mtrx1 = np.random.randint(2, size=(N, N))
    mtrx1_coded = encode(mtrx1)
    print('mtrx1:')
    print(mtrx1)
    print(np.asarray(mtrx1).shape)
    print('---------------------------------------------')
    print('mtrx1_coded:')
    #print(mtrx1_coded)
    print(np.asarray(mtrx1_coded).shape)
    print('---------------------------------------------')
    ######################################################################
    
    global mtrx2
    #mtrx2 = [[np.random.randint(0, 9) for i in range(N)] for j in range(N)]
    mtrx2 = np.random.randint(2, size=(N, N))
    print('mtrx2:')
    print(mtrx2)
    print(np.asarray(mtrx2).shape)
    ######################################################################


   
def multiply_matrix(X, Y):

    Z = [[sum(a * b for a, b in zip(X_row, Y_col)) for Y_col in zip(*Y)]
            for X_row in X]

    return Z

    

def split_matrix(seq, p):
    # to generate rows
    
    rows = []
    n = len(seq) // p
    r = len(seq) % p
    b, e = 0, n + min(1, r)
    for i in range(p):
        rows.append(seq[b:e])
        r = max(0, r - 1)
        b, e = e, e + n + min(1, r)
    
    return rows 


 
##############################################################  
##############################################################
##############################################################


def master():
    '''get all data ready'''
        
    ########################################################################
    init_matrix()
    rows = split_matrix(mtrx1_coded, workers)
    #print('--------------------------------------------------')
    global P
    P = len(rows[0])
    #print('the length of rows[0] which is P:')
    #print(P)
    #print('--------------------------------------------------')

    #mtrx3 is the data which needs to send to workers
    mtrx3 = []        
    for i in range(len(rows)):
        data = np.vstack((rows[i], mtrx2))
        temp = [len(mtrx2)]*(len(data))
        data1 = np.c_[data,temp] #provide the length in the last column
        mtrx3.append(data1)
        #i+=1
    #print('mtrx3(data for workers) is')
    #print(mtrx3)
    #mtrx3_temp = mtrx3[::-1]
    print('--------------------------------------------------------')
    #print('mtrx3_temp(data for workers) is')
    #print(mtrx3_temp)
    print('There are totally {} blocks for workers.'.format(len(mtrx3)))
    print('\n')
    print('========================================================')
    print('data is ready!')
    print('========================================================')
    
    #######################################################################
   
    '''send mtrx3 data to workers'''
    
    #######################################################################
    print('data is sending...')
    print('\n')
    
    for pid in range(workers, 0, -1): 
        #send redundency first
        #req = comm.isend(mtrx3_temp[i], dest=pid, tag=pid)
        #req.wait()
        print('data is sending to rank {}'.format(pid))
        #print(mtrx3[pid-1])
        #print('---------------------------------------------------------')
        comm.send(mtrx3[pid-1], dest=pid, tag=pid)   
                   
    print('========================================================')
    

def master1():    
    
    #######################################################################
    
    start_time = time.time()
    
    #######################################################################
    
    '''receive data from workers'''
    
    x = np.array(list(range(1, (K)+1)))
    vander_matrix = np.vander(x, K, increasing=True)
    #vander_matrix = vander_matrix[::-1]
    print('vander_matrix which is used to figure out the factor...')
    print(vander_matrix)
    print('---------------------------------------------------------')
    
    c = 0
    mtrx4 = [] #group A(redundency)
    mtrx5 = [] #group B
    mtrx6 = [] #store the rank who sent back the redundency-based results
    mtrx7 = [] #store the rank who just send back the results
    
    mtrx15 = []
    
    print('\n')
    print('receiving partial results...')
    print('\n')
    
    start_receiving = time.time()
    
    while c < workers:   # should come from N // len(rows[0])
        start_each = time.time()
        
        row = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=stat)
        source = stat.Get_source()
        
        if c < (N//P):
            
            if (len(mtrx1_coded)//workers)!=0:   #need when run on the LONI
                A = N//(len(mtrx1_coded)//workers)  #how many blocks are enough to decode
                
                
                if source > A:
                    mtrx6.append(source)
                    #mtrx4 += row
                    mtrx4.append(row)
                    print('Back from rank {}'.format(source))
                    print('mtrx4(redundency group) now is', np.asarray(mtrx4).shape)
                    #print(mtrx4)
                    print('\n')
                    
                else:
                    mtrx7.append(source)
                    #mtrx5 += row
                    mtrx5.append(row)
                    print('Back from rank {}'.format(source))
                    print('mtrx5 now is', np.asarray(mtrx5).shape)
                    #print(mtrx5)
                    print('\n')
            
                
            
        # #added part
        # else:
        
            # row1 = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=stat)            
            # source1 = stat.Get_source()
            # mtrx15.append(source1)
            
        end_each = time.time()
        print('time to receive each result:', end_each-start_each)
        
        c+=1
    
    
    end_receiving = time.time()
    print('time to receive all the necessary results:', end_receiving-start_receiving)
    
    print('useless workers:', mtrx15)
    print('\n')
    print('Finally, mtrx4(redundency-based results) is')
    print(np.asarray(mtrx4).shape)
    print('mtrx6(ranks who have contributions to the redundency-based results):', mtrx6)
    print('++++++++++++++++++++++++++++++++++++++++')
    print('Finally, mtrx5 is')
    print(np.asarray(mtrx5).shape)
    print('mtrx7(ranks who have contributions to the results):', mtrx7)
    print('\n')  
    print('========================================================')       
    
    #############################################################################
    
    print('\n')
    print('starting decoding...')
    print('\n')
    print('---------------------------------------------------------')
    
    #$$$$$$$$$$$$$$$$$$$$$$$ Decoding $$$$$$$$$$$$$$$$$$$$$$$$$$$
    
    print('mtrx4 length:', len(mtrx4))
    print('mtrx5 length:', len(mtrx5))
    print('mtrx6 length(rank):', len(mtrx6))
    print('mtrx7 length(rank):', len(mtrx7))
    print('---------------------------------------------------------')
    
    start_processing = time.time()
    
    '''remove all the terms that are not necessary to decode and then update'''
    for i in range(len(mtrx4)):        
        for j in range(len(mtrx5)):    
            alpha = vander_matrix[mtrx6[i]-(N//P)-1][mtrx7[j]-1]  
            print('alpha is', alpha)           
            mtrx4[i] = np.asarray(mtrx4[i]) - alpha*(np.asarray(mtrx5[j]))
            
    print('--------------------------------------------')
    #print('just checking mtrx4:', mtrx4)
    #now, mtrx4 is only about variables that need to be decoded...
    
    
    '''forward elimination'''
    mtrx8 = []
    for i in range(len(mtrx6)):
        mtrx8.append(vander_matrix[mtrx6[i]-(N//P)-1])    
    print('mtrx8 is')
    print(np.asarray(mtrx8))
    print('--------------------------------------------')
    
    mtrx9 = list(range(1,workers+1))
    for x in list(range(workers,workers-(N//P), -1)): #need to remove all the redundency-based rank
        if x in mtrx9:
            mtrx9.remove(x)
    for y in mtrx7:
        #mtrx8.append(workers+1-y)
        if y in mtrx9:
            mtrx9.remove(y)
    
    print('rank that did not send back partial results(mtrx9):')
    print(mtrx9)
    print('---------------------------------------------------------')
    
    mtrx10 = [[temp[k-1] for k in mtrx9] for temp in mtrx8]
    print('mtrx10 is')
    print(np.asarray(mtrx10))
    
    j = 0
    for i in range(len(mtrx4)-1):
        pivot = mtrx10[i][j]   #later consider situation that pivot = 0
        #print('pivot is', pivot)
        for k in range(i+1, len(mtrx4)):
            target = mtrx10[k][j]
            #print('target is', target)
            beta = target/pivot
            print('beta is', beta)
            mtrx4[k] = np.asarray(mtrx4[k]) - beta*np.asarray(mtrx4[i])
            mtrx10[k] = np.asarray(mtrx10[k]) - beta*np.asarray(mtrx10[i])
            #print(np.asarray(mtrx10))
        j+=1
    
    print('--------------------------------------------')
    print('mtrx10 now(triangular matrix) is')
    print(np.asarray(mtrx10))
    print('--------------------------------------------')
    #print('mtrx4 now(new partial results corresponding to triangular matrix) is')
    #print(np.asarray(mtrx4))
    
    #now, mtrx4 should be ready for back-substitution   

    result = [[None]]*(len(mtrx10))
    for i in range(len(mtrx10)-1,-1,-1):
        t = 0
        for k in range(len(mtrx10)-1,i,-1):
            t+=mtrx10[i][k]*result[k]
            
        result[i] = (mtrx4[i]-t)/(mtrx10[i][i])
    
    print('results for workers of mtrx9 which are', mtrx9)
    print(np.asarray(result))
    
    
    print('--------------------------------------------------------------')
    print('mtrx5:')
    print(np.asarray(mtrx5))
    
    print('\n')
    end_processing = time.time()
    print('time for processing:', end_processing-start_processing)
        
    ##################################################################   
    
    end_time = time.time()

    print('--------------------------------------------------------------')
    print('\n')
    print('Time taken in seconds(with sleep)', end_time - start_time)
    print('\n')
    print('--------------------------------------------------------------')
    print('just for comparing...')
    print('mtrx1 * mtrx2:')
    print(np.dot(mtrx1, mtrx2))
        
        
        
def slave():
    start_computing = time.time()
    #req = comm.irecv(32768000, source=0, tag=rank)
    #x = req.wait()
    
    x = comm.recv(source=0, tag=rank)
    
    time.sleep(1)  # difference is here
        
    x = np.asarray(x)
    y = x[:,:-1]       #exclude the last column
    n = x[:,-1][0]     #extract the length of mtrx2
        
    r1 = y[:(len(y)-n)]  #r1 is actually the rows[i] above
    r2 = y[(len(y)-n):]  #r2 is the mtrx2
    
    z = multiply_matrix(r1, r2)
    
    #reqs = comm.isend(z, dest=0, tag=rank)
    #reqs.wait()
        
    comm.send(z, dest=0, tag=rank)
    end_computing = time.time()
    print('computing time on the slave:', end_computing-start_computing)
   
   
   
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$   
   
   
if __name__ == '__main__':
    
    if rank == 0:
        master()
    
    if rank != 0:
        slave()
    
    if rank == 0:
        master1()
    
