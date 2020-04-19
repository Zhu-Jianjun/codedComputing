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

N = 12
K = 6 #number of identity blocks

#N = int(sys.argv[1])
#K = int(sys.argv[2])


def init_matrix():

    def encode(org):
        
        imtrx = np.identity(N, dtype=int)
        print('\n')
        
        global redun
        redun = np.zeros((N,N), dtype=int)  
        M = N//K
        for i in range(0,len(redun),M):
            for j in range(0,len(redun[0]),M):
                for k in range(M):
                    redun[i+k,j+k] = (i/M+1)**(j/M)

        print('redun:', np.asarray(redun).shape)
        #print(redun)
        print('------------------------------------------')
        
        encoder = np.vstack((imtrx, redun))
        print('encoder:', np.asarray(encoder).shape)
        #print(encoder)
        print('------------------------------------------')

        return encoder@org
    ######################################################################
    
    
    global mtrx1
    global mtrx1_coded
    #mtrx1 = [[np.random.randint(0, 9) for i in range(N)] for j in range(N)]
    mtrx1 = np.random.randint(2, size=(N, N))
    mtrx1_coded = encode(mtrx1)
    print('mtrx1:', np.asarray(mtrx1).shape)
    #print(mtrx1)
    print('---------------------------------------------')
    print('mtrx1_coded:', np.asarray(mtrx1_coded).shape)
    #print(mtrx1_coded)
    print('---------------------------------------------')
    ######################################################################
    
    global mtrx2
    #mtrx2 = [[np.random.randint(0, 9) for i in range(N)] for j in range(N)]
    mtrx2 = np.random.randint(2, size=(N, N))
    print('mtrx2:', np.asarray(mtrx2).shape)
    #print(mtrx2)
    ######################################################################


   
def multiply_matrix(X, Y):

    Z = [[sum(a * b for a, b in zip(X_row, Y_col)) for Y_col in zip(*Y)]
            for X_row in X]
    return Z

    

def split_matrix(seq, p):
    
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

    #mtrx3 is the data which needs to send to workers
    mtrx3 = []        
    for i in range(len(rows)):
        data = np.vstack((rows[i], mtrx2))
        temp = [len(mtrx2)]*(len(data))
        data1 = np.c_[data,temp] #provide the length in the last column
        mtrx3.append(data1)

    #print('mtrx3(data for workers) is')
    #print(mtrx3)
    print('--------------------------------------------------------')
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
        print('data is sending to rank {}'.format(pid))
        #print(mtrx3[pid-1])
        #print('---------------------------------------------------------')
        comm.send(mtrx3[pid-1], dest=pid, tag=pid)   #send redundency first
                   
    print('========================================================')
    print('\n')
    
    #######################################################################
    
    start_time = time.time()
    
    #######################################################################
    
    '''receive data from workers'''
    
    x = np.array(list(range(1, (K)+1)))
    vander_matrix = np.vander(x, K, increasing=True)
    #vander_matrix = vander_matrix[::-1]
    print('prepare the coefficients for figuring out the alpha & beta later...')
    print(vander_matrix)
    print('---------------------------------------------------------')
    
    c = 0
    mtrx4 = [] #group A(redundency)
    mtrx5 = [] #group B
    mtrx6 = [] #store the rank who sent back the redundency-based results
    mtrx7 = [] #store the rank who just send back the results
    
    mtrx11 = [] #used rows of vander_matrix 
    mtrx15 = [] #ranks with no contributions to the final results
    
    print('\n')
    print('start receiving...')
    print('\n')
    
    A = N//(len(mtrx1_coded)//workers)
    while True:
        
        if c < A:
            row = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=stat)
            source = stat.Get_source()
            
            global mtrx12
            mtrx12 = list(range(1,A+1))
            
            if source > A:
                mtrx6.append(source)
                mtrx4.append(row)
                print('Back from rank {}'.format(source))
                print('mtrx4(redundency-based results) now is', np.asarray(mtrx4).shape)
                #print(mtrx4)
                
                mtrx11.append(vander_matrix[source-A-1]) 
                print('mtrx11 now is')
                print(np.asarray(mtrx11))
                print('\n')

                
                if len(mtrx5)!=0:
                    for y in mtrx7:
                        if y in mtrx12:
                            mtrx12.remove(y) #update mtrx12
                            
                    for j in range(len(mtrx5)):
                        alpha=mtrx11[-1][mtrx7[j]-1]
                        print('alpha is', alpha)
                        mtrx4[-1]=np.asarray(mtrx4[-1])-alpha*(np.asarray(mtrx5[j]))
                    
                    mtrx11 = [[temp[k-1] for k in mtrx12] for temp in mtrx11]
                    print('mtrx11 after updated is')
                    print(np.asarray(mtrx11))
                        
                
                if len(mtrx4) > 1:
                    
                    ''' forward elimmination'''
                    j = 0
                    for i in range(len(mtrx4)-1):
                        pivot = mtrx11[i][j]
                        print('pivot is', pivot)
                        for k in range(i+1, len(mtrx4)):
                            target = mtrx11[k][j]
                            print('target is', target)
                            beta = target/pivot
                            print('beta is', beta)
                            mtrx4[k]=np.asarray(mtrx4[k])-beta*(np.asarray(mtrx4[i]))
                            mtrx11[k]=np.asarray(mtrx11[k])-beta*(np.asarray(mtrx11[i])) #update mtrx11
                            print('-------------------------------------')
                        j+=1
                    
                    print('mtrx11(coefficients of left terms after forward elimination) is')
                    print(np.asarray(mtrx11))
                    print('--------------------------------------------')
                    #print('mtrx4 now is')
                    #print(np.asarray(mtrx4))
                    #print('--------------------------------------------')
                    print('\n')
                    
                    
            else:
                mtrx7.append(source)
                mtrx5.append(row)
                print('Back from rank {}'.format(source))
                print('mtrx5 now is', np.asarray(mtrx5).shape)
                
                for y in mtrx7:
                    if y in mtrx12:
                        mtrx12.remove(y) #update mtrx12
                print('mtrx12 now is', mtrx12)

                
                if len(mtrx4)!=0:
                
                    '''simplify the redundency-based results'''
                    for i in range(len(mtrx11)):
                        #alpha=vander_matrix[mtrx6[i]-A-1][mtrx7[-1]-1]
                        if mtrx7[-1] == max(mtrx7):
                            alpha = mtrx11[i][mtrx7[-1]-len(mtrx7)]
                        elif mtrx7[-1] == min(mtrx7):
                            alpha = mtrx11[i][mtrx7[-1]-1]
                        else:
                            alpha = mtrx11[i][mtrx7[-1]-len(mtrx7)+1]
                        
                        #alpha=mtrx11[i][mtrx7[-1]-len(mtrx7)]
                        print('alpha is', alpha)
                        mtrx4[i]=np.asarray(mtrx4[i])-alpha*(np.asarray(mtrx5[-1]))                 
                    
                    mtrx11 = [[temp[k-1] for k in mtrx12] for temp in mtrx11]
                    print('mtrx11 after updated is')
                    print(np.asarray(mtrx11))
                    
                    ''' forward elimmination if necessary...'''
                    j = 0
                    for i in range(len(mtrx4)-1):
                        pivot = mtrx11[i][j]
                        print('pivot is', pivot)
                        for k in range(i+1, len(mtrx4)):
                            target = mtrx11[k][j]
                            print('target is', target)
                            beta = target/pivot
                            print('beta is', beta)
                            mtrx4[k]=np.asarray(mtrx4[k])-beta*(np.asarray(mtrx4[i]))
                            mtrx11[k]=np.asarray(mtrx11[k])-beta*(np.asarray(mtrx11[i]))
                            print('mtrx11(coefficients of left terms after forward elimination) now is')
                            print(np.asarray(mtrx11))
                            print('-------------------------------------')
                        j+=1
                
                
                print('\n')            
                            
        else:
        
            row1 = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=stat)            
            source1 = stat.Get_source()
            mtrx15.append(source1)
            
            
        c+=1
        
        if c == workers:
            break                   
      

    result = [[None]]*(len(mtrx11))
    for i in range(len(mtrx11)-1,-1,-1):
        t = 0
        for k in range(len(mtrx11)-1,i,-1):
            t+=mtrx11[i][k]*result[k]
                            
        result[i] = (mtrx4[i]-t)/(mtrx11[i][i])
                            


    print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
    print('Finally, mtrx4(redundency-based results) is')
    print(np.asarray(mtrx4).shape)
    print('mtrx6(ranks who have contributions to the redundency-based results):', mtrx6)
    print('++++++++++++++++++++++++++++++++++++++++')
    print('Finally, mtrx5 is')
    print(np.asarray(mtrx5).shape)
    print('mtrx7(ranks who have contributions to the results):', mtrx7)
    print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
    print('useless workers:', mtrx15) 
    print('\n')
    print('result now is')
    print(np.asarray(result))
    print('\n')  
    print('========================================================')       
    
    #############################################################################
    
    print('--------------------------------------------------------------')
    print('mtrx5:')
    print(np.asarray(mtrx5))    
                            
    #############################################################################                        
                            
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

    #req = comm.irecv(32768000, source=0, tag=rank)
    #x = req.wait()
    x = comm.recv(source=0, tag=rank)

    #comm.Barrier()
    
    time.sleep(1)
    #time.sleep(1/rank)  # difference is here
        
    x = np.asarray(x)
    y = x[:,:-1]       #exclude the last column
    n = x[:,-1][0]     #extract the length of mtrx2
        
    r1 = y[:(len(y)-n)]  #r1 is actually the rows[i] above
    r2 = y[(len(y)-n):]  #r2 is the mtrx2
    
    z = multiply_matrix(r1, r2)
    
    #reqs = comm.isend(z, dest=0, tag=rank)
    #reqs.wait()
    comm.send(z, dest=0, tag=rank)          
                            
                            
                            
                            
                            
                            
                            
                            
                            
if __name__ == '__main__':
    
    if rank == 0:
        master()
        
    else:
        slave()                           
                            
                            
                            
                            
                            
                    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    