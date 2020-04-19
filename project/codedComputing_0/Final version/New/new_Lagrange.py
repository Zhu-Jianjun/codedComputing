
from mpi4py import MPI
import time
import sys
import numpy as np


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
workers = comm.Get_size() - 1

stat = MPI.Status()
barrier = True

N = 12 #mtrix size
G = 3  #identity matrix size, corresponding to number of rows in each block, maximize the number of workers used when G=2

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
    
    start_time = time.time()
    
    '''receive data from workers'''
    
    redun = poly[W:] #prepare poly-coefficients for decoding
    print('redun coefficients is')
    print(redun)
    print('\n')
    
    c = 0
    mtrx4 = [] #group A(redundency)
    mtrx5 = [] #group B
    mtrx6 = [] #store the rank who sent back the redundency-based results
    mtrx7 = [] #store the rank who just send back the results
    
    mtrx11 = [] #used rows of redun 
    
    mtrx12 = list(range(1,W+1))
    while c < workers:
        row = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=stat)
        source = stat.Get_source()
        
        if c < W:    
            
            if source > W:
                mtrx6.append(source)
                mtrx4.append(row)
                #print('Back from rank {}'.format(source))
     
                mtrx11.append(redun[source-W-1]) 
                #print('mtrx11 now is')
                #print(np.asarray(mtrx11))
                #print('\n')
                
                if len(mtrx5)!=0:
                    
                    '''remove all the terms that are not necessary to decode and then update'''      
                    for j in range(len(mtrx5)):
                        alpha=mtrx11[-1][mtrx7[j]-1]
                        #print('alpha is', alpha)
                        mtrx4[-1]=np.asarray(mtrx4[-1])-alpha*(np.asarray(mtrx5[j]))
                        mtrx11[-1][mtrx7[j]-1] = 0
                    
                    #print('mtrx11 after simplify is')
                    #print(np.asarray(mtrx11))
     
                
                if len(mtrx4) > 1:
                    
                    ''' forward elimmination'''
                    j = 0
                    for i in range(len(mtrx4)-1):
                        pivot = mtrx11[i][j]
                        #print('pivot is', pivot)
                        ##############################################
                        if pivot == 0:
                            #print('pivot should not be 0, go to next col which is not 0...')
                            while mtrx11[i][j]==0:
                                j+=1
                                if mtrx11[i][j]!=0:   
         
                                    pivot=mtrx11[i][j]
                                    #print('pivot now should be', pivot)
                                    
                                    target = mtrx11[-1][j]
                                    #print('target should be', target)
                                    beta = target/pivot
                                    #print('beta should be', beta)
                                    mtrx4[-1]=np.asarray(mtrx4[-1])-beta*(np.asarray(mtrx4[i]))
                                    mtrx11[-1]=np.asarray(mtrx11[-1])-beta*(np.asarray(mtrx11[i]))
                                    #print('mtrx11 now should be')
                                    #print(np.asarray(mtrx11))
                                    #print('-------------------------------------')
                                    
                                    '''
                                    for k in range(i+1, len(mtrx4)):
                                        target = mtrx11[k][j]
                                        print('target should be', target)
                                        beta = target/pivot
                                        print('beta should be', beta)
                                        mtrx4[k]=np.asarray(mtrx4[k])-beta*(np.asarray(mtrx4[i]))
                                        mtrx11[k]=np.asarray(mtrx11[k])-beta*(np.asarray(mtrx11[i])) #update mtrx11
                                        print('mtrx11 now should be')
                                        print(np.asarray(mtrx11))
                                        print('-------------------------------------')
                                    '''
                                    continue    
                                   
                        #############################################
                        else:
                            target = mtrx11[-1][j]
                            #print('target should be', target)        
                            beta = target/pivot
                            #print('beta should be', beta)
                            mtrx4[-1]=np.asarray(mtrx4[-1])-beta*(np.asarray(mtrx4[i]))
                            mtrx11[-1]=np.asarray(mtrx11[-1])-beta*(np.asarray(mtrx11[i]))
                            #print('mtrx11 now should be')
                            #print(np.asarray(mtrx11))
                            #print('-------------------------------------')
                            '''
                            for k in range(i+1, len(mtrx4)):
                                target = mtrx11[k][j]
                                print('target is', target)
                                beta = target/pivot
                                print('beta is', beta)
                                mtrx4[k]=np.asarray(mtrx4[k])-beta*(np.asarray(mtrx4[i]))
                                mtrx11[k]=np.asarray(mtrx11[k])-beta*(np.asarray(mtrx11[i])) #update mtrx11
                                print('-------------------------------------')
                            '''
                        j+=1
                    
                    #print('mtrx11(coefficients of left terms after forward elimination) is')
                    #print(np.asarray(mtrx11))
                    #print('--------------------------------------------')
                    #print('mtrx4 now is')
                    #print(np.asarray(mtrx4))
                    #print('--------------------------------------------')
                   
                    
                    
            else:
                mtrx7.append(source)
                mtrx5.append(row)
                #print('Back from rank {}'.format(source))
                #print('mtrx5 now is', np.asarray(mtrx5).shape)
                
                if len(mtrx4)!=0:
                
                    #remove all the terms that are not necessary to decode and then update
                    for i in range(len(mtrx4)):
                        alpha = mtrx11[i][mtrx7[-1]-1] 
                        # if mtrx7[-1] == max(mtrx7):
                            # alpha = mtrx11[i][mtrx7[-1]-len(mtrx7)]
                        # elif mtrx7[-1] == min(mtrx7):
                            # alpha = mtrx11[i][mtrx7[-1]-1]
                        # else:
                            # alpha = mtrx11[i][mtrx7[-1]-len(mtrx7)+1]
                        
                        #print('alpha is', alpha)
                        mtrx4[i]=np.asarray(mtrx4[i])-alpha*(np.asarray(mtrx5[-1]))                 
                        mtrx11[i][mtrx7[-1]-1] = 0
                    
                    #print('mtrx11 after updated is')
                    #print(np.asarray(mtrx11))
                    
                    
                    # forward elimmination if necessary...
                    if mtrx7[-1] < len(mtrx11):
                        ''' forward elimmination'''
                        j = 0
                        for i in range(len(mtrx4)-1):
                            pivot = mtrx11[i][j]
                            #print('pivot is', pivot)
                            ##############################################
                            if pivot == 0:
                                #print('pivot should not be 0, go to next col which is not 0...')
                                while mtrx11[i][j]==0:
                                    j+=1
                                    if mtrx11[i][j]!=0:   
             
                                        pivot=mtrx11[i][j]
                                        #print('pivot now should be', pivot)
                                        for k in range(i+1, len(mtrx4)):
                                            target = mtrx11[k][j]
                                            #print('target should be', target)
                                            beta = target/pivot
                                            #print('beta should be', beta)
                                            mtrx4[k]=np.asarray(mtrx4[k])-beta*(np.asarray(mtrx4[i]))
                                            mtrx11[k]=np.asarray(mtrx11[k])-beta*(np.asarray(mtrx11[i])) #update mtrx11
                                            #print('mtrx11 now should be')
                                            #print(np.asarray(mtrx11))
                                            #print('-------------------------------------')
                                        continue
                            #############################################
                            else:
                            
                                for k in range(i+1, len(mtrx4)):
                                    target = mtrx11[k][j]
                                    #print('target is', target)
                                    beta = target/pivot
                                    #print('beta is', beta)
                                    mtrx4[k]=np.asarray(mtrx4[k])-beta*(np.asarray(mtrx4[i]))
                                    mtrx11[k]=np.asarray(mtrx11[k])-beta*(np.asarray(mtrx11[i])) #update mtrx11
                                    #print('-------------------------------------')
                            j+=1
            
        c+=1
                          
      
    for y in mtrx7:
        if y in mtrx12:
            mtrx12.remove(y) #update mtrx12
    #print('mtrx12(index of left terms) is', mtrx12)
    mtrx11 = [[temp[k-1] for k in mtrx12] for temp in mtrx11]
    print('mtrx11(triangular matrix) is')
    print(np.asarray(mtrx11))
    
    '''
    # forward elimmination
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
    
    print('mtrx11(triangular matrix) finally is')
    print(np.asarray(mtrx11))
    '''
    
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
    #print('just for comparing...')
    #print('mtrx1 * mtrx2:')
    #print(np.dot(mtrx1, mtrx2))                      

    #validation
    for i in range(len(mtrx7)):
        index = mtrx7[i]-1
        value = mtrx5[i]
        result.insert(index, value)
    result = np.asarray(result)
    #print(result)

    observedResults = np.dot(mtrx1, mtrx2)
    a=len(result)
    b=len(result[0])
    c=a*b
    observedResults = np.reshape(observedResults,(a,b,c))
    print('comparison:')
    print(np.subtract(result, observedResults))


                            
          

def slave():

    #req = comm.irecv(32768000, source=0, tag=rank)
    #x = req.wait()
    x = comm.recv(source=0, tag=rank)

    if barrier:
        comm.Barrier()
    
    #time.sleep(1)
        
    x = np.asarray(x)
    y = x[:,:-1]       #exclude the last column
    n = int(x[:,-1][0])     #extract the length of mtrx2
        
    r1 = y[:(len(y)-n)]  #r1 is actually the rows[i] above
    r2 = y[(len(y)-n):]  #r2 is the mtrx2
    
    z = np.dot(r1,r2)
    
    #reqs = comm.isend(z, dest=0, tag=rank)
    #reqs.wait()
    comm.send(z, dest=0, tag=rank)          
                            
                            
##########################################################################                            
##########################################################################                            
                            
                         
if __name__ == '__main__':
    
    if rank == 0:
        master()                    
    
    else:
        slave()
                            
                            
                            
                            
                    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    