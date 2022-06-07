#New Training Set pT independent and unbiased

# Common imports
import numpy as np
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import time
import pandas as pd 
#Random Seed
np.random.seed(42)
#__________________________________________________________________________________________________

#K = 5 #V0
K = 35 #num de variaveis topologicas que vamos considerar

# Define PID and Invariant Mass (GeV/c^2) of particles

#MyPID1 = 3122 #Lambda
MyPID1 = 1 #XiCC
#MyPID1 = 3334 #Omega -

#IM1 = 1.115683 #Lambda
IM1 = 3.6214 #XiCC
#IM1 = 1.67245 #Omega

sigma = 0.08    #0.02+1e-2

div_sg = 5
div_bg = 5
flag_cut = 1
#__________________________________________________________________________________________________
def Count_Particle( data ):

	Counts = 0

	data = np.asarray(data)
    
	for i in range(data.shape[0]):

		if (data[i][K+1] > IM1+5*sigma) or (data[i][K+1] < IM1-5*sigma) or (data[i][K]!=MyPID1):
			continue

		Counts+=1

	return (Counts)
#__________________________________________________________________________________________________
def Count_BG( data, flag ):

	Counts = 0

	data = np.asarray(data)
    
	for i in range(data.shape[0]):
		if(flag==1):
			if (data[i][K+1] > IM1+5*sigma) or (data[i][K+1] < IM1-5*sigma) or (data[i][K]!=0):
				continue;
		else: 		
			if (data[i][K]!=0):
				continue

		Counts+=1

	return (Counts)

def Count_Sig_BG(data, flag):
    Count_sig = 0
    Count_bg = 0
#__________________________________________________________________________________________________
def Build_Signal_Set( data, size ):

	data = np.asarray(data)
	np.random.shuffle(data)

	SigSet = np.zeros(shape=(size,K+3))
	ii = 0
	
	for i in range(data.shape[0]):
	
		if(ii==size):
			break

		if (data[i][K+1] > IM1+5*sigma) or (data[i][K+1] < IM1-5*sigma) or (data[i][K]!=MyPID1):
			continue

		else:
			#Fill
			for j in range(K+3):
				SigSet[ii][j] = data[i][j]
			ii+=1
	if(ii-size!=0):
		print('Huston we have a problem in Build_Signal_Set')

	return(SigSet)
#__________________________________________________________________________________________________
def Build_BG_Set( data, size):

	data = np.asarray(data)
	np.random.shuffle(data)

	BGSet = np.zeros(shape=(size,K+3))
	ii = 0
	
	for i in range(data.shape[0]):

		if(ii==size):
			break

#		if (data[i][K+1] > IM1+5*sigma) or (data[i][K+1] < IM1-5*sigma) or (data[i][K]!=0):
#		if(flag==1):
		if (data[i][K+1] > IM1+5*sigma) or (data[i][K+1] < IM1-5*sigma) or (data[i][K]!=0):
			continue;
#		if(flag==0): 		
#			if (data[i][K]!=0):
#				continue
		else:
			#Fill
			for j in range(K+3):
				BGSet[ii][j] = data[i][j]
			ii+=1
	if(ii-size!=0):
		print( 'Huston we have a problem in Build_BG_Set',ii,size )

	return(BGSet)
#__________________________________________________________________________________________________

start = time.time()
size_train = 0
size_valid = 0
for i in range(6):
    base_size1_total = 0
    base_size2_total = 0
    for data in pd.read_csv('/hadrex1/storage2/rramos/stratrack/machinelearning/samwise/merge01/XiCC_Data_Samwise_toshuf_' + str(i) + '.txt', delim_whitespace=True, skipinitialspace=True, chunksize = 5e7):
        data = data.to_numpy()
        base_size1 = Count_Particle(data)
        base_size2 = Count_BG(data, flag_cut)
        
        base_size1_total += base_size1
        base_size2_total += base_size2
        
    
        SigSet = Build_Signal_Set(data,base_size1)    
        BGSet = Build_BG_Set(data,base_size2)
        for j in range(div_sg):
            if(base_size1%div_sg==0):
                break
    
            base_size1-=1
            SigSet = np.delete(SigSet,0,0)
    
        for k in range(div_bg):
            if(base_size2%div_bg==0):
                break
    
            base_size2-=1
            BGSet = np.delete(BGSet,0,0)        
    	
    	
        SigSet_split = np.split(SigSet,div_sg,0)
        BGSet_split = np.split(BGSet,div_bg,0)
    
        SigSet_train = np.concatenate( (SigSet_split[0],SigSet_split[1],SigSet_split[2],SigSet_split[3]), 0 )
        BGSet_train = np.concatenate( (BGSet_split[0],BGSet_split[1],BGSet_split[2],BGSet_split[3]), 0 )
    
        SigSet_valid = SigSet_split[4]
        BGSet_valid = BGSet_split[4]
        
        SigSet_train = np.concatenate((SigSet_train,BGSet_train),0)
        np.random.shuffle(SigSet_train)
        size_train += len(SigSet_train)
        
        file_train = open('/hadrex1/storage2/rramos/stratrack/machinelearning/samwise/merge01/XiCC_DataTrainUnique_Samwise_toshuf_1.txt', 'a')
        np.savetxt(file_train, SigSet_train)
        file_train.close()
        
        SigSet_valid = np.concatenate((SigSet_valid,BGSet_valid),0)
        np.random.shuffle(SigSet_valid)
        size_valid += len(SigSet_valid)
        
        file_valid = open('/hadrex1/storage2/rramos/stratrack/machinelearning/samwise/merge01/XiCC_DataValidUnique_Samwise_toshuf_1.txt', 'a')
        np.savetxt(file_valid, SigSet_valid)
        file_valid.close()
        
    print('Size Sg',i,':',base_size1_total)
    print('Size Bg',i,':',base_size2_total)
    print('File ',i,' Done!')

print('Train size: ', size_train)
print('Valid size: ', size_valid)
print("The time used to execute this is given below")

end = time.time()

print(end - start)
#print(21*2*base_size)



	


