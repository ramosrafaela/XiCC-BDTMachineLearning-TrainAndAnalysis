#!/usr/bin/env python
# coding: utf-8

# In[2]:


#pip install --upgrade xgboost


# In[2]:


# Python ≥3.5 is required
import sys
assert sys.version_info >= (3, 5)

# Common imports
import numpy as np
import os
import matplotlib as mpl
import matplotlib.pyplot as plt

#XGB
import xgboost as xgb
import xgboost as get_score

#Metrics
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, explained_variance_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV


import joblib
import time


#Random Seed
np.random.seed(42)

print(xgb.__version__)


# In[3]:


#from google.colab import drive
#drive.mount('/content/drive')


# In[4]:


# Define number of features K for ML

#K = 5 #V0
K = 35 #Cascade

# Define PID and Invariant Mass (GeV/c^2) of particles

#MyPID1 = 3122 #Lambda
MyPID1 = 1 #Xi-
#MyPID1 = 3334 #Omega -

#IM1 = 1.115683 #Lambda
IM1 = 3.6214 #XiCC
#IM1 = 1.67245 #Omega

# pT binning

#pT_min = np.linspace(0.0,9.8,50)
pT_min = np.array([0.0,2.0,4.0,6.0,8.0,10.0])
#pT_max = np.linspace(0.2,10.0,50)
pT_max = np.array([2.0,4.0,6.0,8.0,10.0,15.0])
nbins = 6

print(pT_min)
print(pT_max)

#Standard Deviation of the peak arround IM1
sigma = 0.03
num = 2
merge = 1


# In[5]:


def Count_Particle( data , PID ):

    data = np.asarray(data)
#np.unique , devolver um array da contagem de cada valor único e o array ordenado do menor para maior
    PID_Values, Counts = np.unique(data, return_counts=True)
# .shape attribute returns the dimensions of the array. If Y has n rows and m columns, then Y.shape is (n,m). So Y.shape[0] is n        
    for i in range(PID_Values.shape[0]):

        if PID_Values[i] == PID:

            return (Counts[i])

    return 0


# In[6]:


def PID_to_Class(PID):

    #Tamanho do array de PIDs
    size = PID.shape[0]

    #Matriz para guardar as classe relativas a cada PID, cada linha sera um hot-vector [1,0] para classe 0, [0,1] para classe 1
  #aqui cria uma matriz nx2 preenchida com zeros
    Class = np.zeros(shape=(size,))

    #Numero de candidatos na clase 0
    N = np.zeros(shape=(2,))

    for i in range(PID.shape[0]):

        if PID[i]==MyPID1:
            Class[i] = 1
            N[1]+=1

        else:
            Class[i] = 0
            N[0]+=1
  
    return Class, N


# In[7]:

start = time.time()

#LOAD TRAINING SET
#aqui monta os dados para treino
TRAIN = np.load('/hadrex1/storage2/rramos/stratrack/machinelearning/skywalker/data_train02/merge01/lowpt/XiCC_DataTrain_Skywalker_v2_npy.npy')

X_train = TRAIN[:,:K]
PID_train = TRAIN[:,K] 
IM_train = TRAIN[:,K+1]
pT_train = TRAIN[:,K+2]
Y_train, N_train = PID_to_Class(PID_train)#Y vai ser a classe e o N o numero que recebe
#aqui monta os dados para teste
'''
VALID = np.loadtxt('merge0'+str(merge)+'/XiCC_DataValidUnique_Samwise_1.txt')

X_valid = VALID[:,:K]
PID_valid = VALID[:,K]
IM_valid = VALID[:,K+1]
pT_valid = VALID[:,K+2]
Y_valid, N_valid = PID_to_Class(PID_valid)


# In[14]:


import pandas as pd
df = pd.DataFrame(TRAIN)
import matplotlib.pyplot as plt

plt.matshow(df.corr())
plt.show()

'''
# In[15]:

print("Num. de sinal treino:", Count_Particle(PID_train,1))
print("Num. de fundo treino:", Count_Particle(PID_train,0))

###validacao

#print("Num. de sinal validacao:", Count_Particle(PID_valid,1))
#print("Num. de fundo validacao:", Count_Particle(PID_valid,0))

# In[ ]:


#BDT
#this code is to run the GridSearchCV
#params = { 'max_depth': [3,6],
#           'learning_rate': [0.1, 0.5],
#           'n_estimators': [100, 200],
#           #'colsample_bytree': [0.3, 0.7]
#            'objective': ['binary:logistic']}

#BDT = xgb.XGBRegressor(seed=42)

#BDTr =  GridSearchCV(estimator=BDT,
#                     param_grid=params,
#                     scoring='neg_mean_squared_error',
#                     verbose=1)

#BDTr.fit(X_train, Y_train, eval_set=[(X_train, Y_train), (X_valid, Y_valid)])

#print("Best parameters:", BDTr.best_params_)
#print("Lowest RMSE: ", (-BDTr.best_score_)**(1/2.0))


# In[17]:


#BDT
#aqui ele ta treinando 
BDT = xgb.XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, objective='binary:logistic',random_state=42, max_delta_step=10.0)
#BDT = xgb.XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.5, objective='binary:logistic',random_state=1,eval_metric="logloss")

#BDT.fit(X_train, Y_train, eval_set=[(X_train, Y_train)])
BDT.fit(X_train, Y_train)

# In[19]:


#here I rename the variables to identify it
orig_feature_names = ['fXiDCAxyToPV' , 'fXiccDCAxyToPV', 'fXiDCAzToPV', 'fXicPionDCAxyToPV1', 'fXicPionDCAzToPV1', 'fXicPionDCAxyToPV2', 'fXicPionDCAzToPV2', 'fXiccDCAzToPV', 'fPiccDCAzToPV', 'fXicDaughterDCA', 'fXiccDaughterDCA', 'fXicDecayRadius', 'fXiCCtoXiCLength', 'fXicDCAxyToPV', 'fXicDecayDistanceFromPV', 'fXiDecayLength', 'fPiccDCAxyToPV', 'fXiccDecayDistanceFromPV', 'fXiccDecayRadius', 'fXicDCAzToPV', 'fXiHitsAdded', 'fXiDecayRadius', 'fV0DecayRadius', 'fV0DCAxyToPV', 'fV0DCAzToPV', 'fNegativeDCAz', 'fV0DecayLength', 'fXiCtoXiLength', 'fXiDauDCA', 'fNegativeDCAxy', 'fBachelorDCAz', 'fPositiveDCAz', 'fPositiveDCAxy', 'fV0DauDCA', 'fBachelorDCAxy']
BDT.get_booster().feature_names = orig_feature_names


# In[20]:


BDT.get_booster().feature_names
fig, ax = plt.subplots(figsize=(18,10))
xgb.plot_importance(BDT, max_num_features=50, height=0.8, ax=ax, color='lightpink')
plt.savefig('/hadrex1/storage2/rramos/stratrack/machinelearning/skywalker/data_train02/merge01/lowpt/systanalysis/train03/importance_feature_'+str(num)+'.pdf', format='pdf')



# In[35]:


# save in JSON format
BDT.save_model('/hadrex1/storage2/rramos/stratrack/machinelearning/skywalker/data_train02/merge01/lowpt/systanalysis/train03/XiCC_BDT_Recipe_'+str(num)+'.json')
# save in text format
BDT.save_model('/hadrex1/storage2/rramos/stratrack/machinelearning/skywalker/data_train02/merge01/lowpt/systanalysis/train03/XiCC_BDT_Recipe_'+str(num)+'.txt')


# In[36]:


# In[21]:

'''
Y_pred = BDT.predict(X_valid) #previsoes

#np.savetxt('/content/drive/MyDrive/2021 ML/ULTRON/BDT/ML_Pred/Valid_Pred.txt',Y_pred)

#Plot output distribution in Valid Set
N_Particles = Count_Particle(PID_valid,MyPID1) #conta o numero de Xi na amostra
N_BG = PID_valid.shape[0] - N_Particles #todas as particulas que nao sao o Xi-

Class0 = np.ndarray(shape=(N_BG,1))
Class1 = np.ndarray(shape=(N_Particles,1))

i0 = 0
i1 = 0

for i in range(Y_pred.shape[0]):
    
  if (Y_valid[i]==0.):
    Class0[i0] = Y_pred[i]
    i0+=1

  else:
    Class1[i1] = Y_pred[i]
    i1+=1

bins = np.linspace(0,1,100) # produz um array com 100 entradas de 0 a 1 [0, 0.01, 0.02,..., 1]

fig = plt.figure(figsize=(8,8))
ax1 = fig.add_subplot()
ax1.set_ylabel('Normalized Counts', fontsize='xx-large')
plt.xticks(fontsize='xx-large')
#ax1.set_xlabel('BDT Output (P)', fontsize='xx-large')
ax1.set_xlabel('ML(BDT) response', fontsize='xx-large')
plt.yticks(fontsize='xx-large')

weights0 = np.ones_like(Class0) / len(Class0)
weights1 = np.ones_like(Class1) / len(Class1)

plt.hist(Class0,bins,alpha=0.5,label='Background', density=True, color='steelblue')#
plt.hist(Class1,bins,alpha=0.5, label='Signal', density=True, color='crimson')#
#plt.title('BDT output: Valid Set',fontsize='xx-large')
plt.title('ML output probability: Valid Test,     0.0 < pT < 15.0 GeV/c',fontsize='xx-large')
plt.yscale('log')
#plt.ylim([1,5e6])
plt.xlim([0,1])
#plt.ylim([0,1])
plt.legend(fontsize='xx-large')
plt.savefig('merge0'+str(merge)+'/res0'+str(num)+'/ml_output_validset_'+str(num)+'.pdf', format='pdf')



# In[38]:


#from yellowbrick.classifier import ConfusionMatrix
#cm = ConfusionMatrix(BDT)
#cm.fit(X_train, Y_train)
#cm.score(X_valid, Y_valid)


# In[22]:


# retrieve performance metrics
results = BDT.evals_result()
epochs = len(results['validation_0']['logloss'])
x_axis = range(0, epochs)

# plot log loss
plt.figure()    
fig, ax = plt.subplots(figsize=(5,5))
ax.plot(x_axis, results['validation_0']['logloss'], label='Train')
ax.plot(x_axis, results['validation_1']['logloss'], label='Test')
ax.legend()

plt.ylabel('LogLoss', fontsize='xx-large')
plt.xlabel('n_estimator', fontsize='xx-large')
plt.title('XGBoost LogLoss')
plt.savefig('merge0'+str(merge)+'/res0'+str(num)+'/logloss_'+str(num)+'.pdf', format='pdf')


# In[12]:


t = np.linspace(0,1,51)
print(t)


# In[13]:


#Signif = SIG/sqrt(SIG+BG)

sig_thrsh = np.zeros(51)
bg_thrsh = np.zeros(51)

for j in range(51):

    for i in range(Class0.shape[0]):

        if(Class0[i]>=t[j]):
            bg_thrsh[j]+=1

    for k in range(Class1.shape[0]):
        
        if(Class1[k]>=t[j]):
            sig_thrsh[j]+=1


# In[17]:

plt.figure()
plt.plot(t,bg_thrsh,'bo')
plt.ylabel('Background', fontsize='xx-large')
plt.xlabel('Threshold', fontsize='xx-large')
#plt.plot(t,sig_thrsh,'ro')
plt.yscale('log')
plt.savefig('merge0'+str(merge)+'/res0'+str(num)+'/bg_thrsh_'+str(num)+'.pdf', format='pdf')



# In[18]:

plt.figure()
plt.plot(t,sig_thrsh,'ro')
plt.ylabel('Signal', fontsize='xx-large')
plt.xlabel('Threshold', fontsize='xx-large')
plt.yscale('log')
plt.savefig('merge0'+str(merge)+'/res0'+str(num)+'/sig_thrsh_'+str(num)+'.pdf', format='pdf')


# In[65]:


signif = []

for i in range(50):
    signif.append(sig_thrsh[i]/np.sqrt(bg_thrsh[i]+sig_thrsh[i]) )

signif = np.array(signif)

plt.figure()    
#plt.plot(t, sig_thrsh/np.sqrt(sig_thrsh+bg_thrsh))
plt.plot(t[0:50],signif)
plt.ylabel('Significance', fontsize='xx-large')
plt.xlabel('Threshold', fontsize='xx-large')
plt.yscale('log')
plt.savefig('merge0'+str(merge)+'/res0'+str(num)+'/significance_thrsh_'+str(num)+'.pdf', format='pdf')


# In[ ]:



#TEST THE MODEL AND SAVE OUTPUT
n = 5
num_pt = 6
Sg_thrsh_pT = np.zeros(shape=(n,num_pt))
Bg_thrsh_pT = np.zeros(shape=(n,num_pt))
#t = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
t = np.linspace(0,0.98,5)
print(t)

for g in range(num_pt):
    
    #DATA1 = np.loadtxt('/content/drive/MyDrive/2021 ML/ULTRON/Data/XiMinus_Data_ULTRON_'+str (g)+'.txt') 
    #DATA2 = np.loadtxt('/content/drive/MyDrive/2021 ML/ULTRON/Data3/XiMinus_DataMC_ULTRON_'+str (g)+'.txt')
    #DATA = np.concatenate((DATA1,DATA2),0)

    DATA = np.loadtxt('/hadrex1/storage2/rramos/stratrack/machinelearning/samwise/merge0'+str(merge)+'/testset/XiCC_DataTestSet_Samwise_'+str(g)+'.txt')

    #Split the K features and the auxiliar variables
    X_test = DATA[:,:K]
    PID_test = DATA[:,K] 
    IM_test = DATA[:,K+1]
    pT_test = DATA[:,K+2]
    Y_test, N_test = PID_to_Class(PID_test)
    
    print('Num. de sinal test ', g, ':',Count_Particle(PID_test,1))
    print('Num. de fundo test ', g, ':', Count_Particle(PID_test,0))
   
    #Model Prediction
    #D_test = xgb.DMatrix(X_test)
    Y_pred = BDT.predict(X_test)

    PID_test = np.reshape(PID_test,(PID_test.shape[0],1))
    IM_test = np.reshape(IM_test,(IM_test.shape[0],1))
    pT_test = np.reshape(pT_test,(pT_test.shape[0],1))
    Y_pred = np.reshape(Y_pred,(Y_pred.shape[0],1))

    data_out = np.concatenate((PID_test,IM_test,pT_test,Y_pred),1)
    np.savetxt('/hadrex1/storage2/rramos/stratrack/machinelearning/samwise/merge0'+str(merge)+'/res0'+str(num)+'/PID_IM_pT_MLpred_bin'+str(g)+'.txt',data_out)

    #Plot output distribution in Valid Set
    N_Particles = Count_Particle(PID_test,MyPID1)
    N_BG = PID_test.shape[0] - N_Particles

    Class0 = np.ndarray(shape=(N_BG,1))
    Class1 = np.ndarray(shape=(N_Particles,1))

    i0 = 0
    i1 = 0

    for i in range(Y_pred.shape[0]):
    
      if (Y_test[i]==0):
        Class0[i0] = Y_pred[i]
        i0+=1

      else:
        Class1[i1] = Y_pred[i]
        i1+=1

    bins = np.linspace(0,1,100)

    fig = plt.figure(figsize=(8,8))
    ax1 = fig.add_subplot()
    ax1.set_ylabel('Normalized Counts', fontsize='xx-large')
    plt.xticks(fontsize='xx-large')
    #ax1.set_xlabel('BDT Output (P)', fontsize='xx-large')
    ax1.set_xlabel('ML(BDT) response', fontsize='xx-large')
    plt.yticks(fontsize='xx-large')

    weights0 = np.ones_like(Class0) / len(Class0)
    weights1 = np.ones_like(Class1) / len(Class1)

    plt.hist(Class0,bins,alpha=0.5,label='Background', density=True, color='steelblue')#
    plt.hist(Class1,bins,alpha=0.5, label='Signal', density=True, color='crimson')#
    #plt.title('BDT output: Valid Set',fontsize='xx-large')
    plt.title('ML output probability: Test Set '+str (pT_min[g])+'$\leq$ $p_T$ <'+str (pT_max[g])+'', fontsize='xx-large')
    plt.yscale('log')
    #plt.ylim([1,5e6])
    plt.xlim([0,1])
    #plt.ylim([,1])
    plt.legend(fontsize='xx-large')
    plt.savefig('/hadrex1/storage2/rramos/stratrack/machinelearning/samwise/merge0'+str(merge)+'/res0'+str(num)+'/ml_output_validset_'+str(num)+'_'+str(pT_min[g])+'_'+str(pT_max[g])+'.pdf', format='pdf')

# # In[ ]:


for g in range(num_pt):

	#--------------------------------------------------------------------------
	#NN

	DATA = np.loadtxt('/hadrex1/storage2/rramos/stratrack/machinelearning/samwise/merge0'+str(merge)+'/res0'+str(num)+'/PID_IM_pT_MLpred_bin'+str(g)+'.txt')
	print(DATA.shape)

	#Split the K features and the auxiliar variables
	PID_test = DATA[:,0] 
	IM_test = DATA[:,1]
	pT_test = DATA[:,2]
	Y_pred = DATA[:,3]

	#Fill C_thrs_pT and BG_thrsh_pT

	#Data loop
	for i in range(Y_pred.shape[0]):

		#thrsh loop
		for j in range(n):

			#IM cut
			if ( IM_test[i] < IM1 - sigma or IM_test[i] > IM1 + sigma ):
				continue

			#thrsh cut
			if (Y_pred[i]<t[j]):
				continue

			if (PID_test[i]==MyPID1):
				Sg_thrsh_pT[j][g]+=1

			else:
				Bg_thrsh_pT[j][g]+=1

	print(g,' Done!')
  
np.savetxt('/hadrex1/storage2/rramos/stratrack/machinelearning/samwise/merge0'+str(merge)+'/res0'+str(num)+'/Sg_thrsh_pT.txt', Sg_thrsh_pT, delimiter =" ")
np.savetxt('/hadrex1/storage2/rramos/stratrack/machinelearning/samwise/merge0'+str(merge)+'/res0'+str(num)+'/Bg_thrsh_pT.txt', Bg_thrsh_pT, delimiter =" ")
  
#aplica fator de escala

lCentralEventsPerMonth = 4e+9*35./5.6
lAcceptanceFactor = 4.0/1.5
SignalScaleFactor = 2./(178*5*(1/0.05)*(1/0.029))
BackgroundScaleFactor = 2.0
lNEventsSignal = np.sum(Sg_thrsh_pT, axis = 1)	
lNEventsBackground = np.sum(Bg_thrsh_pT, axis = 1)	

Sg_thrsh_pT = lCentralEventsPerMonth * lAcceptanceFactor * SignalScaleFactor * Sg_thrsh_pT
Bg_thrsh_pT = lCentralEventsPerMonth * lAcceptanceFactor * BackgroundScaleFactor * Bg_thrsh_pT

for i in range(n):
	Sg_thrsh_pT[i] = Sg_thrsh_pT[i] / lNEventsSignal[i]
	Bg_thrsh_pT[i] = Bg_thrsh_pT[i] / lNEventsBackground[i]

#calculo significancia

significance = np.zeros(shape=(n,num_pt))

for j in range(n):
	for g in range(num_pt):
		significance[j][g] = Sg_thrsh_pT[j][g] / np.sqrt(Sg_thrsh_pT[j][g] + Bg_thrsh_pT[j][g])  


for i in range(n):
	fig = plt.figure(figsize=(8,8))
	ax1 = fig.add_subplot()
	ax1.set_ylabel('Significance', fontsize='xx-large')
	plt.xticks(fontsize='xx-large')
	ax1.set_xlabel('$\leq$ $p_T$ (GeV/c)', fontsize='xx-large')
	plt.yticks(fontsize='xx-large')
	for j in range(num_pt):
		plt.hline(significance[i][j], pT_min[j], pT_max[j])
	#plt.title('BDT output: Valid Set',fontsize='xx-large')
	plt.title('Significance: Test Set, Thresh: '+str (t[i])+'', fontsize='xx-large')
	plt.yscale('log')
	#plt.ylim([1,5e6])
	plt.xlim([0,15])
	#plt.ylim([,1])
	plt.legend(fontsize='xx-large')
	plt.savefig('/hadrex1/storage2/rramos/stratrack/machinelearning/samwise/merge0'+str(merge)+'/res0'+str(num)+'/significance_testset_thresh_'+str(num)+'_'+str (t[i])+'.pdf', format='pdf')
'''

print("The time used to execute this is given below")

end = time.time()

print(end - start)

