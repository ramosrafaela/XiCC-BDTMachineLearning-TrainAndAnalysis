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
import pandas as pd 

#Metrics
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, explained_variance_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error


from array import array



import joblib
import time

import ROOT


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
lPtBins = np.array([0.,2.,4.,6.,8.,10.,15.])
lNPtBins = 6
print(pT_min)
print(pT_max)

#Standard Deviation of the peak arround IM1
sigma = 0.4
num = 2
num2 = 1
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


#here I rename the variables to identify it

BDT = xgb.Booster()
BDT.load_model('/hadrex1/storage2/rramos/stratrack/machinelearning/skywalker/data_train02/merge01/lowpt/systanalysis/train02/XiCC_BDT_Recipe_'+str(num)+'.json')
#BDT.get_booster().feature_names
orig_feature_names = ['fXiDCAxyToPV' , 'fXiccDCAxyToPV', 'fXiDCAzToPV', 'fXicPionDCAxyToPV1', 'fXicPionDCAzToPV1', 'fXicPionDCAxyToPV2', 'fXicPionDCAzToPV2', 'fXiccDCAzToPV', 'fPiccDCAzToPV', 'fXicDaughterDCA', 'fXiccDaughterDCA', 'fXicDecayRadius', 'fXiCCtoXiCLength', 'fXicDCAxyToPV', 'fXicDecayDistanceFromPV', 'fXiDecayLength', 'fPiccDCAxyToPV', 'fXiccDecayDistanceFromPV', 'fXiccDecayRadius', 'fXicDCAzToPV', 'fXiHitsAdded', 'fXiDecayRadius', 'fV0DecayRadius', 'fV0DCAxyToPV', 'fV0DCAzToPV', 'fNegativeDCAz', 'fV0DecayLength', 'fXiCtoXiLength', 'fXiDauDCA', 'fNegativeDCAxy', 'fBachelorDCAz', 'fPositiveDCAz', 'fPositiveDCAxy', 'fV0DauDCA', 'fBachelorDCAxy']
BDT.feature_names = orig_feature_names

for data in pd.read_csv('/hadrex1/storage2/rramos/stratrack/machinelearning/skywalker/data_test01/background_serial/XiCC_Data_Skywalker_Bg01_merged.txt', delim_whitespace=True, skipinitialspace=True, chunksize = 5e7):

	DATA = data.to_numpy()

	#Split the K features and the auxiliar variables
	X_test = DATA[:,:K]
	PID_test = DATA[:,K] 
	IM_test = DATA[:,K+1]
	pT_test = DATA[:,K+2]

	Y_test, N_test = PID_to_Class(PID_test)

	num_signal = num_signal + Count_Particle(PID_test,1)
	num_background = num_background + Count_Particle(PID_test,0)	
  
	#Model Prediction
	D_test = xgb.DMatrix(X_test, feature_names = orig_feature_names)
	Y_pred = BDT.predict(D_test)

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
# roc curve for models
fpr1, tpr1, thresh1 = roc_curve(Y_test, Y_pred[:,1], pos_label=1)
# roc curve for tpr = fpr 
random_probs = [0 for i in range(len(Y_test))]
p_fpr, p_tpr = roc_curve(Y_test, random_probs, pos_label=1)

# auc scores
auc_score1 = roc_auc_score(Y_test, Y_pred[:,1])
print(auc_score1)

plt.style.use('seaborn')

# plot roc curves
plt.plot(fpr1, tpr1, linestyle='--',color='orange', label='BDT')
plt.plot(p_fpr, p_tpr, linestyle='--', color='blue')
# title
plt.title('ROC curve')
# x label
plt.xlabel('False Positive Rate')
# y label
plt.ylabel('True Positive rate')

plt.legend(loc='best')
plt.savefig('/hadrex1/storage2/rramos/stratrack/machinelearning/skywalker/data_train02/merge01/lowpt/res03/ROC.pdf', format='pdf')

#mean squared error

errors = list()
for i in range(len(Y_test)):
	# calculate error
	err = (Y_test[i] - Y_pred[i])**2
	# store error
	errors.append(err)
# plot errors
plt.plot(errors)
plt.xticks(ticks=[i for i in range(len(errors))], labels=predicted)
plt.xlabel('Predicted Value')
plt.ylabel('Mean Squared Error')
plt.savefig('/hadrex1/storage2/rramos/stratrack/machinelearning/skywalker/data_train02/merge01/lowpt/res03/RMS.pdf', format='pdf')

# calculate errors
errors_MSE = mean_squared_error(Y_test, Y_pred)
# report error
print('Mean Squared Error:')
print(errors_MSE)

# Root Mean Squared Error
errors_RMSE = mean_squared_error(expected, predicted, squared=False)
print('Root Mean Squared Error:')
print(errors_RMSE)
