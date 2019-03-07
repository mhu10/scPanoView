import numpy as np
import pandas as pd
import random
from sklearn import datasets
from sklearn.preprocessing import MinMaxScaler

def Skl_scale(data):
    newdata = data.copy()
    for i in newdata.index:
        scaler = MinMaxScaler(feature_range=(0,10000))
        scaler = scaler.fit(newdata.loc[i,:].values.reshape(len(newdata.columns),1))
        newdata.loc[i,:] = scaler.transform(newdata.loc[i,:].values.reshape(len(newdata.columns),1)).reshape(1,len(newdata.columns))
    return(newdata)
    

R_number = random.sample(range(1000), k=20) # random numners
for i in range(20):
    rnumber = R_number[i]
	for j in rnage(3,23): 
	blobs = datasets.make_blobs(n_samples=500,n_features=20000,random_state=rnumber,centers=j,cluster_std=1)
	inputdf = pd.DataFrame(data=blobs[0])
	blobs_cluster=pd.DataFrame(data=blobs[1],columns=['cluster']) # the ground truth 
	inputdf =Skl_scale(inputdf.transpose()) # expression data for simulation   
