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
    


R_number = random.sample(range(1000), k=20)

for r in R_number:
    for j in range(3,16):
      
        blobs = datasets.make_blobs(n_samples=500,n_features=20000,random_state=r,centers=j,cluster_std=1)        
        blobs_cluster=pd.DataFrame(data=blobs[1],columns=['cluster'])

        inputdf = pd.DataFrame(data=blobs[0])
        inputdf=Skl_scale(data)(inputdf.transpose())

        clusternumber = random.sample(range(j), k=1)
        clustersize = len(blobs_cluster[blobs_cluster.cluster == clusternumber])         
        randomnumber = random.sample(range(clustersize), k=round(clustersize*0.9))
        RandomCell = blobs_cluster[blobs_cluster.cluster == clusternumber].index[randomnumber]

        inputdf2=inputdf.drop(labels=RandomCell,axis=1)
        inputdf2.columns=range(len(inputdf2.columns)) # expression data for the simulation of rare cells
        
        blobs_cluster2 = blobs_cluster.drop(labels=RandomCell,axis=0)           
        blobs_cluster2.index=range(len(blobs_cluster2.index))
        blobs_cluster2=blobs_cluster2.replace(to_replace=clusternumber,value=999) # ground truth
