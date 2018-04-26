import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from numpy import linalg as LA
from scipy.spatial import ConvexHull
from scipy import stats
from scipy.spatial import distance
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram
from scipy.cluster.hierarchy import fcluster
from scipy.cluster import hierarchy
from sklearn.decomposition import PCA
from sklearn.neighbors import BallTree
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
from collections import Counter
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans
import matplotlib as mpl
from statsmodels.sandbox.stats.multicomp import multipletests
np.random.seed(1)
    

##===========================================================================###
##================= Performing PCA  =========================================###
##===========================================================================## 

def RunPCA(data,n):

    pca = PCA(n_components=n)
    pca.fit(data)
    data_trans = pca.transform(data) ### new coordinates after pca transform
    return(data_trans,pca.explained_variance_ratio_)

def ARI(truth,predict):
    
    labels_true = truth
    labels_pred = predict
    
    return(metrics.adjusted_rand_score(labels_true, labels_pred))
    

####===========================================================================================================###  
####============  Select highly variance genes by zscore   =========================== ========================###
####===========================================================================================================###

def jitter(a_series, noise_reduction=1000000):
    return (np.random.random(len(a_series))*a_series.std()/noise_reduction)-(a_series.std()/(2*noise_reduction))


def HighVarGene(data,z,meangene):
    
    
    data=data.transpose()
    data=data.loc[:,(data!=0).any(axis=0)]
    data.loc[:,'average'] = np.mean(data,axis=1)     
    data.loc[:,'genegroup'] = pd.qcut(data.loc[:,'average'] + jitter(data.loc[:,'average']),20,labels=range(1,21)) 
    data.loc[:,'variance']= np.var(data.drop('genegroup',axis = 1),axis=1)
    data.loc[:,'dispersion']= data.drop('genegroup',axis =1).variance/data.drop('genegroup',axis=1).average

    pickdf=[]
    for group in range(1,21):
            
        
        data.loc[data[data.genegroup == group].index,'zscore'] = pd.DataFrame(stats.zscore(data[data.genegroup == group].dispersion)).fillna(0).values
        if len(data[(data.zscore>z)&(data.average>meangene)]) > 0:
            pickdf.append(data[(data.zscore>z)&(data.average>meangene)].index)
            
    if pickdf ==[]:
        for group in range(1,21):
            data.loc[data[data.genegroup == group].index,'zscore'] = pd.DataFrame(stats.zscore(data[data.genegroup == group].dispersion)).fillna(0).values
            if len(data[data.zscore>z]) >0:
                pickdf.append(data[(data.zscore>z)].index)
        
        if pickdf==[]:      
            return([])    
        else:
                
             hvg=(np.unique([g for sublist in pickdf for g in sublist]))
             
             if len(hvg) <3:
                 return([])
             else:
                 return(hvg)
 
    else:
        
        hvg=(np.unique([g for sublist in pickdf for g in sublist]))
    
        if len(hvg) < 3:
            
            return([])
    
        else:
            return(hvg)
            
            

######## Gini Coefficient ########

def gini(data):
    total=0
    for i in data:
        for j in data:
            
            total = total + abs(i-j)
    result = total / (2*len(data)*len(data)*np.mean(data))
    return(result)

    
#################################


def Skl_scale(data):
    ### using sklearn.MinMaxScaler
    newdata = data.copy()
    for i in newdata.index:
        scaler = MinMaxScaler(feature_range=(-2,2))
        scaler = scaler.fit(newdata.loc[i,:].values.reshape(len(newdata.columns),1))
        newdata.loc[i,:] = scaler.transform(newdata.loc[i,:].values.reshape(len(newdata.columns),1)).reshape(1,len(newdata.columns))
    return(newdata)




####===========================================================================================================###  
####============  ordering cell by number of neighbors   =========================== ========================###
####===========================================================================================================###


def OrderCell(data,radius):

    tree = BallTree(data,leaf_size=2)    
    Countnumber=[]
    for point in range(len(data)): 
        count = tree.query_radius(data[point].reshape(1,-1), r=radius, count_only = True) # counting the number of neighbors for each point
        Countnumber.append(count) # storing number of neighbors      
    CountnumberDf = pd.DataFrame(Countnumber,columns =['neighbors'])
    return(CountnumberDf)

###================================================================

def Distohull(xcoord,point,clusthull,clusthullvertices):
    
    disttovertices = [LA.norm(xcoord[i] - xcoord[point]) for i in clusthull.iloc[clusthullvertices,:].index ]
    if np.min(disttovertices) < np.mean(distance.pdist(xcoord[clusthull.iloc[clusthullvertices,:].index])):
       outcheck = 0 # assign to cluster
       outvalue = np.min(disttovertices)
    else:
        outcheck = 1 # not belong to cluster
        outvalue = 9999999999999
    return(outcheck,outvalue)

###==============================================================



def Findlocalmax(countdataframe,xcoordinate,bin):
    
    clusters = []
    newdataframe = countdataframe
    distohighest = [LA.norm(xcoordinate[point] - xcoordinate[np.argmax(newdataframe.neighbors)]) for point in newdataframe.index]
    newdataframe = newdataframe.assign(dist = distohighest)
    hist = np.histogram(distohighest,bin)
    firstclust = newdataframe[newdataframe.dist < hist[1][1]]
    
    clust_1 = list(firstclust.index)    
    if len(clust_1) < 4:
            return(False)
            
    hull = ConvexHull(xcoordinate[firstclust.index],qhull_options ='QJ')  
    clusters.append([firstclust,clust_1,hull])
    
    tempclust = newdataframe[newdataframe.dist >= hist[1][1]]
    checkpoint = len(tempclust)    
     
    while checkpoint > 0:
        
        checkpoint = 0
        neighbornumber = np.unique(tempclust.neighbors)[::-1]
        newcenter = []
        for i in neighbornumber:
            for cell in tempclust[tempclust.neighbors ==i].index:
                checkgroup = []
                checkvalue = []
                for k in range(len(clusters)):
                    check = Distohull(xcoordinate,cell,clusters[k][0],clusters[k][2].vertices)
                    checkgroup.append(check[0])
                    checkvalue.append(check[1])
                if 0 not in checkgroup:
                    newcenter.append(cell)
                    break
                else:
                    tempclust = tempclust.drop(cell)
                    clusters[np.argmin(checkvalue)][1].append(cell)
                    clusters[np.argmin(checkvalue)][0] = newdataframe.loc[clusters[np.argmin(checkvalue)][1],:]
                    clusters[np.argmin(checkvalue)][2] = ConvexHull(xcoordinate[clusters[np.argmin(checkvalue)][1]],qhull_options ='QJ')   
            
            if newcenter !=[]:
                break
            elif i == 1:
                return(clusters)
        
        distohighest = [LA.norm(xcoordinate[point] - xcoordinate[newcenter[0]]) for point in tempclust.index]
        tempclust=tempclust.assign(dist=distohighest)
        hist = np.histogram(distohighest,bin)
        cluster2 = tempclust[tempclust.dist < hist[1][1]]
        clust_2 = list(cluster2.index)
        if len(clust_2) < 4:
            return(clusters)
        else:
            hull2 = ConvexHull(xcoordinate[cluster2.index],qhull_options ='QJ')          
            clusters.append([cluster2,clust_2,hull2])
            tempclust = tempclust[tempclust.dist >= hist[1][1]]
            checkpoint = len(tempclust)                        
    return(clusters)
    


##### main function =================



class Panoite:
     
    def __init__(self,expression):
        
        self.expression = expression
        self.dfgene=[]
        self.membership = pd.DataFrame({'Cluster':0},index=list(range(len(self.expression))))
        self.stopite = False
    def generate_clusters(self,lowgene,zscore):
        
        CellMaximumn=5000
        ginicutoff = 0.05
        Rneighbor= 20
        maxcellibin=20
        maxbb = 20
        
        #print('gene',lowgene,'z',zscore,'gini',ginicutoff,'Rneighbor',Rneighbor,'maxbin',maxcellibin)
        findvarg = HighVarGene(self.expression,zscore,lowgene)
        
        if len(findvarg) > 0:
            self.dfgene.append(findvarg)
            subdf = self.expression.loc[:,self.dfgene[-1]]
        
        elif len(findvarg) == 0:
            self.stopite = True
            return()
            
        
        subdf=Skl_scale(subdf)
        self.pca = RunPCA(subdf.as_matrix().astype(float),3)[0]
        Radius = np.histogram(distance.pdist(self.pca),Rneighbor)[1][1]
       
        temppca = self.pca     
        Ordercell = OrderCell(temppca,Radius)
        
        bb=0
        opt_bins=True
        lm_number = 1
        
        while opt_bins == True:
            bb = bb+1
            
            if len(temppca) >CellMaximumn:
                print('cell > 5000')
                #localmax = FindlocalmaxLarge(Ordercell,temppca)
                localmax = Findlocalmax(Ordercell,temppca,maxcellibin)
                opt_bins = False
            
            else:
                localmax = Findlocalmax(Ordercell,temppca,5*bb)
            
            if localmax == False:
                    
                    #print('find LM1 failed',bb)
                    if bb == 1:
                        
                        self.membership.Cluster= 1
                        self.CSIZE = len(self.membership)
                        return()             
                    else:
                        localmax = Findlocalmax(Ordercell,temppca,5*(bb-1))
                        opt_bins = False                           
                    
            lm_number_next = len(localmax)
            
            if bb > maxbb and len(temppca) <CellMaximumn:
                opt_bins = False
            elif lm_number_next >= lm_number:
               lm_number = lm_number_next
            
            elif lm_number_next < lm_number and len(temppca) <CellMaximumn :
                localmax = Findlocalmax(Ordercell,temppca,5*(bb-1))
                
                opt_bins = False
        
        
        densepoint = []
        for i in range(len(localmax)):
            densepoint.append(np.argmax(localmax[i][0].neighbors))
        
        
        for j in Ordercell.index:
    
            pairdist = distance.cdist([temppca[j]],temppca[densepoint])
            Ordercell.loc[j,'cluster'] = np.argmin(pairdist)+np.max(self.membership.Cluster)+1
        
                 
        Ordercell.cluster = Ordercell.cluster.astype(int)
        
        
        for i in Ordercell.index:
            self.membership.loc[i,'Cluster'] = Ordercell.loc[i,'cluster']      

############ plot eva ########################################
                  
        Eva=[]
        Cluster_size=[]        
        check_CLN = []
        
        
        for i in np.unique(Ordercell.cluster):
            
            Eva.append(np.var(distance.pdist(self.expression.loc[self.membership[self.membership.Cluster ==i].index,self.dfgene[-1]],'correlation')))
            check_CLN.append(i)
        
        
        for i in np.unique(Ordercell.cluster):
            Cluster_size.append(len(self.membership[self.membership.Cluster == i]))
        
        Eva=pd.DataFrame(Eva,index = check_CLN).fillna(0)
        
        
 ############ plot eva ########################################       
        Cluster_size=pd.DataFrame(Cluster_size,index = check_CLN)
        self.eva=Eva
        self.Cluster_size = Cluster_size
        
        cluster_size = self.Cluster_size.loc[self.eva.idxmin(),:].values
        self.CSIZE = cluster_size

        
############ plot gini ########################################
        
        tempEva = np.copy(Eva)
        
        tempEva.sort(axis=0)
        
        ginivalue = []
        
        for j in range(len(Eva)):
            accum = 0
            lorenz=[0]
            
            if j == 0:
                pass
            else:
                   
                Evalist = tempEva[0:j+1]
                for i in Evalist:
                    persent = 100*(i / sum(Evalist))
                    accum = accum + persent
                    lorenz.append(accum)
                ginivalue.append(gini(Evalist))


        Gini=pd.DataFrame(ginivalue)
        self.gin=Gini
        self.rempeva=tempEva

        
########################################################################################################


        self.ordercell = Ordercell        
        
        check_iteation = any(Gini > ginicutoff)

    
        while check_iteation == True:
        
            if all(Gini > ginicutoff):
                
                clust_pick_auto = list(set(np.unique(Ordercell.cluster)))
                
                clust_keep_auto = self.eva.idxmin().values
                
                
                if any(self.Cluster_size.values >= cluster_size):
                    clust_size_cut = self.Cluster_size[self.Cluster_size.values <= cluster_size].index.tolist()
                    
                    
                    clust_pick_auto = list(set(clust_pick_auto) - set(np.concatenate((clust_keep_auto,clust_size_cut)).tolist()))
                else:
                    clust_pick_auto.remove(clust_keep_auto)
                     
                
            
                
                
            ################# plot clusters##########################

                if clust_pick_auto == []:                                 
                    return('serching done 1')
                    
                    
            else:
                clust_pick_auto = list(set(np.unique(Ordercell.cluster)))               
                numberforkeep = self.gin[self.gin.values <= ginicutoff].index.min()+2
                clust_keep_auto = self.eva.sort_values(by=0).index[:numberforkeep].tolist()
                
                
                if any(self.Cluster_size.values >= cluster_size):
                    clust_size_cut = self.Cluster_size[self.Cluster_size.values <= cluster_size].index.tolist()
                    clust_pick_auto = list(set(clust_pick_auto) - set(np.concatenate((clust_keep_auto,clust_size_cut)).tolist()))
                    
                else:
                    for item in clust_keep_auto:                        
                        clust_pick_auto.remove(item)
                        
            

            ################# plot clusters##########################
                
                if clust_pick_auto == []:
                    
                    return('serching done 2')
            
            nextdf = self.expression.loc[self.membership[self.membership.Cluster.isin(clust_pick_auto)].index,:]

            
            
            findvarg = HighVarGene(self.expression,zscore,lowgene)
            if len(findvarg) > 0:
            
                self.dfgene.append(findvarg)
                subdf = self.expression.loc[:,self.dfgene[-1]]
        
            elif len(findvarg) == 0:
                
                self.stopite = True
                return()
                
            
            
            
            
            subdf=Skl_scale(subdf)
            self.pca = RunPCA(subdf.as_matrix().astype(float),3)[0]
        
            Radius = np.histogram(distance.pdist(self.pca),Rneighbor)[1][1]
            
            
            
            temppca = self.pca     
            Ordercell = OrderCell(temppca,Radius)

            bb=0
            opt_bins=True
            lm_number = 1
            while opt_bins == True:
                bb = bb+1
                
                if len(temppca) >CellMaximumn:
                    print('cell > 5000')
                    #localmax = FindlocalmaxLarge(Ordercell,temppca)
                    localmax = Findlocalmax(Ordercell,temppca,maxcellibin)
                    opt_bins = False                         
                else:
                    localmax = Findlocalmax(Ordercell,temppca,5*bb)
                
                if localmax == False:
                    
                    #print('find LM2 failed',bb)
                    if bb == 1:
                        
                        self.membership.loc[nextdf.index,'Cluster'] = np.max(self.membership.Cluster)+1
                        #print('serching done 9999')
                        return('serching done 9999')                      

                    else:
                        localmax = Findlocalmax(Ordercell,temppca,5*(bb-1))
                        opt_bins = False                           
                
                
                lm_number_next = len(localmax)
             
                
                if bb > maxbb and len(temppca) <CellMaximumn:
                    opt_bins = False
            
                elif lm_number_next >= lm_number:
                    lm_number = lm_number_next
            
                elif lm_number_next < lm_number and len(temppca) <CellMaximumn:
                    localmax = Findlocalmax(Ordercell,temppca,5*(bb-1))           
                    opt_bins = False                
        
            densepoint = []
            for i in range(len(localmax)):
                densepoint.append(np.argmax(localmax[i][0].neighbors))
         
            for j in Ordercell.index:
    
                pairdist = distance.cdist([temppca[j]],temppca[densepoint])
                Ordercell.loc[j,'cluster'] = np.argmin(pairdist)+np.max(self.membership.Cluster)+1
                       
            Ordercell.cluster = Ordercell.cluster.astype(int)
            
            cellid_pick = self.membership[self.membership.Cluster.isin(clust_pick_auto)].index
            
            for i in Ordercell.index:
                self.membership.loc[cellid_pick[i],'Cluster'] = Ordercell.loc[i,'cluster']               
            

            
##########################################################################################            
            Eva=[]
            Cluster_size=[]
            check_CLN = [] 
            for i in np.unique(Ordercell.cluster):
                Eva.append(np.var(distance.pdist(self.expression.loc[self.membership[self.membership.Cluster ==i].index,self.dfgene[-1]],'correlation')))
                check_CLN.append(i)
            for i in np.unique(Ordercell.cluster):
                Cluster_size.append(len(self.membership[self.membership.Cluster == i]))            



############ plot eva ########################################       
 
            Eva=pd.DataFrame(Eva,index = check_CLN).fillna(0)
            Cluster_size=pd.DataFrame(Cluster_size,index = check_CLN)
        
            self.eva=Eva
            self.Cluster_size = Cluster_size

            
############ plot gini ########################################

            tempEva = np.copy(Eva)
            tempEva.sort(axis=0)
            ginivalue = []
        
            for j in range(len(Eva)):
                accum = 0
                lorenz=[0]
            
                if j == 0:

                   pass        
                else:
                   
                   Evalist = tempEva[0:j+1]
                   for i in Evalist:
                       persent = 100*(i / sum(Evalist))
                       accum = accum + persent
                       lorenz.append(accum)
                   ginivalue.append(gini(Evalist))

            
            Gini=pd.DataFrame(ginivalue)
            self.gin=Gini
            self.rempeva=tempEva      
            
            check_iteation = any(Gini > ginicutoff)
            



class PanoVIEW:
     
    def __init__(self,expression):
        
        self.raw_exp = expression
        self.log_exp =[]
        self.cell_id =[]
        self.vargene =[]
        self.cell_clusters=[]        
        self.cell_membership=[]
        self.hlinkage_matrix=[]
        self.sim_matrix=[]
        self.tsne2d=np.array([])
        self.clust_dendro=[]
        self.cell_color = []  
        self.cell_dendro_order = []
        self.deg_stat=[]
    def RunFilter(self,gene,cell):
        
        genekeep = self.raw_exp.loc[(self.raw_exp!=0).sum(axis=1)> cell,:].index
        cellkeep = self.raw_exp.loc[:,(self.raw_exp!=0).sum()> gene].columns
        
        self.raw_exp = self.raw_exp.loc[genekeep,cellkeep]
        
        
    def RunPanoView(self,Gnormal=True,Log2=True,GeneLow=None,Zscore=None):
        
        
        self.raw_exp.index.astype(str)
        self.raw_exp.index = self.raw_exp.index.where(~self.raw_exp.index.duplicated(), self.raw_exp.index + '_dp')        
        
        if Gnormal == False:
            if Log2 == False:
                self.log_exp = self.raw_exp.transpose()
                expression = self.log_exp.loc[:,(self.log_exp!=0).any(axis=0)]
            else:
                self.log_exp = np.log2(1+self.raw_exp.transpose())
                expression = self.log_exp.loc[:,(self.log_exp!=0).any(axis=0)]
        else:
            
            self.raw_exp=self.raw_exp.loc[(self.raw_exp!=0).any(axis=1),:]
            raw_norm = pd.DataFrame(normalize(self.raw_exp,norm='l1',axis=0)*self.raw_exp.sum().mean(),index=self.raw_exp.index,columns=self.raw_exp.columns)
            self.log_exp = np.log2(1+raw_norm.transpose())
            expression = self.log_exp
            
        self.cell_id = expression.index
        self.gene_id = expression.columns
        
        expression.index=range(len(expression))
        VarGene=[]
        
        if GeneLow != None:
            GeneLow = GeneLow
        else:
            GeneLow = 0.5
        
        if Zscore != None:
            Zscore = Zscore
        else:
            Zscore = 1.5
            
        result = Panoite(expression)
        result.generate_clusters(GeneLow,Zscore)
        
        VarGene.append(np.unique([gene for sublist in result.dfgene for gene in sublist]))
        
        finalcluster=[]
        smallcluster=[]
    
        for i in np.unique(result.membership.Cluster):
            if np.sum(result.membership.Cluster ==i) >= result.CSIZE:
                finalcluster.append(result.membership[result.membership.Cluster ==i].index.values)
            else:
                smallcluster.append(result.membership[result.membership.Cluster ==i].index.values)
    
        nextsmall=[item for sublist in smallcluster for item in sublist]    
        nextsmall_id = pd.DataFrame(data=nextsmall,columns=['cellid'])

        if len(nextsmall_id.cellid) == 0:
        
            self.vargene= np.unique([gene for sublist in VarGene for gene in sublist])
            self.cell_clusters = finalcluster
                          
                        
            return(print("RunPanoView-Done"))
            
        checkite = True    

        
        while checkite == True:
            
            nextdf = expression.loc[nextsmall_id.cellid,:]
            nextdf.index=range(len(nextdf))
            
            result2 = Panoite(nextdf)
            result2.generate_clusters(GeneLow,Zscore)
            
            
            if result2.stopite == True:
                
                finalcluster.append(nextsmall_id.cellid.values)
                self.cell_clusters = finalcluster
                self.vargene= np.unique([gene for sublist in VarGene for gene in sublist])
                return("RunPanoView-Done")
                
            VarGene.append(np.unique([gene for sublist in result2.dfgene for gene in sublist]))        
    
        
            bigcluster=[]
            smallcluster=[]


            for i in np.unique(result2.membership.Cluster):
                if np.sum(result2.membership.Cluster ==i) >= result2.CSIZE:
                    bigones = result2.membership[result2.membership.Cluster ==i].index
                    bigcluster.append(bigones)
                         
                    finalcluster.append(nextsmall_id.iloc[bigones,:].cellid.values)
    
                else:
            
                    smallcluster.append(result2.membership[result2.membership.Cluster ==i].index)
        
            nextsmall=[item for sublist in smallcluster for item in sublist]


            if len(nextsmall) < 1 or bigcluster==[]:
            
                for i in range(len(smallcluster)):
                    finalcluster.append(nextsmall_id.iloc[smallcluster[i],:].cellid.values)
            
                self.vargene= np.unique([gene for sublist in VarGene for gene in sublist])
                self.cell_clusters = finalcluster
                
                return("RunPanoView-Done")

            else:
                nextsmall_id = nextsmall_id.iloc[nextsmall,:]
            
            
    def RunHierarchy(self,R=None,metric_dis=1):
        
        
        if R != None:
            R=R;
        else:
            R=0.1
        
        
        expression= self.log_exp
        expression.index=range(len(expression))
        expression = Skl_scale(expression)        
        cluster_list=[]
        for i in range(len(self.cell_clusters)):
            cluster_list.append(expression.loc[self.cell_clusters[i],self.vargene])            
            
        sim_mat = pd.DataFrame(0,index=np.arange(1,len(cluster_list)+1),columns=np.arange(1,len(cluster_list)+1))        
        for i in sim_mat.index:
            ci = cluster_list[i-1]
            for j in sim_mat.index:
                cj = cluster_list[j-1]
                if metric_dis == 1:
                    sim_mat.loc[i,j] = np.mean(distance.cdist(ci,cj,metric='correlation'))
                elif metric_dis ==2:
                    sim_mat.loc[i,j] = np.mean(distance.cdist(ci,cj,metric='euclidean'))
                    
        clustdis_within = pd.DataFrame(np.diagonal(sim_mat),index=sim_mat.index,columns=['dis'])
        
        #self.SIMMAT=sim_mat
        linkage_matrix = linkage(sim_mat,method="ward")
        cutree = pd.DataFrame(data=hierarchy.cut_tree(linkage_matrix , n_clusters=None, height=None)+1,index=range(1,len(cluster_list)+1))  
        
        init_lenth=0
        Lgroups=[]
        MergeCluster=[]
    
        for step in range(1,len(cutree.columns)):
        
            repeats=[item for item, count in Counter(cutree[step]).items() if count > 1]
            check_lenth = len(repeats)
        
            lgroups=[]
            for i in repeats:
                lgroups.append(cutree[step][cutree[step]==i].index.tolist())
            Lgroups.append(lgroups)
            
            if check_lenth > init_lenth:
                cgroups=[]
                for i in repeats:
                    cgroups.append(cutree[step][cutree[step]==i].index.tolist())
            
                for g in cgroups:
                    if len(g) ==2:
                        pairdis = sim_mat.loc[g[0],g[1]]
                    
                        if abs(pairdis - clustdis_within.loc[g[0]].dis) <= clustdis_within.min().dis*R or abs(pairdis - clustdis_within.loc[g[1]].dis) <= clustdis_within.min().dis*R:                                        
                            if g not in MergeCluster:                    
                                MergeCluster.append(g)            
            
            elif check_lenth < init_lenth:
                break
            else:
                clust=[i for i in Lgroups[step-1] if i not in Lgroups[step-2]]
                clust=[i for sublist in clust for i in sublist]
                clust_to_merge = [s for s in MergeCluster if bool(set(s) & set(clust))]
                clust_to_merge=[i for sublist in clust_to_merge for i in sublist]
                clust_add = [s for s in clust if s not in clust_to_merge][0]
            
                for c in clust_to_merge:
                
                    pairdis = sim_mat.loc[c,clust_add]
               
                    if abs(pairdis - clustdis_within.loc[c].dis) <= clustdis_within.min().dis*R or abs(pairdis - clustdis_within.loc[clust_add].dis) <= clustdis_within.min().dis*R:
                                   
                        MergeCluster.remove(clust_to_merge)
                        clust_to_merge.append(clust_add)
                        MergeCluster.append(clust_to_merge)
                        break
            init_lenth = check_lenth
        
        mergelist = MergeCluster
            
        if len(mergelist) !=0:
            for i in mergelist:
                frame = [cluster_list[s-1] for s in i ]
                cluster_list.append(pd.concat(frame))
            
            rmlist = [item for sublist in mergelist for item in sublist]
            cluster_list_new = [v for i, v in enumerate(cluster_list) if i+1 not in rmlist]
            
            sim_mat = pd.DataFrame(0,index=np.arange(1,len(cluster_list_new)+1),columns=np.arange(1,len(cluster_list_new)+1))        
            for i in sim_mat.index:
                ci = cluster_list_new[i-1]
                for j in sim_mat.index:                
                    cj = cluster_list_new[j-1]
                    
                    if metric_dis == 1:
                        sim_mat.loc[i,j] = np.mean(distance.cdist(ci,cj,metric='correlation'))
                    elif metric_dis ==2:
                        sim_mat.loc[i,j] = np.mean(distance.cdist(ci,cj,metric='euclidean'))
                        
            cluster_list = cluster_list_new
        
        #self.SIMMAT=sim_mat
        linkage_matrix = linkage(sim_mat,method="ward")
     
        
        #plt.figure(figsize=(10,8))
        #dendrogram(linkage_matrix,distance_sort='descending',leaf_font_size=24,labels=sim_mat.index,color_threshold=0.1)
        dn2=dendrogram(linkage_matrix,distance_sort='descending',leaf_font_size=24,labels=sim_mat.index,color_threshold=0.1,no_plot=True)
        #plt.savefig('Cluster_Hierarchy',dpi=200)
        #plt.show()
        
    
        dfheat_order=[]
        for i in dn2['leaves']:
            dfheat_order.append(cluster_list[i])
    
        membership = pd.DataFrame({'Cluster':0},index=list(range(len(expression))))   
        for i in range(len(dfheat_order)):
            membership.loc[dfheat_order[i].index,'Cluster'] = dn2['leaves'][i]+1
        
        self.clust_dendro=dn2['leaves']
        
        membership=membership.astype(int)
        membership.loc[:,'Cell_ID'] = self.cell_id
        self.cell_membership = membership
        self.hlinkage_matrix=linkage_matrix
        self.sim_matrix=sim_mat
        
        
        
        colormaps = sns.color_palette("hls", len(np.unique(self.cell_membership.Cluster)))
        cellgroup=[]
        heatcolor = []
        cluster_color=[]
        for i in self.clust_dendro:
            cellgroup.append(self.cell_membership[self.cell_membership.Cluster == (i+1)].index)
            cluster_color.append([(i+1),colormaps[i]])
            for j in range(len(self.cell_membership[self.cell_membership.Cluster == (i+1)].index)):    
                    heatcolor.append(colormaps[i])
        
        
        plt.figure(figsize=(10,8))
        dendrogram(linkage_matrix,distance_sort='descending',leaf_font_size=24,labels=sim_mat.index,color_threshold=0.01,above_threshold_color='grey')
        ax = plt.gca()
        xlbls = ax.get_xmajorticklabels()
        c=0
        for lbl in xlbls:
            lbl.set_color(cluster_color[c][1])
            c=c+1
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        #ax.spines['left'].set_visible(False)
        
        plt.savefig('Cluster_Hierarchy',dpi=200)
        plt.show()
        
        self.cell_color = heatcolor  
        self.cell_dendro_order = [item for sublist in cellgroup for item in sublist]
        self.cluster_color=cluster_color
        
        
        
        
        return('Output hierarchy')
    


    def RunTSNE(self,fclust_dis= None, init='pca',n_PCs=10):
    
        if fclust_dis != None:
        
            assignments = fcluster(self.hlinkage_matrix,fclust_dis,'distance')
            Final_cluster = pd.DataFrame({'cluster':assignments})
            Final_cluster.index=self.sim_matrix.index
    
            for i in Final_cluster.index:
                    self.cell_membership.loc[self.cell_membership[self.cell_membership.Cluster ==i].index,'Fcluster'] = Final_cluster.loc[i,'cluster']
            self.cell_membership.loc[:,'Fcluster'] =self.cell_membership.loc[:,'Fcluster'].astype(int)
            
        elif fclust_dis == None and len(self.cell_membership.columns)<3:
            self.cell_membership.loc[:,'Fcluster']=self.cell_membership.loc[:,'Cluster']

        
        expression=self.log_exp
        
        result=self.cell_membership
        tsnedata= Skl_scale(expression.loc[:,self.vargene])
        
        tmodel = TSNE(n_components=2,random_state=1,init=init)
        
        
        
        tsnedata = RunPCA(tsnedata,n_PCs)[0]
        
        if self.tsne2d.size == 0  or n_PCs !=10:
            tcoord = tmodel.fit_transform(tsnedata)
            self.tsne2d = tcoord
        elif self.tsne2d.size != 0:
            tcoord=self.tsne2d
        
        
        plt.figure(figsize=(16,8))
        sns.set(style="white")
        plt.subplot(1,2,1)
        cluster_colors = sns.color_palette("hls", len(np.unique(result.Cluster)))
        j=0
        for i in np.unique(np.unique(result.Cluster)):
        
            plt.scatter(tcoord[result[result.Cluster ==i].index,0],tcoord[result[result.Cluster==i].index,1],color=cluster_colors[j],s=40,label=i)
            j=j+1
        plt.legend(loc='upper left', prop={'size':16}, bbox_to_anchor=(0.99,1),ncol=1)
        plt.grid()
        plt.title('Cluster',fontsize=18)
        plt.xticks([])
        plt.yticks([])
        

        plt.subplot(1,2,2)
        cluster_colors = sns.color_palette("hls", len(np.unique(result.Fcluster)))
        j=0
        for i in np.unique(np.unique(result.Fcluster)):
            plt.scatter(tcoord[result[result.Fcluster ==i].index,0],tcoord[result[result.Fcluster==i].index,1],color=cluster_colors[j],s=40,label=i)
            j=j+1
        plt.legend(loc='upper left', prop={'size':16}, bbox_to_anchor=(0.99,1),ncol=1)
        plt.grid()
        plt.title('Fcluster',fontsize=18)
        plt.xticks([])
        plt.yticks([])
        plt.savefig('Cluster_TSNE',dpi=200)
        plt.show()
        
        self.cell_membership=self.cell_membership[['Cell_ID','Cluster','Fcluster']]
        self.cell_membership.to_excel('Cell_Membership.xlsx')
        
    
    def VisCluster(self,cluster_number):
        
        result=self.cell_membership
        tcoord=self.tsne2d
        plt.figure(figsize=(10,10))
        ## marker expression ##############################
        
        for i in np.unique(np.unique(result.Cluster)):
            if i == cluster_number:
                plt.scatter(tcoord[result[result.Cluster ==i].index,0],tcoord[result[result.Cluster==i].index,1],color='b',s=40,label=i)
            else:
                plt.scatter(tcoord[result[result.Cluster ==i].index,0],tcoord[result[result.Cluster==i].index,1],color='gray',s=40,label=i)
            
        plt.legend(loc='upper left', prop={'size':16}, bbox_to_anchor=(0.99,1),ncol=1)

    
    def VisGeneExp(self,genes):
        
        
        ## marker expression ##############################
        markers=genes
        markerdata = self.log_exp
        markerdata = normalize(markerdata,norm='max') ## normalization
        markerdata=pd.DataFrame(markerdata,index=self.log_exp.index ,columns=self.log_exp.columns)
        
        for i in range(len(markers)):
            plt.figure(figsize=(10, 10))
            plt.suptitle(markers[i],fontsize=36)
            plt.scatter(self.tsne2d[:,0],self.tsne2d[:,1],c=markerdata.loc[:,markers[i]],s=40,cmap='BuPu',edgecolor='gray',alpha=0.5)
            plt.show()
            
            #plt.savefig('marker.png',dpi=200)
        
    
    def VisClusterUser(self,cluster_inform):
        
        cluster_id = np.unique(cluster_inform)
        tcoord=self.tsne2d
        plt.figure(figsize=(10,10))
        sns.set(style="white")
        cluster_colors = sns.color_palette("hls", len(cluster_id))
        j=0
        for i in cluster_id:
            plt.scatter(tcoord[cluster_inform[cluster_inform ==i].index,0],tcoord[cluster_inform[cluster_inform==i].index,1],color=cluster_colors[j],s=40,label=i)
            j=j+1
        
        plt.legend(loc='upper right', prop={'size':16}, bbox_to_anchor=(0.99,1),ncol=1)
        #plt.legend(fontsize=16)
        plt.grid()
        plt.xticks([])
        plt.yticks([])
        #plt.savefig('Cluster_TSNE',dpi=200)
        plt.show()
    
    def HeatMap(self,number,fd):
        
        topgene = self.deg_stat.query("Padj <0.01 & log2FD >@fd").sort_values(by='Padj')[:number].gene
        df = self.log_exp.loc[self.cell_dendro_order,topgene]
        linkage_matrix = linkage(df.T,method='ward')
        dn=dendrogram(linkage_matrix,show_leaf_counts=True,orientation='left',no_plot=True)
        df = df.iloc[:,dn['leaves']]
        sns.set(style="white")                   
        FigHeat = plt.figure(figsize=(10,10))
        plt.tight_layout()
        ax = FigHeat.add_subplot(111)
        cax = ax.matshow(df,aspect='auto',cmap='BuPu')
        cbr=plt.colorbar(cax,fraction=0.02, pad=0.05)
        cbr.ax.set_title('$\log2$',fontsize=10)
        cbr.outline.set_visible(False)
        
        ax.set_xticks(range(number))
        ax.set_xticklabels(df.columns,fontsize=12,rotation=90)
        ax.set_yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)

        axbar = FigHeat.add_axes([0.1, 0.11, 0.02, 0.770])   
        cmap1 = mpl.colors.ListedColormap(self.cell_color[::-1])
        cbar=mpl.colorbar.ColorbarBase(axbar,cmap=cmap1, orientation='vertical',ticks=[])
        cbar.outline.set_visible(False)
        
        
    def RunDEGs(self):
        
        pvalue=[]
        logFD=[]
        datafordeg = self.log_exp.loc[:,(self.log_exp!=0).any(axis=0)]
        for i in datafordeg.columns:
            groups=[]
            fdgroups=[]
            for j in np.unique(self.cell_membership.Cluster):
                groups.append(datafordeg.loc[self.cell_membership[self.cell_membership.Cluster == j].index,i])
                fdgroups.append(datafordeg.loc[self.cell_membership[self.cell_membership.Cluster == j].index,i].mean())
            if max(fdgroups)-min(fdgroups) >=1:
                pvalue.append([i,stats.kruskal(*groups)[1]])
                logFD.append(max(fdgroups)-min(fdgroups))
        
           
        DEGstat=pd.DataFrame(pvalue,columns=['gene','pvalue']).fillna(1)
        p_value_adj=multipletests(DEGstat.iloc[:,1],alpha=0.05,method='fdr_bh')
        DEGstat.loc[:,'Padj'] = p_value_adj[1]
        DEGstat.loc[:,'log2FD'] = logFD
        self.deg_stat = DEGstat
        
        
    def RunKmeans(self,numberK):
        
        
        tcoord=self.tsne2d
        
        km = KMeans(n_clusters=numberK, init = 'random',random_state=1)
        km.fit(tcoord)
    
        PKresults = pd.DataFrame(data=km.labels_,columns=['cluster'])
    
    
        plt.figure(figsize=(10,10))
        cluster_colors = sns.color_palette("hls", len(np.unique(PKresults.cluster)))
        
        
        j=0
        for i in np.unique(PKresults.cluster):
            
            plt.scatter(tcoord[PKresults[PKresults.cluster ==i].index,0],tcoord[PKresults[PKresults.cluster ==i].index,1],color=cluster_colors[j],s=40,label=i)
            j=j+1
        
        plt.legend(prop={'size':16},ncol=1)
        plt.grid()
        plt.xticks([])
        plt.yticks([])
        #plt.savefig('Cluster_TSNE',dpi=200)
        plt.show()

