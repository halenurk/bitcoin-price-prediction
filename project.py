# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 11:48:45 2017

@author: hale nur & emine
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import neighbors, svm
from sklearn.mixture import GaussianMixture

#bitcoin dataframe
bitcoindf = pd.read_csv('bitcoin_dataset.csv', sep=',', header=None)
bitcoindf = bitcoindf.fillna(-1)
bitcoin = np.zeros((np.size(bitcoindf.values,0)-1,np.size(bitcoindf.values,1)-1))

for i in range(np.size(bitcoin,0)):
    for j in range(np.size(bitcoin,1)):
        bitcoin[i,j] = float(bitcoindf.values[i+1,j+1])

#bitcoin difference data
bitcoin_diff = np.zeros((np.size(bitcoin,0)-1,np.size(bitcoin,1)))
i = len(bitcoin)-2
while i>=0:
    bitcoin_diff[i,:] = bitcoin[i+1,:] - bitcoin[i,:]
    i-=1

#bitcoin percentage change data
bitcoin_perc_chan=np.zeros((np.size(bitcoin,0)-1,np.size(bitcoin,1)))
for i in range(np.size(bitcoin_diff,0)):
    for j in range(np.size(bitcoin,1)):
        if bitcoin[i,j]!=0:
            bitcoin_perc_chan[i,j] = bitcoin_diff[i,j]/bitcoin[i,j]


#ndays: number of previous prices to consider
def dataPrep(ndays):
    #pricediff bitcoin price difference for previous 30 days
    priceperc = np.zeros((np.size(bitcoin_perc_chan,0)-ndays, ndays))
    for i in range(np.size(bitcoin_perc_chan,0)-ndays):
        for j in range(ndays):
            priceperc[i,j] = bitcoin_perc_chan[i+j,0]
            
    pricediff = np.zeros((np.size(bitcoin_diff,0)-ndays, ndays))
    for i in range(np.size(bitcoin_diff,0)-ndays):
        for j in range(ndays):
            pricediff[i,j] = bitcoin_diff[i+j,0]
    
    pricenextinc = np.zeros((np.size(pricediff,0),1))
    pricelabels = np.zeros((np.size(pricediff,0),1)) #price up or down
    for i in range(np.size(pricediff,0)):
        pricenextinc[i] = bitcoin_diff[i+ndays,0]
        if pricenextinc[i]>0:
            pricelabels[i] = 1
        elif pricenextinc[i]<0:
            pricelabels[i] = -1
        else:
            pricelabels[i] = 0
    return pricediff, priceperc, pricelabels, pricenextinc


#PCA
def buildPCA(X):
    μ=np.mean(X,axis=0)
    Z=X-μ
    C=np.cov(Z,rowvar=False)
    [λ,V]=np.linalg.eigh(C);
    λ=np.flipud(λ);V=np.flipud(V.T);
    return μ, V

def PCA(X, μ, V, numComp):
    #Takes a matrix and a number representing the number of principle components to be used in the approximation
    #Returns the approximation using numComp components and reconstructed matrix
    Z=X-μ
    P=np.dot(Z,V.T[:,0:numComp]); #Principal components
    R=np.dot(P,V[0:numComp,:]); #Reconstruction using numComp components
    Xrec=R+μ;
    return Z, P, R, Xrec



#Bayesian
def separate(P, labels):
    col = P.shape[1]
    Pos = np.zeros(shape=(len(labels[labels==1]), col))
    Neg = np.zeros(shape=(len(labels[labels==-1]), col))
    Pcount = 0; Ncount = 0;
    for i in range(len(labels)): 
        if labels[i] == 1:
            Pos[Pcount] = P[i,:]
            Pcount += 1
        elif labels[i] == -1:
            Neg[Ncount] = P[i,:]
            Ncount += 1
    return Pos, Neg

    
def Build2DBayesianClassifier(P, labels):
    [Pos, Neg] = separate(P, labels)
    mup = np.mean(Pos, axis=0)
    mun = np.mean(Neg, axis=0)
    covp = np.cov(Pos, rowvar=False)
    covn = np.cov(Neg, rowvar=False)
    Np = len(Pos)
    Nn = len(Neg)
    return mup, mun, covp, covn, Np, Nn 

def pdf(x, mu, sigma):
    d= np.alen(mu)
    dfact1 = (2*np.pi) ** d
    dfact2 = np.linalg.det(sigma)
    fact= 1/np.sqrt(dfact1*dfact2)
    xc=x-mu
    isigma=np.linalg.inv(sigma)
    return fact*np.exp(-0.5*np.einsum('ij,jk,ik->i', xc, isigma, xc))

def Apply2DBayesianClassifier(queries, mup, mun, covp, covn, Np, Nn):
    countP = Np * pdf(queries, mup, covp)
    countN = Nn * pdf(queries, mun, covn)
    resultlabel = np.full(np.alen(queries), "Indeterminate", dtype=object)
    indicesP = countP>countN
    indicesN = countN>countP
    resultlabel[indicesP] = 1
    resultlabel[indicesN] = -1
    resultprob = countP/(countP + countN)
    return resultlabel, resultprob

def ConfusionMatrix(T,R):
    labels = np.unique(T)
    C = len(labels)
    cm = np.zeros([C,C]).astype(int)
    labeldic = {lab: i for lab, i in zip(labels, np.arange(C))}
    for t, r in zip(T,R):
        if r in labeldic: cm[labeldic[t], labeldic[r]] += 1
    return cm

def Sensitivity(cm):
    rng = np.arange(len(cm))
    TP = np.array([cm[t,t] for t in rng])
    NP = np.sum(cm, axis=1)
    return TP/NP

def Specificity(cm):
    rng = np.arange(len(cm))
    TN = np.array([np.sum(cm[np.ix_([i for i in rng if i!=t], [j for j in rng if j!=t])]) for t in rng])
    NP = np.sum(cm, axis=1)
    total = np.sum(cm)
    NN = total - NP
    return TN/NN

ybayes=list();xbayes=list();yknn=list();yem=list();ysvm=list();xsvm=list();
yknnneignum = list()

ndays = 10
while ndays<91:
    
    [pricediff, priceperc, pricelabels, pricenextinc] = dataPrep(ndays)
    
    Xhistory = np.zeros((len(pricelabels),ndays*np.size(bitcoin_perc_chan,1)))
    for i in range(len(pricelabels)):
        for j in range(ndays):
            Xhistory[i,j:j+np.size(bitcoin_perc_chan,1)] = (bitcoin_perc_chan[i+j,:])
    
    
    [mux, Vx] = buildPCA(Xhistory)
    [Zx, Px, Rx, Xrec] = PCA(Xhistory, mux, Vx, ndays)
    [muy, Vy] = buildPCA(priceperc)
    [Zy, Py, Ry, Yrec] = PCA(priceperc, muy, Vy, round(ndays/2))
    
    [mup, mun, covp, covn, Np, Nn] = Build2DBayesianClassifier(Py, pricelabels)
    [bayesianPredictions, bayesianProbabilities] = Apply2DBayesianClassifier(Py, mup, mun, covp, covn, Np, Nn)
    bayesAccuracy = sum([1 for i in range(len(pricelabels)) if bayesianPredictions[i]==pricelabels[i]])/len(pricelabels)
    #print("y bayes")
    #print(bayesAccuracy)
    ybayes.append(bayesAccuracy)
    
    
    [mup, mun, covp, covn, Np, Nn] = Build2DBayesianClassifier(Px, pricelabels)
    [bayesianPred, bayesianProbabilities] = Apply2DBayesianClassifier(Px, mup, mun, covp, covn, Np, Nn)
    xbayesAccuracy = sum([1 for i in range(len(pricelabels)) if bayesianPred[i]==pricelabels[i]])/len(pricelabels)
    #print("x bayes")
    #print(xbayesAccuracy)
    xbayes.append(xbayesAccuracy)
    
    #k-NN
    n_neighbors = 1
    maxaccuracy = 0
    maxaccnum = 0
    knnpred = []
    while (n_neighbors<32):
        # we create an instance of Neighbours Classifier and fit the data.
        knnclf = neighbors.KNeighborsClassifier(n_neighbors)
        knnclf.fit(priceperc[500:2000,:], pricelabels[500:2000,:])
        YKNNpred = knnclf.predict(priceperc[2000:,:])
        knnAccuracy = sum([1 for i in range(len(pricelabels)-2000) if YKNNpred[i]==pricelabels[i+2000]])/(len(pricelabels)-2000)
        #print("y knn")
        #print(knnAccuracy)
        if knnAccuracy > maxaccuracy:
            maxaccuracy = knnAccuracy
            maxaccnum = n_neighbors
            knnpred = YKNNpred
        n_neighbors += 2
    yknn.append(maxaccuracy)
    yknnneignum.append(maxaccnum)
    
    #Expectation Maximization
    estimator = GaussianMixture(n_components=2, covariance_type='full', max_iter=100, n_init=100, init_params='kmeans', random_state=0)
    estimator.fit(pricediff)
    YEMpred = estimator.predict(pricediff)
    EMAccuracy = sum([1 for i in range(len(pricelabels)) if YEMpred[i]==pricelabels[i]])/len(pricelabels)
    #print("y EM")
    #print(EMAccuracy)
    yem.append(EMAccuracy)
    
    #Support Vector Machine
    svmclf = svm.SVC()
    svmclf.fit(pricediff[500:2000,:], pricelabels[500:2000,:])
    Ysvmpred = svmclf.predict(pricediff[2000:,:])
    svmAccuracy = sum([1 for i in range(len(pricelabels)-2000) if Ysvmpred[i]==pricelabels[i+2000]])/(len(pricelabels)-2000)
    #print("y svm")
    #print(svmAccuracy)
    ysvm.append(svmAccuracy)
    
    svmclf.fit(Px[500:2000,:], pricelabels[500:2000,:])
    Xsvmpred = svmclf.predict(Px[2000:,:])
    XsvmAccuracy = sum([1 for i in range(len(pricelabels)-2000) if Xsvmpred[i]==pricelabels[i+2000]])/(len(pricelabels)-2000)
    #print("x svm")
    #print(XsvmAccuracy)
    xsvm.append(XsvmAccuracy)
    
    ndays += 10
    print(ndays)
    
print("DONE!")
t = np.arange(10,100,10)
plt.plot(t,ybayes, 'r--', label = 'Bayes')
plt.plot(t, yknn, 'bs', label='knn')
plt.plot(t,ysvm, 'g^', label='svm')
plt.plot( t, yem, 'ko', label='EM')
plt.legend(loc='upper rigt')
plt.xlabel('Number of Days')
plt.ylabel('Accuracy')
plt.grid(True)
plt.figure
plt.plot(t,xbayes, 'r--', label = 'Bayes')
plt.plot(t,xsvm, 'g^', label='svm')
plt.legend(loc='upper rigt')
plt.xlabel('Number of Days')
plt.ylabel('Accuracy')
plt.grid(True)

#ybayes=list();xbayes=list();yknn=list();yem=list();ysvm=list();xsvm=list();
