import csv
import numpy as np
import matplotlib.pyplot as plt
import scipy.spatial
from collections import defaultdict


def whiten(dataCov,data):
    #newData = data.copy()
    d,u = np.linalg.eig(dataCov)
    c1 = np.sqrt(d)
    c1 = 1/c1
    c1 = np.diag(c1)
    meanRow = [np.mean(data[:,i])for i in range(7)]
    means = []
    for i in range(len(data)):
        means.append(meanRow)
    means = np.array(means)
    newData = []
    for i in range(len(data)):
        newData.append(np.matmul(c1,np.matmul(u.transpose(),(data[i]-means[i]))))    
    newData = np.array(newData)
    
    return newData

def kmeans(k):
    c = np.random.randint(0,len(data),size=k)
    centres = []
    centres+=(data[i]for i in c)
    centres = np.array(centres)
    eta = 0.001
    sse = 0
    vals_at_cent = defaultdict(list)
    new_vals_at_cent = defaultdict(list)
    #count = 0
    match = False
    while not match:
        #print count
        cents = []
        new_vals_at_cent.clear()
        for i in range(len(data)):
            nearest = 0
            nearest_dist = np.inf
            for j in range(len(centres)):
                dist = scipy.spatial.distance.euclidean(data[i],centres[j])
                if dist < nearest_dist:
                    nearest = j
                    nearest_dist = dist
            deltC = eta*(data[i]-centres[nearest])
            centres[nearest] += deltC
            sse+= nearest_dist**2
            cents.append(nearest)
            new_vals_at_cent[nearest].append(data[i].tolist())
        for r in range(k):
            if vals_at_cent[r] == new_vals_at_cent[r]:
                match = True
            else:
                #print 'cent ',r,' does not match'
                match = False
                break
        vals_at_cent = new_vals_at_cent.copy()
        #count += 1
    #plt.scatter(data[:,0], data[:,1])
    #plt.scatter(centres[:,0],centres[:,1], c='r')
    return centres,sse, vals_at_cent,cents

def calc_err(outs,targs):
    error = 0
    for i in range(len(targs)):
        error += (outs[i] - targs[i])**2
    error = error/len(targs)
    return error


def kfold(k,data,labels,sigma,regularise,lam):
    order = np.random.permutation(len(data)).tolist()
    error = []
    for j in range(k):
        test = []
        test_lab = []
        test_indices = []
        sigs = []
        train = []
        train_lab = []
        for i in range(len(data)/k):
            test.append(data[order[0]])
            test_lab.append(labels[order[0]])
            test_indices.append(order.pop(0))
        for i in range(len(order)):
            train.append(data[order[i]])
            train_lab.append(labels[order[i]])
       
        test_mat = np.array(test)
        test_lab_mat = np.array(test_lab)
        
        train_mat = np.array(train)
        train_lab_mat = np.array(train_lab)
        
        train_cov = np.cov(train_mat,rowvar=False)
        train_mat = whiten(train_cov, train_mat)
        
        for i in range(len(train_mat[0])):
            sig = np.std(train_mat[:,i])
            sigs.append(sig)
            train_mat[:,i]= train_mat[:,i]/sig
        
        centres,sse, vals_at_cent,cents = kmeans(50)
        
        
        dist_mat =[]
        for i in range(len(centres)):
            cent_inv = np.linalg.inv(np.cov(vals_at_cent[i], rowvar= False))
            cdist = scipy.spatial.distance.cdist(train_mat,centres,metric='mahalanobis',VI=cent_inv)
            dist_mat.append(cdist[:,i])
        dist_mat = np.array(dist_mat)
        dist_mat = dist_mat.transpose()       
        
        phi = np.exp(-((dist_mat**2)/2*(sigma**2)))
        phinv = np.linalg.pinv(phi)
        if not regularise:
            w = np.matmul(phinv,train_lab)
        else:
            t1 = np.matmul(phi.transpose(),phi)
            t2 = np.linalg.inv(t1+lam*np.identity(len(centres)))
            t3 = np.matmul(phi.transpose(),train_lab_mat)
            w = np.matmul(t2,t3)
        
        test_mat = whiten(train_cov,test_mat)
        
        #----- preprocess based on training data -----#
        for i in range(len(test_mat[0])):
            test_mat[:,i]= test_mat[:,i]/sigs[i]
        
        dist_mat =[]
        for i in range(len(centres)):
            cent_inv = np.linalg.inv(np.cov(vals_at_cent[i], rowvar= False))
            cdist = scipy.spatial.distance.cdist(test_mat,centres,metric='mahalanobis',VI=cent_inv)
            dist_mat.append(cdist[:,i])
        dist_mat = np.array(dist_mat)
        dist_mat = dist_mat.transpose()       
        
        phi = np.exp(-((dist_mat**2)/2*(sigma**2)))
        y = np.dot(phi,w)
        error.append(calc_err(y,test_lab))
        
        for i in range(len(test_indices)):
            order.append(test_indices.pop(0))
    return np.mean(error)

#----- script -----#
data = []
with open('data105882.csv','rb') as dataset:
    dataReader = csv.reader(dataset)
    for row in dataReader:
        data.append(row)
for i in range(len(data)):
    for j in range(len(data[i])):
        data[i][j] = float(data[i][j])

data = np.array(data)
labels = data[:,7]
data = data[:,:7]

data = whiten(np.cov(data,rowvar=False),data)
cov = np.cov(data, rowvar=False)

for i in range(len(data[0])):
    sig = np.std(data[:,i])
    data[:,i]= data[:,i]/sig

lam_vals = np.arange(0,1,step = 0.2)
errors = []
for i in range(len(lam_vals)):
    errors.append(kfold(5,data.copy(),labels.copy(),0.1,True,lam_vals[i]))

















































'''def kmeans(k):
    c = np.random.randint(0,len(data),size=k)
    centres = []
    centres+=(data[i]for i in c)
    centres = np.array(centres)
    eta = 0.000001
    sse = 0
    vals_at_cent = defaultdict(list)
    new_vals_at_cent = defaultdict(list)
    count = 0
    match = False
    while not match:
        print count
        cents = []
        new_vals_at_cent.clear()
        for i in range(len(data)):
            nearest = 0
            nearest_dist = np.inf
            for j in range(len(centres)):
                dist = scipy.spatial.distance.euclidean(data[i],centres[j])
                if dist < nearest_dist:
                    nearest = j
                    nearest_dist = dist
            deltC = eta*(data[i]-centres[nearest])
            centres[nearest] += deltC
            sse+= nearest_dist**2
            cents.append(nearest)
            new_vals_at_cent[nearest].append(data[i].tolist())
        for r in range(k):
            if vals_at_cent[r] == new_vals_at_cent[r]:
                match = True
            else:
                print 'cent ',r,' does not match'
                match = False
                break
        vals_at_cent = new_vals_at_cent.copy()
        count += 1
    plt.scatter(data[:,0], data[:,1])
    plt.scatter(centres[:,0],centres[:,1], c='r')
    return centres,sse, vals_at_cent,cents


def whiten(dataCov,data):
    d,u = np.linalg.eig(dataCov)
    c1 = np.sqrt(d)
    c1 = 1/c1
    c1 = np.diag(c1)
    meanRow = [np.mean(data[:,i])for i in range(7)]
    means = []
    for i in range(len(data)):
        means.append(meanRow)
    means = np.array(means)
    newData = []
    for i in range(len(data)):
        newData.append(np.matmul(c1,np.matmul(u.transpose(),(data[i]-means[i]))))    
    newData = np.array(newData)
    
    return d,u,means,newData

data = []
with open('data105882.csv','rb') as dataset:
    dataReader = csv.reader(dataset)
    for row in dataReader:
        data.append(row)
for i in range(len(data)):
    for j in range(len(data[i])):
        data[i][j] = float(data[i][j])

data = np.array(data)
targs = data[:,7]
data = data[:,:7]

d,u,means,data = whiten(np.cov(data,rowvar = False),data)
for i in range(len(data[0])):
    sig = np.std(data[:,i])
    data[:,i]= data[:,i]/sig

centres,sse, vals_at_cent,cents = kmeans(50)

#----- Calculate Phi Matrix -----#

#----- get distance matrix -----#

dist_mat =[]
for i in range(len(centres)):
    cent_inv = np.linalg.inv(np.cov(vals_at_cent[i], rowvar= False))
    cdist = scipy.spatial.distance.cdist(data,centres,metric='mahalanobis',VI=cent_inv)
    dist_mat.append(cdist[:,i])
dist_mat = np.array(dist_mat)
dist_mat = dist_mat.transpose()

sigma = 0.6
    
phi = np.exp(-((dist_mat**2)/2*(sigma**2)))'''

    


    
'''def kmeans(k):
    c = np.random.randint(0,len(data),size=k)
    centres = []
    centres+=(data[i]for i in c)
    centres = np.array(centres)
    eta = 0.000001
    sse = 0
    vals_at_cent = defaultdict(list)
    new_vals_at_cent = defaultdict(list)
    count = 0
    match = False
    while not match:
        print count
        new_vals_at_cent.clear()
        for i in range(len(data)):
            nearest = 0
            nearest_dist = np.inf
            for j in range(len(centres)):
                dist = scipy.spatial.distance.euclidean(data[i],centres[j])
                if dist < nearest_dist:
                    nearest = j
                    nearest_dist = dist
            deltC = eta*(data[i]-centres[nearest])
            centres[nearest] += deltC
            sse+= nearest_dist**2
            new_vals_at_cent[nearest].append(data[i].tolist())
        for r in range(k):
            if vals_at_cent[r] == new_vals_at_cent[r]:
                match = True
            else:
                print 'cent ',r,' does not match'
                match = False
                break
        vals_at_cent = new_vals_at_cent.copy()
        count += 1
    plt.scatter(data[:,0], data[:,1])
    plt.scatter(centres[:,0],centres[:,1], c='r')
    return centres,sse, vals_at_cent,new_vals_at_cent'''
    


'''def kmeans(k):
    c = np.random.randint(0,len(data),size=k)
    centres = []
    centres+=(data[i]for i in c)
    centres = np.array(centres)
    newCentres = centres.copy()
    eta = 0.01
    sse = 0
    vals_at_cent = defaultdict(list)
    count = 0
    match = False
    while not match:
        print count
        for i in range(len(data)):
            nearest = 0
            nearest_dist = np.inf
            for j in range(len(centres)):
                dist = scipy.spatial.distance.euclidean(data[i],centres[j])
                if dist < nearest_dist:
                    nearest = j
                    nearest_dist = dist
            deltC = eta*(data[i]-centres[nearest])
            newCentres[nearest] += deltC
            #sse+= nearest_dist**2
        for r in range(k):
            centDist = scipy.spatial.distance.euclidean(centres[r],newCentres[r])
            if centDist < 2:
                print 'converged'
                match = True
            else:
                print r,'dist = ', centDist
                match = False
                break
        centres = newCentres.copy()
        count += 1
    plt.scatter(data[:,0], data[:,1])
    plt.scatter(centres[:,0],centres[:,1], c='r')
    return centres,sse, vals_at_cent'''