from sklearn.preprocessing import MinMaxScaler
import time
import os
import copy
import cv2
import torch.distributed as dist
import time
import os
import copy
from torch.utils.data import TensorDataset, DataLoader,Dataset, Sampler
import torch.utils.data as data_utils
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import multiprocessing
import pickle
import dill
import encoding
import qsvm
import numpy as np
import os
import psutil
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score

def metricas_modelos(y_true, y_pred):
    from sklearn.metrics import accuracy_score
    accuracy = accuracy_score(y_true, y_pred)
    return(accuracy)

def Dataset(X, y, test_size_split=0.2): #For PCA it is important to Standarized data. It is first done on training set and applied to 
    class_labels = [r'0', r'1']

    n_samples = np.shape(X)[0]
    training_size = int(n_samples-(n_samples*test_size_split))
    test_size =int(n_samples-training_size)
    train_sample, test_sample, train_label, test_label = \
        train_test_split(X, y, stratify=y, test_size=test_size_split, random_state=12)

    std_scale = StandardScaler().fit(train_sample)
    train_sample = std_scale.transform(train_sample)
    test_sample = std_scale.transform(test_sample)

    samples = np.append(train_sample, test_sample, axis=0)
    minmax_scale = MinMaxScaler((-1, 1)).fit(samples)
    train_sample = minmax_scale.transform(train_sample)
    test_sample = minmax_scale.transform(test_sample)

    return train_sample, train_label, test_sample, test_label

class Fitness:

    def __init__(self, nqubits, X, y):
        self.nqubits = nqubits
        self.X = X
        self.y = y
        
    def __call__(self, POP):
        return self.fitness(POP)

    def fitness(self, POP):
        
        Components = POP[:6] #number of components - First 6 bits
        POP_ind = POP[7:] # Quantum circuit
        param = int("".join(str(i) for i in Components),2)
        
        if param < self.X.shape[0] and param >=2:  #death penalty
                        
            training_features, training_labels, test_features, test_labels = \
            Dataset(self.X, self.y) ###normalizamos el dataset

            faces_pca = PCA(n_components=param,svd_solver='randomized',whiten=True)
            faces_pca.fit(training_features) #We fit the PCA method in training 
            training_features = faces_pca.transform(training_features)
            test_features = faces_pca.transform(test_features) #transform the test dataset by using the fitted method.
            
            cc = encoding.CircuitConversor(self.nqubits,param)
            model = qsvm.QSVM(lambda parameters: cc(POP_ind, parameters)[0],
                              training_features, training_labels)
            y_pred = model.predict(test_features) 
            acc = metricas_modelos(test_labels, y_pred) #We predict the label
            POP_ind=''.join(str(i) for i in POP_ind)
            _, gates = cc(POP_ind, training_features[:,[0,1]])
            gate = gates/self.nqubits
            wc = gate + (gate*(acc**2))
            models = (param,POP_ind)
            if acc > 0.7: #we save all the indivicuals with acc higher than 0.7
                with open(r'C:\Users\sergi\Desktop\AUTOQML_CODE\COVID\COV_PCA_05\ind\tuples'+'_'+str(acc)+'_'+str(wc)+'.pkl', "wb") as dill_file:
                        dill.dump(models, dill_file)
        else:
            wc = 100000
            acc = 0
        return wc, acc #
