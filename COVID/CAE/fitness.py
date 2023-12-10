from sklearn.preprocessing import MinMaxScaler
import time
import os
import copy
import pandas as pd
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

def Dataset(X, y, test_size_split=0.2):
    class_labels = [r'0', r'1']
    
    train_x_0 =pd.DataFrame([])
    for i in range(len(X)):
            face = pd.Series(X[i].flatten())
            train_x_0 = train_x_0.append(face,ignore_index=True)  

    n_samples = np.shape(train_x_0)[0]
    training_size = int(n_samples-(n_samples*test_size_split))
    test_size =int(n_samples-training_size)
    train_sample, test_sample, train_label, test_label = \
        train_test_split(train_x_0, y, stratify=y, test_size=test_size_split, random_state=12)

    std_scale = StandardScaler().fit(train_sample)
    train_sample = std_scale.transform(train_sample)
    test_sample = std_scale.transform(test_sample)

    samples = np.append(train_sample, test_sample, axis=0)
    minmax_scale = MinMaxScaler((-1, 1)).fit(samples)
    train_sample = minmax_scale.transform(train_sample)
    test_sample = minmax_scale.transform(test_sample)
    return train_sample, train_label, test_sample, test_label

class Fitness:

    def __init__(self, nqubits, X, y, debug=False):
        self.nqubits = nqubits
        self.X = X
        self.y = y
        self.debug = debug

    def __call__(self, POP):
        return self.fitness(POP)

    def fitness(self, POP):
            POP_ind = POP    #The individual is all the binary string         
            training_features, training_labels, test_features, test_labels = \
            Dataset(self.X, self.y)
            cc = encoding.CircuitConversor(self.nqubits,training_features.shape[1])
            model = qsvm.QSVM(lambda parameters: cc(POP_ind, parameters)[0],
                              training_features, training_labels)
            y_pred = model.predict(test_features) 
            acc = metricas_modelos(test_labels, y_pred) 
            POP_ind=''.join(str(i) for i in POP_ind)
            _, gates = cc(POP_ind, training_features[:,[0,1]])
           # if self.debug:
           #     print(f'String: {POP}\n -> accuracy = {acc}, gates = {gates}')
            gate = gates/self.nqubits #calculamos las puertas por qubits
            wc = gate + (gate*(acc**2)) # variacional en la fitness -- para el tema de los pesos
            models = (training_features.shape[1],POP_ind)
            if acc > 0.8:
                with open(r'C:\Users\sergi\Desktop\AUTOQML_CODE\COVID\ConvAE_Aproximation2\ind\tuples'+'_'+str(acc)+'_'+str(wc)+'.pkl', "wb") as dill_file:
                        dill.dump(models, dill_file)
                        
            return wc, acc #
