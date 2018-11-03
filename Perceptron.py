import numpy as np
import pandas as pd
from pandas.compat import StringIO
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

class Perceptron():
    def __init__(self):
        self.datanorm=0
        self.classe =0
        #taxa de aprendizado
        self.m = 0
        # vetor de pessos
        self.pesos= 0

        self.epoch=0
        

    def ativacao (self,w,x):
        result= w @ x
        if result <= 0:
            return -1
        else:
            return 1
   
    def ajust_pesos(self,w,x,y,i):
        w.shape=(-1,1)
        result = w + (self.m*(self.classe[i]-y)*x)
        return result
    
    def fit(self,X,Y,m):
        self.datanorm= X
        self.classe =Y
        self.m = m

        w = np.zeros(len(self.datanorm.columns)+1,dtype = int)
        w.shape = (1,len(self.datanorm.columns)+1)
        resultados = np.zeros(len(self.datanorm),dtype = int)
        i=0
        time = 0

        #inicializa loop
        while not(np.array_equal(resultados,self.classe)): 
           
            #pegando individualmente cada item do dataset
            xf=0    
            xf = np.array(self.datanorm.iloc[i,:])
            xf = np.concatenate(([1],xf))
            xf.shape= (len(xf),1)
           
            #aplicando função de ativação e corrigindo pesos
            resultados[i]=self.ativacao(w,xf) # y(n)
            w=self.ajust_pesos(w,xf,resultados[i],i)
            
            #voltando para o formato inicial
            w.shape=(1,len(self.datanorm.columns)+1)
            
            # contando os item
            i = i+1

            if i == len(self.datanorm):
                i = 0
                self.epoch = self.epoch + 1

            if self.epoch == 1000:
                break

        self.pesos = w
    
    def apply (self,X):
        classificacao = np.zeros(len(X))
        for i in range(0,len(X)):
            xf=0    
            xf = np.array(X.iloc[i,:])
            xf = np.concatenate(([1],xf))
            result= self.pesos @ xf
            if result <= 0:
                classificacao[i]=-1
            else:
                classificacao[i]=1
        return classificacao        

    def returnEpoch(self):
        return self.epoch
