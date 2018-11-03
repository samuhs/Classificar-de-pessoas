import numpy as np
import pandas as pd
from pandas.compat import StringIO

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import time
# meus pacotes
from Perceptron import *


#separando as classe e importando o dataset
dataset = pd.read_csv("out.csv")

# categorizando os dados
def nome(x):
    if x == 'lucas':
        return -1
    else:
        return 1   
dataset['Classes'] = dataset['Classes'].apply(nome)

#separando as classes do dataset
classe =dataset['Classes']
dataset=dataset.drop(['Classes'],axis=1)

#criando grupo de treino e teste
group_train,group_test,classe_train,classe_test=train_test_split(dataset,classe,test_size=0.33,random_state=100)

# transformando em array
grupo = np.array(group_train)
clasif=np.array(classe_train)

# Normalização do dataset
grupo.shape =(-1,len(dataset.columns))
scaler=preprocessing.MinMaxScaler()
grupoN = scaler.fit_transform(grupo)


#criando dataset normalizado
datanorm = 0
atributos=[]
for i in range(0,60):
    atributos.append('atri'+ str(i))
datanorm = pd.DataFrame(grupoN,columns=(atributos))

# taxa de aprendizado
m=0.01

neuronio = Perceptron()

#treinando neuronio
print("\nTreinando Neuronio...")
start = time.time()
neuronio.fit(datanorm,clasif,m)

total = float((time.time() - start))
print("\nTempo de treinamento")
print("--- %s seconds ---" %(total) )

# transformando em array
grupo_test = np.array(group_test)
clasif_test=np.array(classe_test)

# Normalização do dataset
grupo_test.shape =(-1,len(dataset.columns))
scaler=preprocessing.MinMaxScaler()
grupoN_test = scaler.fit_transform(grupo_test)


#criando dataset normalizado
datanorm_test = 0
datanorm_test = pd.DataFrame(grupoN_test,columns=(atributos))

#aplicando grupo de teste
star = time.time()
result_test=neuronio.apply(datanorm_test)
print("\nTempo de teste:")
print("--- %s seconds ---" %(time.time() - start) )

total = total+float((time.time() - start))
print("\nResultado de acuracia com o grupo de teste")
print(result_test == clasif_test)

epoca = neuronio.returnEpoch()
print("\nNumero de Epocas: ", epoca)

print("\nResultado das classes predizidas no grupo de teste")
for index,item in enumerate(clasif_test):
    if item == -1:
        print("O %d é Lucas"%index)
    else:
        print("O %d é James"%index)

print(" Tempo total de execução foi: ",total)