import numpy as np
import pandas as pd
from skimage import io, color, img_as_ubyte
from skimage.feature import greycomatrix, greycoprops
from sklearn.metrics.cluster import entropy
import skimage
import os

class Features ():
    def __init__(self):
        self.nada =0
    
    def vetor_classif(self,img_name):
        rgbImg = io.imread('process/'+ img_name)
        grayImg = img_as_ubyte(color.rgb2gray(rgbImg))

        distances = [1, 2, 3]
        angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
        #properties = ['energy', 'homogeneity']
        properties = ['contrast', 'energy', 'homogeneity', 'correlation', 'dissimilarity']

        glcm = greycomatrix(grayImg, 
                    distances=distances, 
                    angles=angles,
                    symmetric=True,
                    normed=True)

        feats = np.hstack([greycoprops(glcm, prop).ravel() for prop in properties])
        np.set_printoptions(precision=4)

        # retornando haralick descritor , 60 caracteristicas
        return feats

    def gerar_csv(self):
        # pegando imagens da pasta
        train_path = "process"
        train_names = os.listdir(train_path)

        # carregando os dados para uma lista antes de virar csv
        lista_caracteristicas = np.empty([len(train_names),60])
        classes=np.empty(len(train_names),dtype=object)
        for index,item in enumerate (train_names):
            vet = self.vetor_classif(item)
            name = item.split("-",1)
            classes[index]=name[0]
            lista_caracteristicas[index,:] = vet

        #criando o csv
        atributos=[]
        for i in range(0,60):
            atributos.append('atri'+ str(i))

        dataset = pd.DataFrame(lista_caracteristicas,columns=atributos)
        dataset['Classes']=classes
        dataset.to_csv("out.csv",index=False)   

        print("Csv gerado com sucesso")

#aux=skimage.measure.shannon_entropy(glcm)
