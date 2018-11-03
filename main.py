import numpy as np
import pandas as pd
from skimage import io, color, img_as_ubyte
from skimage.feature import greycomatrix, greycoprops
from sklearn.metrics.cluster import entropy
import skimage
import os
from Features import *
from camera import *

print(" Gerarando nova classe!")

controler = input (" Deseja adicionar uma nova classe?: y/n    ")

if controler == 'y':
    nome= input("Insira o nome da nova classe: ")

    nova_classe= camera(nome)
    nova_classe.save_img()


    features = Features()
    features.gerar_csv()


    
