
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import tensorflow as tf
import pandas as pnd

class datos_preparados():

    def __init__(self,X, y):
        self.X=X
        self.y=y

    def train_test(self, indice):
        observaciones = pnd.read_csv("datas/sonar.all-data.csv")

        print("N.º columnas: ",len(observaciones.columns))
        #Para el aprendizaje solo tomamos loa datos procedentes del sonar
        self.X = observaciones[observaciones.columns[0:60]].values

        #Solo se toman los etiquetados
        self.y = observaciones[observaciones.columns[60]]

        #Se codifica: Las minas son iguales a 0 y las rocas son iguales 1
        encoder = LabelEncoder()
        encoder.fit(self.y)
        self.y = encoder.transform(self.y)

        #Se añade un cifrado para crear clases:
        # Si es una mina [1,0]
        # Si es una roca [0,1]
        n_labels = len(y)
        n_unique_labels = len(np.unique(self.y))
        one_hot_encode = np.zeros((n_labels,n_unique_labels))
        one_hot_encode[np.arange(n_labels),self.y] = 1
        Y=one_hot_encode

        #Verificación tomando los registros 0 y 97
        print("Clase Roca:",int(Y[0][1]))
        print("Clase Mina:",int(Y[97][1]))

        #---------------------------------------------
        # CREACIÓN DE LOS CONJUNTOS DE APRENDIZAJE Y DE PRUEBAS
        #---------------------------------------------

        #Mezclamos
        X, Y = shuffle(X, Y, random_state=1)

        #Creación de los conjuntos de aprendizaje
        train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=0.07, random_state=42)
        soluciones=[train_x, test_x, train_y, test_y]

        if indice==0:
            return soluciones[0]
        if indice==1:
            return soluciones[1]
        if indice==2:
            return soluciones[2]
        if indice==3:
            return soluciones[3]
        if indice==4:
            return Y
        if intice==5:
            return X



    #---------------------------------------------
    # PARAMETRIZACIÓN DE LA RED NEURONAL
    #---------------------------------------------
    def neuEntrada(self):

        #Variable TensorFLow correspondiente a los 60 valores de las neuronas de entrada
        tf_neuronas_entradas_X = tf.placeholder(tf.float32,[None, 60])
        return tf_neuronas_entradas_X

    def varReales(Self):
        #Variable TensorFlow correspondiente a las 2 neuronas de salida
        tf_valores_reales_Y = tf.placeholder(tf.float32,[None, 2])
        return tf_valores_reales_Y


    def pesos(self):
        pesos = {
            #60 neuronas de las entradas hacia 24 Neuronas de la capa oculta
            'capa_entrada_hacia_oculta': tf.Variable(tf.random_uniform([60, 12], minval=-0.3, maxval=0.3), tf.float32),

            # 12 neuronas de la capa oculta hacia 2 de la capa de salida
            'capa_oculta_hacia_salida': tf.Variable(tf.random_uniform([12, 2], minval=-0.3, maxval=0.3), tf.float32),
        }
        return pesos

    def pesos_sesgo(self):

        peso_sesgo = {
            #1 sesgo de la capa de entrada hacia las 24 neuronas de la capa oculta
            'peso_sesgo_capa_entrada_hacia_oculta': tf.Variable(tf.zeros([12]), tf.float32),

            #1 sesgo de la capa oculta hacia las 2 neuronas de la capa de salida
            'peso_sesgo_capa_oculta_hacia_salida': tf.Variable(tf.zeros([2]), tf.float32),
        }
        return peso_sesgo

