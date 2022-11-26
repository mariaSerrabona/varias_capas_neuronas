
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf


class preparacion_datos():

    def __init__(self, observaciones, X, y):
        self.observaciones=observaciones
        self.X=X
        self.y=y

    def preparacion_de_datos(self):

        #---------------------------------------------
        # PREPARACIÓN DE LOS DATOS
        #---------------------------------------------

        print("Nº columnas: ",len(self.observaciones.columns))
        #Para el aprendizaje olo se toman los datos procedentes del sonar
        self.X = self.observaciones[self.observaciones.columns[0:60]].values

        #Solo se toman los etiquetados
        self.y = self.observaciones[self.observaciones.columns[60]]

        #Se codifica: Las minas son iguales a 0 y las rocas son iguales a 1

        encoder = LabelEncoder()
        encoder.fit(self.y)
        self.y = encoder.transform(self.y)

        #Se añade un cifrado para crear clases:
        # Si es una mina [1,0]
        # Si es una roca [0,1]

        n_labels = len(self.y)
        n_unique_labels = len(np.unique(self.y))
        one_hot_encode = np.zeros((n_labels,n_unique_labels))
        one_hot_encode[np.arange(n_labels),self.y] = 1
        Y=one_hot_encode

        #Verificación tomando los registros 0 y 97
        print("Clase Roca:",int(Y[0][1]))
        print("Clase Mina:",int(Y[97][1]))

        #Mezclamos

        self.X, Y = shuffle(self.X, Y, random_state=1)


        #---------------------------------------------
        # PARAMETRIZACIÓN DE LA RED NEURONAL
        #---------------------------------------------
        train_x, test_x, train_y, test_y = train_test_split(self.X, Y, test_size=0.20, random_state=42)
        soluciones=[train_x, test_x, train_y, test_y]
        return soluciones

    def neuEntrada(self):

        #Variable TensorFLow correspondiente a los 60 valores de las neuronas de entrada
        tf_neuronas_entradas_X = tf.compat.v1.placeholder(tf.float32,[None, 60])
        return tf_neuronas_entradas_X

    def varReales(self):
        #Variable TensorFlow correspondiente a las 2 neuronas de salida
        tf_valores_reales_Y = tf.compat.v1.placeholder(tf.float32,[None, 2])
        return tf_valores_reales_Y

    def pesos(self):
        pesos = {
            # 60 neuronas de entradas hacia 24 Neuronas de la capa oculta
            'capa_entrada_hacia_oculta': tf.Variable(tf.random_normal([60, 31]), tf.float32),

            # 24 neuronas de la capa oculta hacia 2 de la capa de salida
            'capa_oculta_hacia_salida': tf.Variable(tf.random_normal([31, 2]), tf.float32),
        }
        return pesos

    def peso_sesgo(self):
        peso_sesgo = {
            #1 sesgo de capa de entrada hacia las 24 neuronas de la capa oculta
            'peso_sesgo_capa_entrada_hacia_oculta': tf.Variable(tf.zeros([31]), tf.float32),

            #1 sesgo de la capa oculta hacia las 2 neuronas de la capa de salida
            'peso_sesgo_capa_oculta_hacia_salida': tf.Variable(tf.zeros([2]), tf.float32),
        }
        return peso_sesgo