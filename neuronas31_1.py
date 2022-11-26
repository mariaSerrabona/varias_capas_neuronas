
import pandas as pnd
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
import matplotlib.pyplot as plt
from preparacion_datos_31 import preparacion_datos



class neurona31():

    def __init__(self, tf_valores_reales_Y ,observaciones_en_entradas):
        self.tf_valores_reales_Y=tf_valores_reales_Y
        self.observaciones_en_entradas=observaciones_en_entradas

    #---------------------------------------------
    # FUNCIÓN DE CREACIÓN DE LA RED NEURONAL
    #---------------------------------------------

    def aprendizaje(self, tasa_aprendizaje, epochs, red):

        observaciones = pnd.read_csv("datas/sonar.all-data.csv")
        X = observaciones[observaciones.columns[0:60]].values
        y = observaciones[observaciones.columns[60]]

        datos_preparados=preparacion_datos(observaciones, X, y)

        def red_neuronas_multicapa():
            #Cálculo de la activación de la primera capa
            primera_activacion = tf.sigmoid(tf.matmul(datos_preparados.neuEntrada(), datos_preparados.pesos()['capa_entrada_hacia_oculta']) + datos_preparados.peso_sesgo()['peso_sesgo_capa_entrada_hacia_oculta'])

            #Cálculo de la activación de la segunda capa
            activacion_capa_oculta = tf.sigmoid(tf.matmul(primera_activacion, datos_preparados.pesos()['capa_oculta_hacia_salida']) + datos_preparados.peso_sesgo()['peso_sesgo_capa_oculta_hacia_salida'])

            return activacion_capa_oculta

        red = red_neuronas_multicapa(datos_preparados.neuEntrada(), datos_preparados.pesos(), datos_preparados.peso_sesgo())


        lista_train_test=datos_preparados.preparacion_de_datos()

        train_x=lista_train_test[0]
        train_y = lista_train_test[2]
        #Función de error de media cuadrática MSE
        funcion_error = tf.reduce_sum(tf.pow(self.tf_valores_reales_Y-red,2))

        optimizador = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=tasa_aprendizaje).minimize(funcion_error)
        #---------------------------------------------
        # APRENDIZAJE
        #---------------------------------------------

        #Inicialización de la variable
        init = tf.compat.v1.global_variables_initializer()

        #Inicio de una sesión de aprendizaje
        sesion = tf.compat.v1.Session()
        sesion.run(init)

        #Para la realización del gráfico para la MSE
        Grafica_MSE=[]


        #Para cada epoch
        for i in range(epochs):

            #Realización del aprendizaje con actualización de los pesos
            sesion.run(optimizador, feed_dict = {datos_preparados.neuEntrada: datos_preparados.preparacion_de_datos[0], datos_preparados.varReales:datos_preparados.preparacion_de_datos[2]})

            #Calcular el error
            MSE = sesion.run(funcion_error, feed_dict = {datos_preparados.neuEntrada: datos_preparados.preparacion_de_datos[0], datos_preparados.varReales:datos_preparados.preparacion_de_datos[2]})

            #Visualización de la información
            Grafica_MSE.append(MSE)
            print("EPOCH (" + str(i) + "/" + str(epochs) + ") -  MSE: "+ str(MSE))


        #Visualización gráfica
        plt.plot(Grafica_MSE)
        plt.ylabel('MSE')
        plt.show()


    def verificar_aprendizaje(self, red):
        observaciones = pnd.read_csv("datas/sonar.all-data.csv")
        X = observaciones[observaciones.columns[0:60]].values
        y = observaciones[observaciones.columns[60]]

        datos_preparados=preparacion_datos(observaciones, X, y)

        clasificaciones = tf.argmax(red, 1)

        #En la tabla de valores reales:
        #Las minas se codifican como [1,0] y el índice de mayor valor es 0
        #Las rocas toman el valor [0,1] y el índice de mayor valor es 1

        #Si la clasificación es [0.90, 0.34 ] el índice de mayor valor es 0
        #Si es una mina [1,0] y el índice de mayor valor es 0
        #Si los dos índices son idénticos, entonces se puede afirmar que es una buena clasificación
        formula_calculo_clasificaciones_correctas = tf.equal(clasificaciones, tf.argmax(datos_preparados.varReales,1))


        #La precisión se calcula haciendo la media (tf.mean)
        # de las clasificaciones buenas (después de haberlas convertido en decimal tf.cast, tf.float32)
        formula_precision = tf.reduce_mean(tf.cast(formula_calculo_clasificaciones_correctas, tf.float32))



        #-------------------------------------------------------------------------
        # PRECISIÓN EN LOS DATOS DE PRUEBAS
        #-------------------------------------------------------------------------

        n_clasificaciones = 0
        n_clasificaciones_correctas = 0
        sesion = tf.compat.v1.Session()
        #Miramos todo el conjunto de los datos de prueba (text_x)
        for i in range(0,datos_preparados.preparacion_de_datos[1].shape[0]):

            #Recuperamos la información
            datosSonar = datos_preparados.preparacion_de_datos[1][i].reshape(1,60)
            clasificacionEsperada = datos_preparados.preparacion_de_datos[3][i].reshape(1,2)

            # Hacemos la clasificación
            prediccion_run = sesion.run(clasificaciones, feed_dict={datos_preparados.neuEntrada:datosSonar})

            #Se calcula la precisión de la clasificación con ayuda de la fórmula antes establecida
            accuracy_run = sesion.run(formula_precision, feed_dict={datos_preparados.neuEntrada:datosSonar, datos_preparados.varReales:clasificacionEsperada})


            #Se muestra para observación la clase original y la clasificación realizada
            print(i,"Clase esperada: ", int(sesion.run(datos_preparados.varReales[i][1],feed_dict={datos_preparados.varReales:datos_preparados.preparacion_de_datos[3]})), " Clasificación: ", prediccion_run[0] )

            n_clasificaciones = n_clasificaciones+1
            if(accuracy_run*100 ==100):
                n_clasificaciones_correctas = n_clasificaciones_correctas+1


        print("-------------")
        print("Precisión en los datos de pruebas = "+str((n_clasificaciones_correctas/n_clasificaciones)*100)+"%")


        #-------------------------------------------------------------------------
        # PRECISIÓN EN LOS DATOS DE APRENDIZAJE
        #-------------------------------------------------------------------------

        n_clasificaciones = 0
        n_clasificaciones_correctas = 0
        for i in range(0,datos_preparados.preparacion_de_datos[0].shape[0]):

            # Recuperamos la información
            datosSonar = datos_preparados.preparacion_de_datos[0][i].reshape(1, 60)
            clasificacionEsperada = datos_preparados.preparacion_de_datos[2][i].reshape(1, 2)

            # Realizamos la clasificación
            prediccion_run = sesion.run(clasificaciones, feed_dict={datos_preparados.neuEntrada: datosSonar})

            # Calculamos la precisión de la clasificación con la ayuda de la fórmula antes establecida
            accuracy_run = sesion.run(formula_precision, feed_dict={datos_preparados.neuEntrada: datosSonar, datos_preparados.varReales: clasificacionEsperada})

            n_clasificaciones = n_clasificaciones + 1
            if (accuracy_run * 100 == 100):
                n_clasificaciones_correctas = n_clasificaciones_correctas + 1


        print("Precisión en los datos de aprendizaje = " + str((n_clasificaciones_correctas / n_clasificaciones) * 100) + "%")


        #-------------------------------------------------------------------------
        # PRECISIÓN EN EL CONJUNTO DE LOS DATOS
        #-------------------------------------------------------------------------


        n_clasificaciones = 0;
        n_clasificaciones_correctas = 0
        for i in range(0,207):

            prediccion_run = sesion.run(clasificaciones, feed_dict={datos_preparados.neuEntrada:X[i].reshape(1,60)})
            accuracy_run = sesion.run(formula_precision, feed_dict={datos_preparados.neuEntrada:X[i].reshape(1,60), datos_preparados.varReales:Y[i].reshape(1,2)})

            n_clasificaciones = n_clasificaciones + 1
            if (accuracy_run * 100 == 100):
                n_clasificaciones_correctas = n_clasificaciones_correctas + 1


        print("Precisión en el conjunto de los datos = " + str((n_clasificaciones_correctas / n_clasificaciones) * 100) + "%")




        sesion.close()




def main():
    tf_neuronas_entradas_X = tf.compat.v1.placeholder(tf.float32,[None, 60])

    tf_valores_reales_Y = tf.compat.v1.placeholder(tf.float32,[None, 2])
    observaciones = pnd.read_csv("datas/sonar.all-data.csv")
    epochs = 600
    cantidad_neuronas_entrada = 60
    cantidad_neuronas_salida = 2
    tasa_aprendizaje = 0.01

    pesos = {
            # 60 neuronas de entradas hacia 24 Neuronas de la capa oculta
            'capa_entrada_hacia_oculta': tf.Variable(tf.random_normal([60, 31]), tf.float32),

            # 24 neuronas de la capa oculta hacia 2 de la capa de salida
            'capa_oculta_hacia_salida': tf.Variable(tf.random_normal([31, 2]), tf.float32),
        }

    pesos_sesgo = {
            #1 sesgo de capa de entrada hacia las 24 neuronas de la capa oculta
            'peso_sesgo_capa_entrada_hacia_oculta': tf.Variable(tf.zeros([31]), tf.float32),

            #1 sesgo de la capa oculta hacia las 2 neuronas de la capa de salida
            'peso_sesgo_capa_oculta_hacia_salida': tf.Variable(tf.zeros([2]), tf.float32),
        }


    neurona31=neurona31(tf_neuronas_entradas_X, tf_valores_reales_Y ,observaciones_en_entradas, pesos, pesos_sesgo, observaciones)
    red = neurona31.red_neuronas_multicapa(tf_neuronas_entradas_X, pesos, pesos_sesgo)



