from preparacion_datos_12 import datos_preparados
import tensorflow as tf
import pandas as pnd

class neurona12():

    def __init__(self ,tasa_aprendizaje, epochs):
        self.tasa_aprendizaje=tasa_aprendizaje
        self.epochs=epochs



    #---------------------------------------------
    # FUNCIÓN DE CREACIÓN DE LA RED NEURONAL
    #---------------------------------------------

    def creacion_red_neuronal(self):

        observaciones = pnd.read_csv("datas/sonar.all-data.csv")
        X = observaciones[observaciones.columns[0:60]].values()
        y = observaciones[observaciones.columns[60]]

        datos_preparados_12=datos_preparados(observaciones, X, y)

        def red_neuronas_multicapa():

            #Cálculo de la activación de la primera capa
            primera_activacion = tf.math.sigmoid(tf.linalg.matmul(datos_preparados.neuEntrada(),
                                            datos_preparados.pesos()['capa_entrada_hacia_oculta']) +
                                                        datos_preparados.pesos_sesgo()['datos_preparados.peso_sesgo()_capa_entrada_hacia_oculta'])

            #Cálculo de la activación de la segunda capa
            activacion_capa_oculta = tf.math.sigmoid(tf.linalg.matmul(primera_activacion,
                                                    datos_preparados.pesos()['capa_oculta_hacia_salida']) +
                                                    datos_preparados.pesos_sesgo()['datos_preparados.peso_sesgo()_capa_oculta_hacia_salida'])

            return activacion_capa_oculta
    #---------------------------------------------
    # CREACIÓN DE LA RED NEURONAL
    #---------------------------------------------
        red = red_neuronas_multicapa()


        #---------------------------------------------
        # ERROR Y OPTIMIZACIÓN
        #---------------------------------------------

        #Función de error de media cuadrática MSE
        funcion_error = tf.reduce_sum(tf.pow(datos_preparados.varReales()-red,2))

        #Función de precisión
        #funcion_precision = tf.metrics.accuracy(labels=datos_preparados.varReales(),predictions=red)


        #Descenso de gradiente con una tasa de aprendizaje fijada a 0,1
        optimizador = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=self.tasa_aprendizaje).minimize(funcion_error)


        #---------------------------------------------
        # APRENDIZAJE
        #---------------------------------------------

        #Inicialización de la variable
        init = tf.compat.v1.global_variables_initializer()

        #Inicio de una sesión de aprendizaje
        sesion = tf.compat.v1.Session()
        sesion.run(init)

        #Para la realización de la gráfica para la MSE
        Grafica_MSE=[]


        #Para cada epoch
        for i in range(self.epochs):

            #Realización del aprendizaje con actualización de los datos_preparados.pesos()
            sesion.run(optimizador, feed_dict = {datos_preparados.neuEntrada(): datos_preparados.train_test(self ,0),
                                                            datos_preparados.varReales():datos_preparados.train_test(self ,2)})

            #Calcular el error de aprendizaje
            MSE = sesion.run(funcion_error, feed_dict = {datos_preparados.neuEntrada(): datos_preparados.train_test(self ,0),
                                                                    datos_preparados.varReales():datos_preparados.train_test(self ,2)})

            #Visualización de la información
            Grafica_MSE.append(MSE)
            print("EPOCH (" + str(i) + "/" + str(self.epochs) + ") -  MSE: "+ str(MSE))


        #Visualización gráfica MSE
        import matplotlib.pyplot as plt
        plt.plot(Grafica_MSE)
        plt.ylabel('MSE')
        plt.show()


        #---------------------------------------------
        # VERIFICACIÓN DEL APRENDIZAJE
        #---------------------------------------------

        #Las probabilidades de cada clase 'Mina' o 'roca' procedentes del aprendizaje
        # se almacenan en el modelo.
        #Con la ayuda de tf.argmax, se recuperan los índices de las probabilidades
        # más elevados para cada observación
        #Ejemplo: Si para una observación tenemos [0.56, 0.89] enviará 1 porque el valor
        # más elevado se encuentra en el índice 1
        #Ejemplo: Si para una observación tenemos [0.90, 0.34] enviará 0 porque el valor
        # más elevado se encuentra en el índice 0
        clasificaciones = tf.argmax(red, 1)

        #En la tabla de valores reales:
        #Las minas están codificadas como [1,0] y el índice que tiene el mayor valor es 0
        #Las rocas tienen el valor [0,1] y el índice que tiene el mayor valor es 1

        #Si la clasificación es [0.90, 0.34], el índice que tiene el mayor valor es 0
        #Si es una mina [1,0], el índice que tiene el mayor valor es 0
        #Si los dos índices son idénticos, entonces se puede afirmar que es una buena clasificación
        formula_calculo_clasificaciones_correctas = tf.equal(clasificaciones, tf.argmax(datos_preparados.varReales(),1))


        #La precisión se calcula haciendo la media (tf.mean)
        # de las clasificaciones buenas (después de haberlas convertido en decimales tf.cast, tf.float32)
        formula_precision = tf.reduce_mean(tf.cast(formula_calculo_clasificaciones_correctas, tf.float32))



        #-------------------------------------------------------------------------
        # PRECISIÓN EN LOS DATOS DE PRUEBAS
        #-------------------------------------------------------------------------

        n_clasificaciones = 0
        n_clasificaciones_correctas = 0

        #Se mira el conjunto de los datos de prueba (text_x)
        for i in range(0,datos_preparados.train_test(self , 1).shape[0]):

            #Se recupera la información
            datosSonar = datos_preparados.train_test(self ,1)[i].reshape(1,60)
            clasificacionEsperada = datos_preparados.train_test(self ,1)[i].reshape(1,2)

            # Se realiza la clasificación
            prediccion_run = sesion.run(clasificaciones, feed_dict={datos_preparados.neuEntrada():datosSonar})

            #Se calcula la precisión de la clasificación con la ayuda de la fórmula establecida antes
            accuracy_run = sesion.run(formula_precision, feed_dict={datos_preparados.neuEntrada():datosSonar,
                                                                        datos_preparados.varReales():clasificacionEsperada})


            #Se muestra para observación la clase original y la clasificación realizada
            print(i,"Clase esperada: ", int(sesion.run(datos_preparados.varReales()[i][1],feed_dict={datos_preparados.varReales():
                                                                datos_preparados.train_test(self ,3)})), "Clasificación: ", prediccion_run[0] )

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
        for i in range(0,datos_preparados.train_test(self ,0).shape[0]):

            # Recuperamos la información
            datosSonar = datos_preparados.train_test(self ,0)[i].reshape(1, 60)
            clasificacionEsperada = datos_preparados.train_test(self ,2)[i].reshape(1, 2)

            # Realizamos la clasificación
            prediccion_run = sesion.run(clasificaciones, feed_dict={datos_preparados.neuEntrada(): datosSonar})

            # Calculamos la precisión de la clasificación con la ayuda de la fórmula establecida antes
            accuracy_run = sesion.run(formula_precision, feed_dict={datos_preparados.neuEntrada(): datosSonar,
                                                    datos_preparados.varReales(): clasificacionEsperada})

            n_clasificaciones = n_clasificaciones + 1
            if (accuracy_run * 100 == 100):
                n_clasificaciones_correctas = n_clasificaciones_correctas + 1


        print("Precisión en los datos de aprendizaje = " + str((n_clasificaciones_correctas / n_clasificaciones) * 100) + "%")


        #-------------------------------------------------------------------------
        # PRECISIÓN EN EL CONJUNTO DE DATOS
        #-------------------------------------------------------------------------


        n_clasificaciones = 0
        n_clasificaciones_correctas = 0
        for i in range(0,207):

            prediccion_run = sesion.run(clasificaciones, feed_dict={datos_preparados.neuEntrada():
                                                                    datos_preparados.train_test(self ,5)[i].reshape(1,60)})
            accuracy_run = sesion.run(formula_precision, feed_dict={datos_preparados.neuEntrada():
                                                                    datos_preparados.train_test(self ,5)[i].reshape(1,60),
                                                                    datos_preparados.varReales():datos_preparados.train_test(self, 4)[i].reshape(1,2)})

            n_clasificaciones = n_clasificaciones + 1
            if (accuracy_run * 100 == 100):
                n_clasificaciones_correctas = n_clasificaciones_correctas + 1


        print("Precisión en el conjunto de datos = " + str((n_clasificaciones_correctas / n_clasificaciones) * 100) + "%")




        sesion.close()


def main():
    tasa_aprendizaje = 0.01
    epochs = 300

    prueba_neurona12=neurona12(tasa_aprendizaje, epochs )
    prueba_neurona12.creacion_red_neuronal()

if __name__ == '__main__':
    main()
