
import pandas as pnd
import tensorflow as tf
import matplotlib.pyplot as plt
from preparacion_datos_31 import preparacion_datos



class neurona31():

    def __init__(self ,tasa_aprendizaje, epochs, red):
        self.tasa_aprendizaje=tasa_aprendizaje
        self.epochs=epochs
        self.red=red

    #---------------------------------------------
    # FUNCIÓN DE CREACIÓN DE LA RED NEURONAL
    #---------------------------------------------

    def aprendizaje(self):

        observaciones = pnd.read_csv("datas/sonar.all-data.csv")
        X = observaciones[observaciones.columns[0:60]].values()
        y = observaciones[observaciones.columns[60]]

        datos_preparados=preparacion_datos(observaciones, X, y)

        def red_neuronas_multicapa():
            #Cálculo de la activación de la primera capa
            primera_activacion = tf.sigmoid(tf.matmul(datos_preparados.neuEntrada(), datos_preparados.pesos()['capa_entrada_hacia_oculta']) + datos_preparados.peso_sesgo()['peso_sesgo_capa_entrada_hacia_oculta'])

            #Cálculo de la activación de la segunda capa
            activacion_capa_oculta = tf.sigmoid(tf.matmul(primera_activacion, datos_preparados.pesos()['capa_oculta_hacia_salida']) + datos_preparados.peso_sesgo()['peso_sesgo_capa_oculta_hacia_salida'])

            return activacion_capa_oculta

        self.red = red_neuronas_multicapa()

        #Función de error de media cuadrática MSE
        funcion_error = tf.reduce_sum(tf.pow(datos_preparados.varReales()-self.red,2))

        optimizador = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=self.tasa_aprendizaje).minimize(funcion_error)
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
        for i in range(self.epochs):

            #Realización del aprendizaje con actualización de los pesos
            sesion.run(optimizador, feed_dict = {datos_preparados.neuEntrada(): datos_preparados.preparacion_de_datos[0], datos_preparados.varReales():datos_preparados.preparacion_de_datos[2]})

            #Calcular el error
            MSE = sesion.run(funcion_error, feed_dict = {datos_preparados.neuEntrada(): datos_preparados.preparacion_de_datos[0], datos_preparados.varReales():datos_preparados.preparacion_de_datos[2]})

            #Visualización de la información
            Grafica_MSE.append(MSE)
            print("EPOCH (" + str(i) + "/" + str(self.epochs) + ") -  MSE: "+ str(MSE))


        #Visualización gráfica
        plt.plot(Grafica_MSE)
        plt.ylabel('MSE')
        plt.show()


    def verificar_aprendizaje(self):
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
        for i in range(0,datos_preparados.preparacion_de_datos()[1].shape[0]):

            #Recuperamos la información
            datosSonar = datos_preparados.preparacion_de_datos()[1][i].reshape(1,60)
            clasificacionEsperada = datos_preparados.preparacion_de_datos()[3][i].reshape(1,2)

            # Hacemos la clasificación
            prediccion_run = sesion.run(clasificaciones, feed_dict={datos_preparados.neuEntrada():datosSonar})

            #Se calcula la precisión de la clasificación con ayuda de la fórmula antes establecida
            accuracy_run = sesion.run(formula_precision, feed_dict={datos_preparados.neuEntrada():datosSonar, datos_preparados.varReales():clasificacionEsperada})


            #Se muestra para observación la clase original y la clasificación realizada
            print(i,"Clase esperada: ", int(sesion.run(datos_preparados.varReales()[i][1],feed_dict={datos_preparados.varReales():datos_preparados.preparacion_de_datos()[3]})), " Clasificación: ", prediccion_run[0] )

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
        for i in range(0,datos_preparados.preparacion_de_datos()[0].shape[0]):

            # Recuperamos la información
            datosSonar = datos_preparados.preparacion_de_datos()[0][i].reshape(1, 60)
            clasificacionEsperada = datos_preparados.preparacion_de_datos()[2][i].reshape(1, 2)

            # Realizamos la clasificación
            prediccion_run = sesion.run(clasificaciones, feed_dict={datos_preparados.neuEntrada(): datosSonar})

            # Calculamos la precisión de la clasificación con la ayuda de la fórmula antes establecida
            accuracy_run = sesion.run(formula_precision, feed_dict={datos_preparados.neuEntrada(): datosSonar, datos_preparados.varReales(): clasificacionEsperada})

            n_clasificaciones = n_clasificaciones + 1
            if (accuracy_run * 100 == 100):
                n_clasificaciones_correctas = n_clasificaciones_correctas + 1


        print("Precisión en los datos de aprendizaje = " + str((n_clasificaciones_correctas / n_clasificaciones) * 100) + "%")


        #-------------------------------------------------------------------------
        # PRECISIÓN EN EL CONJUNTO DE LOS DATOS
        #-------------------------------------------------------------------------


        n_clasificaciones = 0
        n_clasificaciones_correctas = 0
        for i in range(0,207):

            prediccion_run = sesion.run(clasificaciones, feed_dict={datos_preparados.neuEntrada():X[i].reshape(1,60)})
            accuracy_run = sesion.run(formula_precision, feed_dict={datos_preparados.neuEntrada():X[i].reshape(1,60), datos_preparados.varReales():Y[i].reshape(1,2)})

            n_clasificaciones = n_clasificaciones + 1
            if (accuracy_run * 100 == 100):
                n_clasificaciones_correctas = n_clasificaciones_correctas + 1


        print("Precisión en el conjunto de los datos = " + str((n_clasificaciones_correctas / n_clasificaciones) * 100) + "%")




        sesion.close()




def main():

    observaciones = pnd.read_csv("datas/sonar.all-data.csv")
    X = observaciones[observaciones.columns[0:60]].values
    y = observaciones[observaciones.columns[60]]

    datos_preparados=preparacion_datos(observaciones, X, y)
    epochs = 600
    tasa_aprendizaje = 0.01


    neurona31=neurona31(tasa_aprendizaje, epochs, red)
    neurona31.apredizaje()
    neurona31.verificar_apredizaje()

if __name__ == "__main__":
    main()



