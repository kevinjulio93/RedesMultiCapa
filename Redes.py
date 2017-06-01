__author__ = 'Kevin julio - Andres Bobadilla '
# benigno =0 ; maligno = 1
import xlrd
import numpy as np
from pybrain2.tools.shortcuts import buildNetwork
from pybrain2.structure import SoftmaxLayer
from pybrain.structure.modules import TanhLayer
from pybrain.datasets import SupervisedDataSet, ClassificationDataSet
from pybrain2.supervised.trainers import BackpropTrainer as bp
from pybrain2.utilities import percentError
from scipy import diag, arange, meshgrid, where

from copy import copy
from sklearn.preprocessing import normalize

from sklearn import metrics
from sklearn.metrics import confusion_matrix

libro = xlrd.open_workbook('datos.xls')
hoja = libro.sheet_by_index(0)

f = hoja.nrows  # numero de filas
c = hoja.ncols  # numero columnas
datos = np.ones((f, c))

for i in xrange(f):
    keys = [hoja.cell(i, col_index).value for col_index in xrange(c)]

    for j in xrange(c):
        datos[i][j] = keys[j]

entrena = round((f * 0.50))  # 50% de la columna x1 para entrenamiento
validacion = round(f * 0.25)  # 25% de la columna x1 para entrenamiento
prueba = round((f * 0.25))  # 25% de la columna x1 para prueba

entrena = int(entrena)  # convirtiendo a entero; entrena = 32
validacion = int(validacion)  ## convirtiendo a entero; entrena = 32
prueba = int(prueba)  # convirtiendo a entero; prueba = 21

datos_entrena = np.ones((entrena, 30))
datos_val = np.ones((validacion, 30))
datos_prueba = np.ones((prueba, 30))

y_entrena = np.ones((entrena, 1))
y_val = np.ones((validacion, 1))
y_prueba = np.ones((prueba, 1))

# -------------------------------------------X y Y entrenamiento---------------------------------------------------------------#

for i in xrange(entrena):
    for j in xrange(30):
        datos_entrena[i][j] = datos[i][j]

for i in xrange(entrena):
    y_entrena[i][0] = datos[i][30]






#-------------------------------------------X y Y Validacion---------------------------------------------------------------#
for i in xrange(validacion):
    for j in xrange(30):
        datos_val[i][j] = datos[i + entrena][j]

for i in xrange(validacion):
    y_val[i][0] = datos[i + entrena][30]

#-------------------------------------------X y Y Prueba ---------------------------------------------------------------------#

for i in xrange(prueba):
    for j in xrange(30):
        datos_prueba[i][j] = datos[i + (entrena + validacion)][j]

for i in xrange(prueba):
    y_prueba[i][0] = datos[i + (entrena + validacion)][30]

#--------------------------------------constryendo la Red -----------------------------------------------------------------------#



dataset_entrenamiento = ClassificationDataSet(30,1, nb_classes=2)  # Creacion de los dataset para entreamiento, validacion y prueba del algoritmo
dataset_validacion = ClassificationDataSet(30, 1, nb_classes=2)
dataset_prueba = ClassificationDataSet(30, 1, nb_classes=2)
datos_entrena = datos_entrena / datos_entrena.max(axis=0)
datos_val = datos_val/ datos_val.max(axis=0)
datos_prueba = datos_prueba / datos_prueba.max(axis=0)

for i in xrange(entrena):
    dataset_entrenamiento.addSample((datos_entrena[i][0], datos_entrena[i][1], datos_entrena[i][2], datos_entrena[i][3],
                                     datos_entrena[i][4], datos_entrena[i][5],
                                     datos_entrena[i][6], datos_entrena[i][7], datos_entrena[i][8], datos_entrena[i][9],
                                     datos_entrena[i][10], datos_entrena[i][11],
                                     datos_entrena[i][12], datos_entrena[i][13], datos_entrena[i][14], datos_entrena[i][15],
                                     datos_entrena[i][16], datos_entrena[i][17],
                                     datos_entrena[i][18], datos_entrena[i][19], datos_entrena[i][20], datos_entrena[i][21],
                                     datos_entrena[i][22], datos_entrena[i][23],
                                     datos_entrena[i][24], datos_entrena[i][25], datos_entrena[i][26], datos_entrena[i][27],
                                     datos_entrena[i][28], datos_entrena[i][29]), y_entrena[i][0])

for i in xrange(validacion):
    dataset_validacion.addSample(
        (datos_val[i][1], datos_val[i][2], datos_val[i][3], datos_val[i][4], datos_val[i][5],
         datos_val[i][6], datos_val[i][7], datos_val[i][8], datos_val[i][9], datos_val[i][10], datos_val[i][11],
         datos_val[i][12], datos_val[i][13], datos_val[i][14], datos_val[i][15], datos_val[i][16], datos_val[i][17],
         datos_val[i][18], datos_val[i][19], datos_val[i][20], datos_val[i][21], datos_val[i][22], datos_val[i][23],
         datos_val[i][24], datos_val[i][25], datos_val[i][26], datos_val[i][27], datos_val[i][28], datos_val[i][29], datos_val[i][0]),
        y_val[i][0])

for i in xrange(prueba):
    dataset_prueba.addSample((datos_prueba[i][0], datos_prueba[i][1], datos_prueba[i][2], datos_prueba[i][3], datos_prueba[i][4],
                              datos_prueba[i][5], datos_prueba[i][6],
                              datos_prueba[i][7], datos_prueba[i][8], datos_prueba[i][9], datos_prueba[i][10],
                              datos_prueba[i][11], datos_prueba[i][12],
                              datos_prueba[i][13], datos_prueba[i][14], datos_prueba[i][15], datos_prueba[i][16],
                              datos_prueba[i][17], datos_prueba[i][18],
                              datos_prueba[i][19], datos_prueba[i][20], datos_prueba[i][21], datos_prueba[i][22],
                              datos_prueba[i][23], datos_prueba[i][24],
                              datos_prueba[i][25], datos_prueba[i][26], datos_prueba[i][27], datos_prueba[i][28],
                              datos_prueba[i][29]), y_prueba[i][0])





dataset_entrenamiento._convertToOneOfMany(bounds=(0,1))
dataset_validacion._convertToOneOfMany(bounds=(0,1))
dataset_prueba._convertToOneOfMany(bounds=(0,1))




#red = buildNetwork(30, 10, 1, hiddenclass=TanhLayer, outclass=SoftmaxLayer)
momento=0.1
weidecay=0.0001
learngrate=0.1
neuronaoculta=400
red = buildNetwork(dataset_entrenamiento.indim, neuronaoculta, dataset_entrenamiento.outdim, outclass=SoftmaxLayer, bias=True)  # (entradas , capa oculta [5], salida)
entrenar_red = bp(red, dataset=dataset_entrenamiento, momentum=momento, verbose=True, weightdecay=weidecay, learningrate=learngrate)

for i in xrange(100):
    entrenar_red.trainEpochs(1)
    entrenar_resultados = percentError(entrenar_red.testOnClassData(),dataset_entrenamiento['class'])
    valid_resultados = percentError(entrenar_red.testOnClassData(dataset=dataset_validacion),dataset_validacion['class'])
    prueba_resultados = percentError(entrenar_red.testOnClassData(dataset=dataset_prueba),dataset_prueba['class'])



    print "epoch: %4d" % entrenar_red.totalepochs, \
          "  error entrenamiento: %5.2f%%" % entrenar_resultados, \
          "  error prueba: %5.2f%%" % prueba_resultados, \


outsalida = entrenar_red.testOnClassData(dataset_prueba)
print "Salidas Entrenamiento :",entrenar_red.testOnClassData(dataset_entrenamiento)
print "Salidas prueba :",outsalida

print "parametros neurona momento weidecay learngrate",neuronaoculta,momento,weidecay,learngrate





t_cont = 0  # numero de veces que la prediccion es correcta
t_miss = 0  # numero de fallos en la prediccion
t_falsoP=0  #numero de falsos positivos
t_faloN=0   #numero de falsos negativos
for i in range(len(y_prueba)):
    if outsalida[i] == y_prueba[i]:
        t_cont += 1
    else:
        t_miss += 1
        if y_prueba[i] == 1 and outsalida[i] == 0:
            t_falsoP += 1
        else:
            t_faloN += 1

print "Matriz de Confusion Conjunto de Prueba \n"
print confusion_matrix(y_prueba,outsalida, labels=[1 , 0])
print "\n"
print "falsos positivo",t_falsoP
print "falsos Negativos",t_faloN

t_p = t_cont/float(len(y_prueba))

print "Numero de aciertos en prediccion: ", t_cont
print "Numero de fallos en prediccion: ", t_miss
print "porcentaje de prediccion acertada(Accuracy): ", t_p, "\n"

print metrics.classification_report(y_prueba, outsalida)
