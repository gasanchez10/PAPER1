import pandas as pd
from numpy import loadtxt
from matplotlib import pyplot  as plt
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
#from google.colab import drive
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import os
from sklearn.linear_model import LogisticRegression, LinearRegression
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow
from tensorflow import keras
from itertools import chain
from keras.models import Sequential
from keras.layers import Dense
tensorflow.__version__
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tqdm import tqdm
#keras.config.disable_interactive_logging()

# DEFINE PARAMETERS
epochs_keras=500
iterations_models=3000
alphas=[0,0.033,0.066,0.1,0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
#alphas=[0.1,0.3,0.5,0.7, 0.9]
#alphas=[1]
alphas_ganite=[-50, -40,-30,-20, -10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50]
prefix="/Users/german.sanchez/Desktop/PAPER1/PROD_GANITE/DATA_PROD/BINARY/TWINS/ORIGINAL/"
prefix_ganite="/Users/german.sanchez/Desktop/PAPER1/PROD_GANITE/DATA_PROD/BINARY/TWINS/GANITE/"
prefix_results="/Users/german.sanchez/Desktop/PAPER1/PROD_GANITE/BINARY_MODEL/TWINS/RESULTS/"
version='vc2.csv'
suffix="_prod_"+version


def doubly_robust(df, X, T, Y, X_PS):
    ps = LogisticRegression(C=1e6, max_iter=100000).fit(df[X_PS], df[T]).predict_proba(df[X_PS])[:, 1]
    mu0 = LinearRegression().fit(df.query(f"{T}==0")[X], df.query(f"{T}==0")[Y]).predict(df[X])
    mu1 = LinearRegression().fit(df.query(f"{T}==1")[X], df.query(f"{T}==1")[Y]).predict(df[X])
    return (
        np.mean(df[T]*(df[Y] - mu1)/ps + mu1) -
        np.mean((1-df[T])*(df[Y] - mu0)/(1-ps) + mu0)
    )

def estructuras(X, Y):
    input_unit = X.shape[1] # Dimensión de entrada
    hidden_unit = 4 # Unidades de capa oculta
    Y=Y.reshape([Y.shape[0],1])
    output_unit = Y.shape[1] # Dimensión de variable de salida
    return (input_unit, hidden_unit, output_unit)

def inicializacion(input_unit, hidden_unit, output_unit):
    np.random.seed(10)
    W1 = np.random.randn(input_unit, hidden_unit)*0.01
    b1 = np.zeros((1, hidden_unit))
    W2 = np.random.randn(hidden_unit, output_unit)*0.01
    b2 = np.zeros((1, output_unit))
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    return parameters

def sigmoid(z):
    return 1/(1+np.exp(-z))

def forward_propagation(X, parameters):

    #Se cargan los valores de los parámetros

    W1 = parameters['W1'] #Valor de los pesos de la primera capa
    b1 = parameters['b1'] #Valor de los interceptos de la primera capa
    W2 = parameters['W2'] #Valor de los pesos de la salida
    b2 = parameters['b2'] #Valor de los interceptos de la salida

    Z1 = np.dot(X, W1) + b1 #Cálculo de la transformación afín de la primera capa
    A1 = sigmoid(Z1) #Evaluación de la función sigmoide de la primera capa
    Z2 = np.dot(A1, W2) + b2 #Cálculo de la transformación afín de la salida
    A2 = sigmoid(Z2) #Evaluación de la función sigmoide de la salida
    cache = {"Z1": Z1,"A1": A1,"Z2": Z2,"A2": A2}

    return A2, cache

def loss_cust(A2, Y,Y_CF,model,alpha=1):
    # Muestras de entrenamiento
    n = Y.shape[0]
    # Calcular cross entropy
    res=0
    if model!="eval":
        logprobs = np.multiply(np.log(A2), Y) + np.multiply((1-Y), np.log(1 - A2))
        cost = - np.sum(logprobs) / n
        #cost2= np.sum((np.array(flatten_chain(Y_CF))-np.array(flatten_chain(A2)))**2) /n
        #print("BCE",  cost)
        #print("MSE", cost2)
        Y_CF_flat=flatten_chain(Y_CF)
        A2_flat=flatten_chain(A2)
        cost2_v2=np.sum(np.log(Y_CF_flat/A2_flat)-np.log((1-Y_CF_flat)/(1-A2_flat)))/n
        res=cost*alpha+(1-alpha)*cost2_v2
    else:
        logprobs = np.multiply(np.log(A2), Y) + np.multiply((1-Y), np.log(1 - A2))
        cost = - np.sum(logprobs) / n
        res= cost
    return res

def backward_propagation(parameters, cache, X, Y,Y_CF,alpha=1):

    #Muestras de entrenamiento

    n = X.shape[0]
    W1 = parameters['W1']
    W2 = parameters['W2']
    A1 = cache['A1']
    A2 = cache['A2']

    dA2 = alpha *((A2 - Y) / (A2 * (1 - A2)))/n + (1-alpha)*2 * (A2 - Y_CF)
    dZ2 = alpha *(A2-Y)/n + (1-alpha)*2*(A2 - Y_CF)*A2*(1-A2)/n
    dW2 = np.dot(A1.T,dZ2)
    db2 = np.sum(dZ2, axis=0, keepdims=True)
    dZ1 = np.multiply(np.dot(dZ2,W2.T), A1 - np.power(A1, 2))
    dW1 = np.dot(X.T,dZ1)
    db1 = np.sum(dZ1, axis=0, keepdims=True)
    grads = {"dW1": dW1, "db1": db1, "dW2": dW2,"db2": db2}
    return grads

def gradient_descent(parameters, grads, learning_rate = 0.02):

    #Se capturan los valores actuales para los parámetros de la red neuronal
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']

    #Se capturan los valores del gradiente para cada una de sus componentes
    dW1 = grads['dW1']
    db1 = grads['db1']
    dW2 = grads['dW2']
    db2 = grads['db2']


    #Se actualizan los valores de los parámetros siguiendo la dirección contraria del gradiente
    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1
    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2

    parameters = {"W1": W1, "b1": b1,"W2": W2,"b2": b2}

    return parameters

def neural_network_model(X, Y,Y_CF, hidden_unit, num_iterations = 10000, alpha=1):
    np.random.seed(10)
    input_unit = estructuras(X, Y)[0]
    output_unit = estructuras(X, Y)[2]
    #Se inicializan los parámetros de manera aleatoria
    parameters = inicializacion(input_unit, hidden_unit, output_unit)

    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    costs=[]


    for i in range(0, num_iterations):

        A2, cache = forward_propagation(X, parameters) #Se calcula la salida de la red para los datos utilizados y los valores de los parámetros para cada iteración
        cost = loss_cust(A2, Y, Y_CF,"train", alpha) #Se calcula la función de costos
        costs.append(cost) #Se guardan el valor de la función de costos para cada iteración

        grads = backward_propagation(parameters, cache, X, Y, Y_CF, alpha) #Se calcula el gradiente de la función de costos utilizando el método de backpropagation para cada iteración

        parameters = gradient_descent(parameters, grads) #Se actualiza el valor de los parámetros de acuerdo al gradiente calculado en cada iteración

    return parameters, costs

def prediction(parameters, X):
    A2, cache = forward_propagation(X, parameters)
    predictions = A2
    return predictions

def flatten_chain(matrix):
    return np.array(list(chain.from_iterable(matrix)))

def cross_entropy(yreal, ypred):
    ypred=flatten_chain(ypred)
    with np.errstate(divide='ignore', invalid='ignore'):
        return -1*(yreal*np.log(ypred) + (1-yreal)*np.log(1-ypred))

def evaluate_keras(probabilities, y, Y_CF, model="train", alpha=1):
    loss = loss_cust(probabilities,y, Y_CF, model, alpha)
    predictions = (probabilities > 0.5).astype(int)
    accuracy=np.mean((predictions)==y)
    return loss, accuracy

def metrics_calculator(reference):
  random.seed(10)
  model = Sequential()
  model.add(Dense(10, activation='sigmoid'))
  model.add(Dense(1,activation='sigmoid'))
  model.compile( loss='binary_crossentropy', metrics=['accuracy'])
  history=model.fit(X_train, reference, epochs=epochs_keras, batch_size=25)
  aux=(evaluate_keras((model.predict(X_test)), test_y.reshape([test_y.shape[0],1]), [], "eval"))
  loss=aux[0]
  acc=aux[1]
  X_test_df=pd.DataFrame(X_test)
  X_test_df_o=pd.DataFrame(X_test_o)
  prediccion = model.predict(X_test)
  predictions_df=pd.DataFrame(prediccion,columns=["Output"])
  treatment_df=pd.DataFrame(test_t, columns=["Treatment"])
  T = 'Treatment'
  Y = 'Output'
  X = X_test_df_o.columns
  X_PS = [22]
  full_test_df=pd.concat([X_test_df_o, treatment_df, predictions_df], axis=1)
  full_test_df["Treatment"]=full_test_df["Treatment"].astype('bool')
  ate=doubly_robust(full_test_df, X, T, Y,X_PS)
  del model
  return [loss, ate, acc]

def metrics_calculator_v2(reference, reftest, alpha):
  random.seed(10)
  (input_unit, hidden_unit, output_unit) = estructuras(X_train, y_train)
  X=X_train
  Y=y_train
  hidden_unit=10
  num_iterations=15000
  input_unit = estructuras(X, Y)[0]
  output_unit = estructuras(X, Y)[2]
  #Se inicializan los parámetros de manera aleatoria
  parameters = inicializacion(input_unit, hidden_unit, output_unit)

  W1 = parameters['W1']
  b1 = parameters['b1']
  W2 = parameters['W2']
  b2 = parameters['b2']
  costs=[]

  parameters, costs = neural_network_model(X_train, y_train.reshape([y_train.shape[0],1]), reftest.reshape([reftest.shape[0],1]), 10, iterations_models, alpha)
  #predictions = prediction(parameters, X_train)
  #print ('Accuracy Train: %d' % float((np.dot(reference.reshape([reference.shape[0],1]).T, predictions) + np.dot(1 - reference.reshape([reference.shape[0],1]).T, 1 - predictions))/float(reference.reshape([reference.shape[0],1]).size)*100) + '%')
  predictions = prediction(parameters, X_test)
  #print ('Accuracy Test: %d' % float((np.dot(test_y.reshape([test_y.shape[0],1]).T, predictions) + np.dot(1 - test_y.reshape([test_y.shape[0],1]).T, 1 - predictions))/float(test_y.reshape([test_y.shape[0],1]).size)*100) + '%')
  aux=(evaluate_keras((predictions), test_y.reshape([test_y.shape[0],1]), reftest.reshape([reftest.shape[0],1]), "eval", alpha))
  loss=aux[0]
  acc=aux[1]
  X_test_df=pd.DataFrame(X_test)
  X_test_df_o=pd.DataFrame(X_test_o)
  predictions_df=pd.DataFrame(predictions,columns=["Output"])
  treatment_df=pd.DataFrame(test_t, columns=["Treatment"])
  T = 'Treatment'
  Y = 'Output'
  X = X_test_df_o.columns
  X_PS = range(2)
  full_test_df=pd.concat([X_test_df_o, treatment_df, predictions_df], axis=1)
  full_test_df["Treatment"]=full_test_df["Treatment"].astype('bool')
  ate=doubly_robust(full_test_df, X, T, Y,X_PS)
  return [loss, ate, acc]

def metrics_calculator_vganite(reference, reftest, alpha):
  random.seed(10)
  (input_unit, hidden_unit, output_unit) = estructuras(X_train_ganite, y_train_ganite)
  X=X_train_ganite
  Y=y_train_ganite
  hidden_unit=10
  num_iterations=15000
  input_unit = estructuras(X, Y)[0]
  output_unit = estructuras(X, Y)[2]
  #Se inicializan los parámetros de manera aleatoria
  parameters = inicializacion(input_unit, hidden_unit, output_unit)

  W1 = parameters['W1']
  b1 = parameters['b1']
  W2 = parameters['W2']
  b2 = parameters['b2']
  costs=[]

  parameters, costs = neural_network_model(X_train_ganite, y_train_ganite.reshape([y_train_ganite.shape[0],1]), reftest.reshape([reftest.shape[0],1]), 10, iterations_models, alpha)
  #predictions = prediction(parameters, X_train)
  #print ('Accuracy Train: %d' % float((np.dot(reference.reshape([reference.shape[0],1]).T, predictions) + np.dot(1 - reference.reshape([reference.shape[0],1]).T, 1 - predictions))/float(reference.reshape([reference.shape[0],1]).size)*100) + '%')
  predictions = prediction(parameters, X_test_ganite)
  #print ('Accuracy Test: %d' % float((np.dot(test_y.reshape([test_y.shape[0],1]).T, predictions) + np.dot(1 - test_y.reshape([test_y.shape[0],1]).T, 1 - predictions))/float(test_y.reshape([test_y.shape[0],1]).size)*100) + '%')
  aux=(evaluate_keras((predictions), test_y_ganite.reshape([test_y_ganite.shape[0],1]), reftest.reshape([reftest.shape[0],1]), "eval", alpha))
  loss=aux[0]
  acc=aux[1]
  X_test_df=pd.DataFrame(X_test_ganite)
  X_test_df_o=pd.DataFrame(X_test_o_ganite)
  predictions_df=pd.DataFrame(predictions,columns=["Output"])
  treatment_df=pd.DataFrame(test_t_ganite, columns=["Treatment"])
  T = 'Treatment'
  Y = 'Output'
  X = X_test_df_o.columns
  X_PS = range(2)
  full_test_df=pd.concat([X_test_df_o, treatment_df, predictions_df], axis=1)
  full_test_df["Treatment"]=full_test_df["Treatment"].astype('bool')
  ate=doubly_robust(full_test_df, X, T, Y,X_PS)
  return [loss, ate, acc]


X_train = pd.read_csv(prefix+"train_x"+suffix, delimiter=",",skiprows=0,  index_col=0).values
print(X_train.shape)
train_t = pd.read_csv(prefix+"train_t"+suffix, delimiter=",",skiprows=0,  index_col=0).values
print(train_t.shape)
y_train = pd.read_csv(prefix+"train_y"+suffix, delimiter=",",skiprows=0,  index_col=0).values.astype(float)
print(y_train.shape)
train_cf_y = pd.read_csv(prefix+"train_cf_y"+suffix, delimiter=",",skiprows=0,  index_col=0).values
print(train_cf_y.shape)
train_cf_manual = pd.read_csv(prefix+"train_cf_manual"+suffix, delimiter=",",skiprows=0,  index_col=0).values
print(train_cf_manual.shape)
train_cf_random = pd.read_csv(prefix+"train_cf_rand"+suffix, delimiter=",",skiprows=0,  index_col=0).values
print(train_cf_random.shape)
X_test = pd.read_csv(prefix+"test_x"+suffix, delimiter=",",skiprows=0,  index_col=0).values
print(X_test.shape)
X_test_o = pd.read_csv(prefix+"test_x"+suffix, delimiter=",",skiprows=0,  index_col=0).values
print(X_test_o.shape)
test_t = pd.read_csv(prefix+"test_t"+suffix, delimiter=",",skiprows=0,  index_col=0).values
print(test_t.shape)
test_y = pd.read_csv(prefix+"test_y"+suffix, delimiter=",",skiprows=0,  index_col=0).values
print(test_y.shape)
test_cf_y = pd.read_csv(prefix+"test_cf_y"+suffix, delimiter=",",skiprows=0,  index_col=0).values
print(test_cf_y.shape)
test_cf_manual = pd.read_csv(prefix+"test_cf_manual"+suffix, delimiter=",",skiprows=0,  index_col=0).values
print(test_cf_manual.shape)
test_cf_random = pd.read_csv(prefix+"test_cf_rand"+suffix, delimiter=",",skiprows=0,  index_col=0).values
print(test_cf_random.shape)


print((X_train.shape))
print(X_train[0][22])
train_t=train_t.reshape([9120,1])
X_train = np.append(X_train, train_t, axis=1)
print((X_train.shape))

print((X_test.shape))
print(X_test[0][22])
test_t=test_t.reshape([2280,1])
X_test = np.append(X_test, test_t, axis=1)
print((X_test.shape))



atest1=[]
lossest1=[]
atest2=[]
lossest2=[]
atest3=[]
lossest3=[]
atest4=[]
lossest4=[]
atest5=[]
lossest5=[]


acctest1=[]
acctest2=[]
acctest3=[]
acctest4=[]
acctest5=[]


#Keras
rest2=metrics_calculator((y_train))
lossest2.append(rest2[0])
atest2.append(rest2[1])
acctest2.append(rest2[2])


for e in tqdm(alphas, unit='F'):
  print("-----ALPHA----",e)
  #Manual code
  rest1=metrics_calculator_v2((y_train),(train_cf_y), e)
  lossest1.append(rest1[0])
  atest1.append(rest1[1])
  acctest1.append(rest1[2])
  rest3=metrics_calculator_v2((y_train),(train_cf_manual), e)
  lossest3.append(rest3[0])
  atest3.append(rest3[1])
  acctest3.append(rest3[2])
  rest4=metrics_calculator_v2((y_train),(train_cf_random), e)
  lossest4.append(rest4[0])
  atest4.append(rest4[1])
  acctest4.append(rest4[2])

li = [ alphas]
Alphas=pd.DataFrame(data = li).T
Alphas.columns=["Alpha"]

li = [ lossest1, atest1, acctest1]
No_reg=pd.DataFrame(data = li).T
No_reg.columns=["Loss YCF Y", "ATE YCF Y", "ACC YCF Y"]

li = [ lossest3, atest3, acctest3]
reg_2=pd.DataFrame(data = li).T
reg_2.columns=["Loss YCF MANUAL", "ATE YCF MANUAL", "ACC YCF MANUAL"]


li = [ lossest4, atest4, acctest4]
reg_3=pd.DataFrame(data = li).T
reg_3.columns=["Loss YCF RAND", "ATE YCF RAND", "ACC YCF RAND"]


li = [ lossest2, atest2, acctest2]
reg_1=pd.DataFrame(data = li).T
reg_1.columns=["Loss KERAS", "ATE KERAS", "ACC KERAS"]

results=[]
results = pd.concat([Alphas,No_reg,reg_2,reg_3], axis=1)
results_keras=[]
results_keras = pd.concat([reg_1], axis=1)

results.to_csv(prefix_results+'MANUAL_RESULTS.csv', index=False)
results_keras.to_csv(prefix_results+'KERAS_RESULTS.csv', index=False)
print(results)




def data_loader(alpha_ganite):  
    suf="_alpha_"+str(alpha_ganite)+'_twins'+version
    X_train_ganite = np.loadtxt(prefix_ganite+"train_x_ganite"+suf, delimiter=",",skiprows=1)
    #print(X_train_ganite.shape)
    train_t_ganite = np.loadtxt(prefix_ganite+"train_t_ganite"+suf, delimiter=",",skiprows=1)
    #print(train_t_ganite.shape)
    y_train_ganite = np.loadtxt(prefix_ganite+"train_y_ganite"+suf, delimiter=",",skiprows=1)
    #print(y_train_ganite.shape)
    train_potential_y_ganite = np.loadtxt(prefix_ganite+"train_potential_y_ganite"+suf, delimiter=",",skiprows=1)
    #print(train_potential_y_ganite.shape)
    X_test_ganite = np.loadtxt(prefix_ganite+"test_x_ganite"+suf, delimiter=",",skiprows=1)
    #print(X_test_ganite.shape)
    X_test_o_ganite = np.loadtxt(prefix_ganite+"test_x_ganite"+suf, delimiter=",",skiprows=1)
    #print(X_test_o_ganite.shape)
    test_t_ganite = np.loadtxt(prefix_ganite+"test_t_ganite"+suf, delimiter=",",skiprows=1)
    #print(test_t_ganite.shape)
    test_potential_y_ganite = np.loadtxt(prefix_ganite+"test_potential_y_ganite"+suf, delimiter=",",skiprows=1)
    #print(test_potential_y_ganite.shape)
    test_y_hat_ganite = np.loadtxt(prefix_ganite+"test_potest_y_hat_ganite"+suf, delimiter=",",skiprows=1)
    #print(test_y_hat_ganite.shape)
    test_y_ganite = np.loadtxt(prefix_ganite+"test_y_ganite"+suf, delimiter=",",skiprows=1)
    #print(test_y_ganite.shape)
    train_y_hat_ganite=test_y_hat_ganite[0:9120]
    test_y_hat_ganite=test_y_hat_ganite[9120:9120+2280]
    # Extract counterfactuals
    train_cf_ganite=[]
    for i in range(len(train_t_ganite)):
      train_cf_ganite.append(train_y_hat_ganite[i][1-int(train_t_ganite[i])])
    train_cf_ganite=[x for x in train_cf_ganite]
    test_cf_ganite=[]
    for i in range(len(test_t_ganite)):
      test_cf_ganite.append(test_y_hat_ganite[i][1-int(test_t_ganite[i])])
    test_cf_ganite = np.array([x for x in test_cf_ganite])
    train_cf_ganite=np.array(train_cf_ganite)
    #print(train_cf_ganite.shape)
    #print(test_cf_ganite.shape)
    #print((X_train_ganite.shape))
    train_t_ganite=train_t_ganite.reshape([9120,1])
    X_train_ganite = np.append(X_train_ganite, train_t_ganite, axis=1)
    #print((X_train_ganite.shape))
    #print((X_test_ganite.shape))
    test_t_ganite=test_t_ganite.reshape([2280,1])
    X_test_ganite = np.append(X_test_ganite, test_t_ganite, axis=1)
    #print((X_test_ganite.shape))
    return X_train_ganite, train_t_ganite, y_train_ganite, train_potential_y_ganite, X_test_ganite, X_test_o_ganite, test_t_ganite, test_potential_y_ganite, test_y_hat_ganite, test_y_ganite, train_cf_ganite, test_cf_ganite

for h in tqdm(alphas_ganite, unit='F'):
  print("-----ALPHA GANITE----",h)
  X_train_ganite, train_t_ganite, y_train_ganite, train_potential_y_ganite, X_test_ganite, X_test_o_ganite, test_t_ganite, test_potential_y_ganite, test_y_hat_ganite, test_y_ganite, train_cf_ganite, test_cf_ganite=data_loader(h)
  for e in tqdm(alphas, unit='F'):
    print("-----ALPHA----",e)
    rest5=metrics_calculator_vganite((y_train_ganite),(train_cf_ganite), e)
    lossest5.append(rest5[0])
    atest5.append(rest5[1])
    acctest5.append(rest5[2])

li = [ lossest5, atest5, acctest5]
reg_4=pd.DataFrame(data = li).T
reg_4.columns=["Loss YCF Ganite", "ATE YCF Ganite", "ACC YCF Ganite"]
lix = [np.repeat(i, len(alphas_ganite)).tolist() for i in alphas]
Alphasx=pd.DataFrame(data = [flatten_chain(lix)]).T
Alphasx.columns=["Alpha"]
lix2 = [alphas_ganite for i in alphas]
Alphasx2=pd.DataFrame(data = [flatten_chain(lix2)]).T
Alphasx2.columns=["Alpha_Ganite"]
results_ganite = pd.concat([Alphasx, Alphasx2,reg_4], axis=1)
results_ganite.to_csv(prefix_results+'GANITE_RESULTS.csv', index=False)