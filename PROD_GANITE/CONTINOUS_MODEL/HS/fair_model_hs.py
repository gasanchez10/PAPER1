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
#from google.colab import drive
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
keras.config.disable_interactive_logging()


def doubly_robust(df, X, T, Y, X_PS):
    ps = LogisticRegression(C=1e6, max_iter=100000).fit(df[X_PS], df[T]).predict_proba(df[X_PS])[:, 1]
    mu0 = LinearRegression().fit(df.query(f"{T}==0")[X], df.query(f"{T}==0")[Y]).predict(df[X])
    mu1 = LinearRegression().fit(df.query(f"{T}==1")[X], df.query(f"{T}==1")[Y]).predict(df[X])
    return (
        np.mean(df[T]*(df[Y] - mu1)/ps + mu1) -
        np.mean((1-df[T])*(df[Y] - mu0)/(1-ps) + mu0)
    )


def flatten_chain(matrix):
    return list(chain.from_iterable(matrix))


def metrics_calculator(reference, reftest):
  random.seed(10)
  model = Sequential()
  model.add(Dense(10,  activation='relu'))
  model.add(Dense(10, activation='relu'))
  model.add(Dense(1))
  model.compile(loss='mse', optimizer='adam')
  history=model.fit(X_train, reference, epochs=epochs_keras, batch_size=25)
  loss=(model.evaluate(X_test, reftest))
  X_test_df=pd.DataFrame(X_test)
  X_test_df_o=pd.DataFrame(X_test_o)
  prediccion = model.predict(X_test)
  predictions_df=pd.DataFrame(prediccion,columns=["Output"])
  treatment_df=pd.DataFrame(test_t, columns=["Treatment"])
  T = 'Treatment'
  Y = 'Output'
  X = X_test_df_o.columns
  X_PS = range(2)
  full_test_df=pd.concat([X_test_df_o, treatment_df, predictions_df], axis=1)
  full_test_df["Treatment"]=full_test_df["Treatment"].astype('bool')
  ate=doubly_robust(full_test_df, X, T, Y,X_PS)
  del model
  return [loss, ate]


def metrics_calculator_ganite(reference, reftest):
  random.seed(10)
  model = Sequential()
  model.add(Dense(10,  activation='relu'))
  model.add(Dense(10, activation='relu'))
  model.add(Dense(1))
  model.compile(loss='mse', optimizer='adam')
  history=model.fit(X_train_ganite, reference, epochs=epochs_keras, batch_size=25)
  loss=(model.evaluate(X_test_ganite, reftest))
  X_test_df=pd.DataFrame(X_test_ganite)
  X_test_df_o=pd.DataFrame(X_test_o_ganite)
  prediccion = model.predict(X_test_ganite)
  predictions_df=pd.DataFrame(prediccion,columns=["Output"])
  treatment_df=pd.DataFrame(test_t_ganite, columns=["Treatment"])
  T = 'Treatment'
  Y = 'Output'
  X = X_test_df_o.columns
  X_PS = range(2)
  full_test_df=pd.concat([X_test_df_o, treatment_df, predictions_df], axis=1)
  full_test_df["Treatment"]=full_test_df["Treatment"].astype('bool')
  ate=doubly_robust(full_test_df, X, T, Y,X_PS)
  del model
  return [loss, ate]


# DEFINE PARAMETERS
epochs_keras=500
alphas=[0,0.033,0.066,0.1,0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
#alphas=[0.1,0.3,0.5,0.7, 0.9]
#alphas=[1]
alphas_ganite=[-50, -40,-30,-20, -10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50]
prefix="/Users/german.sanchez/Desktop/PAPER1/PROD_GANITE/DATA_PROD/CONTINOUS/HS/ORIGINAL/"
prefix_ganite="/Users/german.sanchez/Desktop/PAPER1/PROD_GANITE/DATA_PROD/CONTINOUS/HS/GANITE/"
prefix_results="/Users/german.sanchez/Desktop/PAPER1/PROD_GANITE/CONTINOUS_MODEL/HS/RESULTS/"
version='v0_continous.csv'
ganite_version='vc2.csv'
suffix="_prod_"+version


def data_loader(alpha_ganite):
    suf="_alpha_"+str(alpha_ganite)+'_cont'+ganite_version
    X_train_ganite = np.loadtxt(prefix_ganite+"train_x_ganite"+suf, delimiter=",",skiprows=1)
    print(X_train_ganite.shape)
    train_t_ganite = np.loadtxt(prefix_ganite+"train_t_ganite"+suf, delimiter=",",skiprows=1)
    print(train_t_ganite.shape)
    y_train_ganite = np.loadtxt(prefix_ganite+"train_y_ganite"+suf, delimiter=",",skiprows=1)
    print(y_train_ganite.shape)
    train_potential_y_ganite = np.loadtxt(prefix_ganite+"train_potential_y_ganite"+suf, delimiter=",",skiprows=1)
    print(train_potential_y_ganite.shape)
    X_test_ganite = np.loadtxt(prefix_ganite+"test_x_ganite"+suf, delimiter=",",skiprows=1)
    print(X_test_ganite.shape)
    X_test_o_ganite = np.loadtxt(prefix_ganite+"test_x_ganite"+suf, delimiter=",",skiprows=1)
    print(X_test_o_ganite.shape)
    test_t_ganite = np.loadtxt(prefix_ganite+"test_t_ganite"+suf, delimiter=",",skiprows=1)
    print(test_t_ganite.shape)
    test_potential_y_ganite = np.loadtxt(prefix_ganite+"test_potential_y_ganite"+suf, delimiter=",",skiprows=1)
    print(test_potential_y_ganite.shape)
    test_y_hat_ganite = np.loadtxt(prefix_ganite+"test_potest_y_hat_ganite"+suf, delimiter=",",skiprows=1)
    print(test_y_hat_ganite.shape)
    test_y_ganite = np.loadtxt(prefix_ganite+"test_y_ganite"+suf, delimiter=",",skiprows=1)
    print(test_y_ganite.shape)
    print(test_y_ganite)
    train_y_hat_ganite=test_y_hat_ganite[0:116]
    test_y_hat_ganite=test_y_hat_ganite[116:145]
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
    print(train_cf_ganite.shape)
    print(test_cf_ganite.shape)
    print((X_train_ganite.shape))
    train_t_ganite=train_t_ganite.reshape([116,1])
    X_train_ganite = np.append(X_train_ganite, train_t_ganite, axis=1)
    print((X_train_ganite.shape))
    print((X_test_ganite.shape))
    test_t_ganite=test_t_ganite.reshape([29,1])
    X_test_ganite = np.append(X_test_ganite, test_t_ganite, axis=1)
    print((X_test_ganite.shape))
    return X_train_ganite, train_t_ganite, y_train_ganite, train_potential_y_ganite, X_test_ganite, X_test_o_ganite, test_t_ganite, test_potential_y_ganite, test_y_hat_ganite, test_y_ganite, train_cf_ganite, test_cf_ganite



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


alphas=[0,0.033,0.066,0.1,0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
alphas_ganite=[-50, -40, -30, -20, -10, -9,-8,-7,-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6,7,8,9,10,20,30,40,50]
#alphas_ganite=[-5,-4,-3,-2,-1,0,1,2,3,4,5]

for h in tqdm(alphas_ganite, unit='F'):
  print("-----ALPHA----",h)
  X_train_ganite, train_t_ganite, y_train_ganite, train_potential_y_ganite, X_test_ganite, X_test_o_ganite, test_t_ganite, test_potential_y_ganite, test_y_hat_ganite, test_y_ganite, train_cf_ganite, test_cf_ganite=data_loader(h)
  for e in tqdm(alphas, unit='F'):
    print(e)
    #Regularization
    rest5=metrics_calculator_ganite(e*y_train_ganite+(1-e)*np.array(train_cf_ganite), test_y_ganite)
    lossest5.append(rest5[0])
    atest5.append(rest5[1])


li = [ lossest5, atest5]
reg_4=pd.DataFrame(data = li).T
reg_4.columns=["Loss YCF Ganite", "ATE YCF Ganite"]
lix = [np.repeat(i, len(alphas)).tolist() for i in alphas_ganite]
Alphasx=pd.DataFrame(data = [flatten_chain(lix)]).T
Alphasx.columns=["Alpha_Ganite"]

lix2 = [alphas for i in alphas_ganite]
Alphasx2=pd.DataFrame(data = [flatten_chain(lix2)]).T
Alphasx2.columns=["Alpha"]
results_ganite = pd.concat([Alphasx, Alphasx2,reg_4], axis=1)
results_ganite.to_csv(prefix_results+'GANITE_RESULTS.csv', index=False)
print(results_ganite)