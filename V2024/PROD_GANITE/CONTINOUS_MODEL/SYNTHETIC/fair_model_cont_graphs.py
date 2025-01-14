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
import matplotlib.pyplot as plt
keras.config.disable_interactive_logging()


prefix_1= "/Users/german.sanchez/Desktop/PAPER1/PROD_GANITE/CONTINOUS_MODEL/SYNTHETIC/RESULTS/"
res1 = pd.read_csv(prefix_1+"MANUAL_RESULTS.csv", delimiter=",",skiprows=0)
res1.columns=["Alpha","Loss_YCF_Y","ATE_YCF_Y","Loss_YCF_MANUAL","ATE_YCF_MANUAL","Loss_YCF_RAND","ATE_YCF_RAND" ]
res2 = pd.read_csv(prefix_1+"GANITE_RESULTS.csv", delimiter=",",skiprows=0)
res2.columns=["Alpha_Ganite", "Alpha"	, "Loss_YCF_Ganite", 	"ATE_YCF_Ganite"]

print(res1)


plt.xlabel('Alpha')
plt.ylabel('Loss')
plt.title('Loss of the fair predictor')
plt.plot(res1["Alpha"].tolist(),res1["Loss_YCF_Y"].tolist(), color='blue', linewidth = 3,  label = 'YCF=Y')
plt.plot(res1["Alpha"].tolist(),res1["Loss_YCF_MANUAL"].tolist(), color='red', linewidth = 5,  label = 'YCF=MANUAL')
plt.plot(res1["Alpha"].tolist(),res1["Loss_YCF_RAND"].tolist(), color='green', linewidth = 5,  label = 'YCF=RAND')
plt.legend()
plt.savefig("Loss"+".png")
plt.close()


plt.xlabel('Alpha')
plt.ylabel('ATE')
plt.title('ATE of the fair predictor')
plt.plot(res1["Alpha"].tolist(),res1["ATE_YCF_Y"].tolist(), color='blue', linewidth = 3,  label = 'YCF=Y')
plt.plot(res1["Alpha"].tolist(),res1["ATE_YCF_MANUAL"].tolist(), color='red', linewidth = 5,  label = 'YCF=MANUAL')
plt.plot(res1["Alpha"].tolist(),res1["ATE_YCF_RAND"].tolist(), color='green', linewidth = 5,  label = 'YCF=RAND')
plt.legend()
plt.savefig("ATE"+".png")
plt.close()

print(res2)


plt.xlabel('Alpha')
plt.ylabel('ATE')
plt.title('GANITE ATE')
alphas=[0,0.033, 0.066,0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7 ,0.8, 0.9, 1]
alphas_ganite=[-50,-40,-30,-20,-10,-9,-8,-7,-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6,7,8,9,10,20,30,40,50]
for j in alphas_ganite:
      plt.plot(res2[(res2["Alpha_Ganite"]==j)]["Alpha"].tolist(),res2[(res2["Alpha_Ganite"]==j)]["ATE_YCF_Ganite"].tolist(), linewidth = 3,  label = 'A_Ganite: '+str(j))
plt.legend()
plt.savefig("GANITE_ATE"+".png")
plt.close()


test=res2.groupby("Alpha_Ganite").mean().reset_index()
plt.xlabel('Alpha Ganite')
plt.ylabel('ATE')
plt.title('AVERAGE GANITE ATE')
plt.plot(test["Alpha_Ganite"].tolist(),test["ATE_YCF_Ganite"].tolist(), linewidth = 3,  label = 'ATE FOR A SET ALPHA GANITE')
plt.legend()
plt.savefig("GANITE_ATE_AVG"+".png")
plt.close()

test=res2[res2["Alpha"]==1].groupby("Alpha_Ganite").mean().reset_index()
plt.xlabel('Alpha Ganite')
plt.ylabel('ATE')
plt.title('GANITE ATE FOR FIXED INTERNAL ALPHA')
plt.plot(test["Alpha_Ganite"].tolist(),test["ATE_YCF_Ganite"].tolist(), linewidth = 3,  label = 'ATE FOR INTERNAL ALPHA 1')
plt.legend()
plt.savefig("GANITE_ATE_FIXED_ALPHA"+".png")
plt.close()

test=res2[(res2["Alpha_Ganite"]>=3)&(res2["Alpha_Ganite"]<=3)].groupby("Alpha").mean().reset_index()
plt.xlabel('Alpha ')
plt.ylabel('ATE')
plt.title('GANITE ATE FOR FIXED ALPHA GANITE')
plt.plot(test["Alpha"].tolist(),test["ATE_YCF_Ganite"].tolist(), linewidth = 3,  label = 'ATE FOR GANITE ALPHA 3')
plt.legend()
plt.savefig("GANITE_ATE_FIXED_ALPHA"+".png")
plt.close()