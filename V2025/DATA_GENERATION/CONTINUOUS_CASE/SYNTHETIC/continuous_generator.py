import numpy as np
from scipy.special import expit
import pandas as pd
import matplotlib.pyplot as plt
import os


N= 100  # Set N
mu=0    # Mean for distributions
sigma=2 # Deviation for distributions
np.random.seed(123) # Set Seed 
train_rate =0.8 # Train rate for train-test split
current_dir = os.path.dirname(os.path.realpath(__file__))


# Useful functions
def sigmoid(z):
    return 1/(1+np.exp(-z))


# Build Endogenous & Exogenous variables variables
Z1=np.random.normal(mu, sigma,N) 
Z2=np.random.normal(mu, sigma,N) 
p= sigmoid(2*Z1-1*Z2)
T=[np.random.binomial(1,i,[1,1]) for i in p]
T=np.array(T)
T=np.asarray(T).reshape([N,])
UJ=np.random.normal(mu, sigma,N)
UP=np.random.normal(mu, sigma,N)
UD=np.random.normal(mu, sigma,N)
U=np.random.normal(mu, sigma,N)
J=3*T-2*Z1+3*Z2+UJ
P=J-2*T+2*Z2+0.5*Z1+UP
D=2*J+P-5*T+Z1+Z2+UD
Y= 3*T+2*Z1+2*Z2+0.5*J+D+U

# Generating the factual df
df=pd.DataFrame(data=[T, Z1, Z2, J, P, D, Y]).T
df.columns=["T", "Z1", "Z2", "J", "P", "D", "Y"]


## Building counterfactuals
# 1. Abduction: Find "unobserved" values using observations (here we have ours)
df['U'] = U
# 2. Action: Modify the SCM (invert the treatment)
T_CF= 1-T
J=3*T_CF-2*Z1+3*Z2+UJ
P=J-2*T_CF+2*Z2+0.5*Z1+UP
D=2*J+P-5*T_CF+Z1+Z2+UD
# 3. Prediction: use the value of U and the modified SCM to calculate the counterfactuals
Y_cf_Y=Y
Y_cf_Manual= 3*T_CF+2*Z1+2*Z2+0.5*J+D+U
Y_cf_Random= np.random.normal(mu, 100,N)
df['Y_cf_Y'] = Y_cf_Y
df['Y_cf_Manual'] = Y_cf_Manual
df['Y_cf_Random'] = Y_cf_Random

# Save plot
save_path = os.path.join(current_dir, 'T_vs_Y.png')
plt.scatter(df['T'],df['Y'] )
plt.title("T vs Y relation")
plt.xlabel("T")
plt.ylabel("Y")
plt.savefig(save_path)

# Train test split
X=np.array([Z1, Z2, J, P, D]).T
idx = np.random.permutation(N)
train_idx = idx[:int(train_rate * N)]
test_idx = idx[int(train_rate * N):]

# Train
train_x = np.array(X)[train_idx]
train_t = T[train_idx]
train_y = np.array(Y)[train_idx]
train_cf_y= np.array(Y_cf_Y)[train_idx]
train_cf_manual= np.array(Y_cf_Manual)[train_idx]
train_cf_random= np.array(Y_cf_Random)[train_idx]
# Test data
test_x = np.array(X)[test_idx]
test_t = T[test_idx]
test_y = np.array(Y)[test_idx]
test_cf_y= np.array(Y_cf_Y)[test_idx]
test_cf_manual= np.array(Y_cf_Manual)[test_idx]
test_cf_random= np.array(Y_cf_Random)[test_idx]

# Save data
names=["train_x", "train_t", "train_y", "train_cf_y","train_cf_manual","train_cf_random", "test_x", "test_t", "test_y", "test_cf_y", "test_cf_manual", "test_cf_random" ]
arr=[train_x, train_t, train_y, train_cf_y, train_cf_manual, train_cf_random, test_x, test_t, test_y, test_cf_y, test_cf_manual, test_cf_random ]
for i in range(len(arr)):
  aux = pd.DataFrame(arr[i])
  print(current_dir)
  save_path = os.path.join('../DATA/', names[i]+'_v0_continous.csv')
  aux.to_csv(save_path, index=False)
