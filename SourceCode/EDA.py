import pickle
import pandas as pd
import numpy as np
import re
from sklearn.ensemble import AdaBoostRegressor,RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import Normalizer
from sklearn.metrics import mean_squared_error,r2_score
import matplotlib.pyplot as plt
train=pd.read_csv('C:\Users\Krishna\DataScienceCompetetions\AVBlackFriday\\train_oSwQCTC\\train.csv')
for i in range(len(train['Age'].values)):
    flag=0
    for j in train['Age'][i]:
        if j=='-':
            ls=np.array([int(k) for k in str(train['Age'][i]).split('-')])
            # print ls
            train['Age'][i]=ls.mean()
            flag=1
    if flag==0:
        ls=train['Age'][i].split('+')
        train['Age'][i]=int(ls[0])
        # print ls


    # print train['Age'][i]
print train['Age'].values
# plt.hist(train['Age'].values)
plt.show()