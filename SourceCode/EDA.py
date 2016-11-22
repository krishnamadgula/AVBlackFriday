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
# for i in range(len(train['Age'].values)):
#     flag=0
#     for j in train['Age'][i]:
#         if j=='-':
#             ls=np.array([int(k) for k in str(train['Age'][i]).split('-')])
#             # print ls
#             train['Age'][i]=ls.mean()
#             flag=1
#     if flag==0:
#         ls=train['Age'][i].split('+')
#         train['Age'][i]=int(ls[0])
#         # print ls
train.Age[train['Age']=='0-17']=17
train.Age[train['Age']=='18-25']=22
train.Age[train['Age']=='26-35']=31
train.Age[train['Age']=='36-45']=41
train.Age[train['Age']=='46-50']=48
train.Age[train['Age']=='51-55']=53
train.Age[train['Age']=='55+']=62

y1=train.Purchase[train['Age']==17].values
y2=train.Purchase[train['Age']==22].values
y3=train.Purchase[train['Age']==31].values
y4=train.Purchase[train['Age']==41].values
y5=train.Purchase[train['Age']==48].values
y6=train.Purchase[train['Age']==53].values
y7=train.Purchase[train['Age']==62].values
# train.Product_Category_1[train['Product_Category_1'].isna()]=0
train.Product_Category_2[pd.isnull(train['Product_Category_2'])]=0
print train.Product_Category_2.values
# y1=train.Purchase[train['Age']==17]
#     # print train['Age'][i]
# plt.boxplot(y2)
# plt.show()
# plt.boxplot(y3)
# plt.show()
# plt.boxplot(y4)
#
# plt.show()
# plt.boxplot(y5)
# plt.show()
# plt.boxplot(y6)
# plt.show()
# plt.boxplot(y7)
# plt.show()
# # print train['Age'].values
# # plt.hist(train['Age'].values)
# # plt.show()