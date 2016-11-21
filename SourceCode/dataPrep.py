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
# def encode_onehot(df, cols):
#     """
#
#     """
#     vec = DictVectorizer()
#
#     vec_data = pd.DataFrame(vec.fit_transform(df[cols].to_dict(outtype='records')).toarray())
#     vec_data.columns = vec.get_feature_names()
#     vec_data.index = df.index
#
#     df = df.drop(cols, axis=1)
#     df = df.join(vec_data)
#     return df



train=pd.read_csv('C:\Users\Krishna\DataScienceCompetetions\AVBlackFriday\\train_oSwQCTC\\train.csv')
test=pd.read_csv('C:\Users\Krishna\DataScienceCompetetions\AVBlackFriday\\test_HujdGe7\\test.csv')
train.Age[train['Age']=='0-17']=17
train.Age[train['Age']=='18-25']=22
train.Age[train['Age']=='26-35']=31
train.Age[train['Age']=='36-45']=41
train.Age[train['Age']=='46-50']=48
train.Age[train['Age']=='51-55']=53
train.Age[train['Age']=='55+']=62
train.Age[train['Age']==]=62

test.Age[test['Age']=='0-17']=17
test.Age[test['Age']=='18-25']=22
test.Age[test['Age']=='26-35']=31
test.Age[test['Age']=='36-45']=41
test.Age[test['Age']=='46-50']=48
test.Age[test['Age']=='51-55']=53
test.Age[test['Age']=='55+']=62

print train.values
list_y=train['Purchase'].values
train=train.drop('Purchase',axis=1)
# train=encode_onehot(df=train,cols=['Age','Gender','City_Category','Stay_In_Current_City_Years','Marital_Status'])
for i in ['Age','Gender','City_Category','Stay_In_Current_City_Years','Marital_Status','Product_Category_1','Product_Category_2','Product_Category_3']:
    oneHot=pd.get_dummies(train[i] )
    oneHot1=pd.get_dummies(test[i])
    # oneHot.columns=train.columns
    # oneHot.index=train.index
    train=train.drop(i,axis=1)
    test=test.drop(i,axis=1)
    train[oneHot.columns]=oneHot
    test[oneHot1.columns]=oneHot1
print train.columns
# for i in train.values:
#     print i
#
#


# test=pd.read_csv('C:\Users\Krishna\DataScienceCompetetions\AVBlackFriday\\test_HujdGe7\\test.csv')
# test=encode_onehot(df=train,cols=['Age','Gender','City_Category','Stay_In_Current_City_Years','Marital_Status','Product_Category_1','Product_Category_2','Product_Category_3'])

# for i in ['Age','Gender','City_Category','Stay_In_Current_City_Years','Marital_Status','Product_Category_1','Product_Category_2','Product_Category_3']:
# train.dropna(axis=0,how='any')
count=0
# train=train[train.Product_Category_1.notnull()]
print len(train.values)
# for i in train['City_Category'].values:
#     if i==np.nan or i==None or not np.isfinite(i) :
#         count=count+1
# print count
list_x=train.values
list_x_test=test.values
columns=train.columns
index=train.index
for i in list_x:
    # print i[1]
    list_i=i[1].split('P')
    i[1]= int(list_i[1])
for i in list_x_test:
    # print i[1]
    list_i=i[1].split('P')
    i[1]= int(list_i[1])
# train =(train-train.mean())/(train.std())
# train_x=pd.DataFrame(list_x,index=index,columns=columns)

#
# vectorizer=DictVectorizer()
# vectorizedTrainData=pd.DataFrame(vectorizer.fit_transform(train['Age','Gender','City_Category','Stay_In_Current_City_Years','Marital_Status','Product_Category_1','Product_Category_2','Product_Category_3'].to_dict(outtype='records')))
# vectorizedTrainData.columns=vectorizer.get_feature_names()
# vectorizedTrainData.index=train.index
# train=train.drop(['Age','Gender','City_Category','Stay_In_Current_City_Years','Marital_Status','Product_Category_1','Product_Category_2','Product_Category_3'])
# train=train.join(vectorizedTrainData)
# oneHot=pd.get_dummies(train['Age'] )
# train=encode_onehot(df=train,cols=['Age','Gender','City_Category','Stay_In_Current_City_Years','Marital_Status','Product_Category_1','Product_Category_2','Product_Category_3'])
# for i in train.values:
#     print i
# train=train.drop('Age',axis=1)
# oneHot.columns=
# train=train.join(oneHot)
# print 'hi'
# print oneHot.values
print len(list_y)
# for i in range(len(list_x)):
#     for j in list_x[i]:
#         if j==np.nan or j==None:
#             list_x.remove(list_x[i])
#             list_y.remove(list_y[i])
print len(list_x),len(list_y)
normalizer=Normalizer()
list_x_test=normalizer.fit_transform(list_x_test)
list_x=normalizer.fit_transform(list_x)
file=open('C:\Users\Krishna\DataScienceCompetetions\AVBlackFriday\\trainFile','wb')
X_Train,X_Test,Y_Train,Y_Test=train_test_split(list_x,list_y,test_size=0.3)
X_valid=X_Test
Y_valid=Y_Test
pickle.dump([X_Train,X_valid,Y_Train,Y_valid,list_x_test],file)
file.close()
