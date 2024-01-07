#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 23 16:29:59 2022

@author: Parth patel
"""

import pandas as pd
import os
import numpy as np
 
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score


#Load & check the data:
path = "/Users/sambhu/Desktop/sem-2/Supervised learning 247/Assignment /Assignment-3"
filename = 'student-por.csv'
fullpath = os.path.join(path,filename)
data_parth = pd.read_csv(fullpath,sep=';')

#Carryout some initial investigations:
print(data_parth.dtypes)
print(data_parth.isnull().sum())
print(data_parth.describe())
print(data_parth.median())
print(data_parth.columns.values)
print(data_parth.head(5))
print(data_parth.info())
print(data_parth['address'].unique())
print(data_parth['famsize'].unique())
print(data_parth['school'].unique())

# Create a new target variable
data_parth['sum']= data_parth['G1']+ data_parth['G2'] + data_parth['G3']

data_parth['pass_parth']= np.where(data_parth['sum']>=35,1,0)

data_parth=data_parth.drop(columns=['G1','G2','G3','sum'])

#Separate the features from the target variable
colnames=data_parth.columns.values.tolist()
predictors=colnames[:30]
target=colnames[30]

feature_parth= data_parth[predictors]
target_parth = data_parth[target]

#Create two lists one to save the names of your numeric fields and on to save the names of your categorical fields
numberic_data = feature_parth.select_dtypes(include='int64')
numberic_features_parth= numberic_data.columns.values.tolist()

cat_data= feature_parth.select_dtypes(include='object')
cat_features_parth = cat_data.columns.values.tolist()

# Prepare a column transformer to handle all the categorical variables
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

transformer_parth =  ColumnTransformer(transformers =[ ("cat",OneHotEncoder(),cat_features_parth)])
 
#Prepare a classifier decision tree model 
from sklearn.tree import DecisionTreeClassifier   
clf_parth = DecisionTreeClassifier(criterion="entropy",max_depth = 5)

from sklearn.pipeline import Pipeline
pipeline_first_parth = Pipeline(
    steps=[("preprocessor", transformer_parth), ("m", clf_parth)]
    )

from sklearn.model_selection import train_test_split
X_train_parth,X_test_parth, y_train_parth, y_test_parth = train_test_split(feature_parth,target_parth, test_size = 0.2, random_state=43)

#Build Classification Models
pipeline_first_parth.fit(X_train_parth,y_train_parth)


print(pipeline_first_parth.score(X_test_parth,y_test_parth))

crossvalidation = KFold(n_splits=10, shuffle=True, random_state=43)
score = np.mean(cross_val_score(pipeline_first_parth, X_train_parth, y_train_parth, scoring='accuracy', cv=crossvalidation ))
print(score)

scores = []
for i in range(1,10):
    clf_parth1 = DecisionTreeClassifier(criterion="entropy",max_depth = i)
    pipeline_first_parth1 = Pipeline(
    steps=[("preprocessor", transformer_parth), ("classifier", clf_parth1)]
    )
    pipeline_first_parth1.fit(X_train_parth,y_train_parth)
    
    score = np.mean(cross_val_score(pipeline_first_parth, X_train_parth, y_train_parth, scoring='accuracy', cv=crossvalidation ))
    print(score)
    scores.append(score)

scores=pd.DataFrame(scores)
print("mean of 10 scores:",scores.mean())

#-----------------------------
#Visualize the tree using Graphviz.
import graphviz
from sklearn import tree

fn = transformer_parth.get_feature_names()

#clf_parth.fit(fn,y_train_parth)

dot_data = tree.export_graphviz(clf_parth, out_file=None, 
                                feature_names=fn,  
                                class_names=['0','1'],
                                filled=True)

graph = graphviz.Source(dot_data, format="png") 
graph
graph.render("/Users/sambhu/Desktop/sem-2/Supervised learning 247/Assignment /Assignment-3/decision_tree")

#-----------------------------



score = np.mean(cross_val_score(pipeline_first_parth, X_train_parth, y_train_parth, scoring='accuracy', cv=crossvalidation ))
print(score)


score = np.mean(cross_val_score(pipeline_first_parth, X_test_parth, y_test_parth, scoring='accuracy', cv=crossvalidation ))
print(score)


y_pred= pipeline_first_parth.predict(X_test_parth)

from sklearn import metrics 
labels = y_test_parth.unique()

print("Accuracy:",metrics.accuracy_score(y_test_parth, y_pred))

from sklearn.metrics import confusion_matrix
CM=confusion_matrix(y_test_parth, y_pred, labels)

print("Confusion matrix :"  , CM)

from sklearn.metrics import recall_score
print("recall:",recall_score(y_test_parth, y_pred))

from sklearn.metrics import precision_score
print("precision:",precision_score(y_test_parth, y_pred))


#Fine tune the model
from sklearn.model_selection import RandomizedSearchCV
parameters=[{'m__min_samples_split' : range(10,300,20),
            'm__max_depth': range(1,30,2),
            'm__min_samples_leaf':range(1,15,3)}]



RGS = RandomizedSearchCV(estimator= pipeline_first_parth,
                         scoring='accuracy',
                         param_distributions=parameters,
                         cv=5,
                         n_iter = 7,
                         refit = True,
                         verbose = 3)

#pipeline_first_parth.get_params().keys()
RGS.fit(X_train_parth, y_train_parth)

print("best parameters:",RGS.best_params_)

print("Score",RGS.best_score_)

print("best estimator:" , RGS.best_estimator_)

y_pred1 = RGS.predict(X_test_parth)

print("recall:",recall_score(y_test_parth, y_pred1))

print("precision:",precision_score(y_test_parth, y_pred1))

print("Accuracy:",metrics.accuracy_score(y_test_parth, y_pred1))

best_model = RGS.best_estimator_
import joblib
joblib.dump(best_model, 
            '/Users/sambhu/Desktop/sem-2/Supervised learning 247/Assignment /Assignment-3/best_model.pkl')


joblib.dump(pipeline_first_parth, 
            '/Users/sambhu/Desktop/sem-2/Supervised learning 247/Assignment /Assignment-3/pipe_svm_parth.pkl')








