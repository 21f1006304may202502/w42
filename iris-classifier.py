import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from pandas.plotting import parallel_coordinates
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn import metrics
import joblib

data = pd.read_csv("iris.csv")
data.head(5)

train,test = train_test_split(data,test_size=0.4,stratify=data['species'],random_state=42)
X_train = train[['sepal_length','sepal_width','petal_length','petal_width']]
y_train = train.species
X_test = test[['sepal_length','sepal_width','petal_length','petal_width']]
y_test = test.species

mod_dt = DecisionTreeClassifier(max_depth=3,random_state=1)
mod_dt.fit(X_train,y_train)
prediction = mod_dt.predict(X_test)

accuracy = metrics.accuracy_score(y_test,prediction) 
precision = metrics.precision_score(y_test,prediction,average='macro') 
recall = metrics.recall_score(y_test,prediction,average='macro') 
f1 = metrics.f1_score(y_test,prediction,average='macro') 

print("Accuracy : ",accuracy)
print("Precision : ",precision)
print("Recall : ",recall)
print("F1 : ",f1)

metrics_df = pd.DataFrame({
    'Metrics' : ['Accuracy','Precision','Recall','F1'],
    'Score' : [accuracy,precision,recall,f1]
    })
metrics_df.to_csv('metrics.csv',index=False)

joblib.dump(mod_dt, "model.joblib")
