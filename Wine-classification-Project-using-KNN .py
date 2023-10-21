# -*- coding: utf-8 -*-
"""
Created on Sat Oct 21 03:42:07 2023

@author: aile
"""

#wine classification with using knn 



#import library
import pandas as pd
import numpy as np
from sklearn import datasets #datasets modülü örnek datasetler içerir. wine datasetini bu modül ile aktaracağız.



#import data
wine = datasets.load_wine() #wine değişkenine sklearnde bulunan wine veri setini yükler.
df= pd.DataFrame(wine["data"],columns=wine["feature_names"]) #wine verileri dataframe'e dönüştürülür. 
df["target"]=wine["target"] #veri setinin hedef sınıf etiketlerini temsil eden target sütununu df ye ekler
print(df.head())
print(df.shape) #boyutu kontrol etmek için shape metodu ile satır sütun sayısını döndürürüz.
print(df.isna().sum()) #her sütunun altında kaç tane eksik değer olduğunu gösteren bir çıktı sağlar. Bu, veri setinin eksik veri içerip içermediğini değerlendirmek için kullanışlıdır


# SPLİT DATA
from sklearn.model_selection import train_test_split
x=df
y=x.pop("target")
print(x.head())
print(x.shape)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=55)
print(x_train.shape)
print(x_test.shape)

#TRAİN MODEL
from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=5)
knn.fit(x_train,y_train)
print(knn.score(x_test, y_test)) #here is basicaly checking accuracy of model basing on the test scores how it performs


#tunning sensitivity of model  to n_neighbors
k_range  = range(1,25)
scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(x_train,y_train)
    scores.append(knn.score(x_test,y_test))

#PLOT
import matplotlib.pyplot as plt
plt.figure()
plt.xlabel("K count")
plt.ylabel("Model Accuracy")
plt.scatter(k_range, scores)
plt.grid()
plt.xticks([0,5,10,15,20,30])
plt.show()





#PLOT
test_sizes=[0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1]
knn=KNeighborsClassifier(n_neighbors=5)
plt.figure()
for test_size in test_sizes:
    scores=[]
    
    for i in range(1,1000):
        x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1-test_size)
        knn.fit(x_train,y_train)
        scores.append(knn.score(x_test,y_test))
    plt.plot(test_size,np.mean(scores),"bo")

plt.xlabel("Training split %")
plt.ylabel("Accuracy")
plt.show()




#MAKE PREDİCTİONS
prediction = knn.predict(x_test)
print(prediction)
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test, prediction) 
print(cm)
plt.figure(figsize=(8,7))
import seaborn as sns
sns.heatmap(cm,annot=True)
plt.title("Confusion Matrix")
plt.ylabel("Truth")
plt.xlabel("Prediction")
plt.show()




