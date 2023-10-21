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
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt





#VERİ ÖN HAZIRLIK İŞLEMLERİ
# Wine veri setini yükleyin
wine = datasets.load_wine()


# Veriyi bir DataFrame'e dönüştürün
df = pd.DataFrame(wine.data, columns=wine.feature_names)


# Hedef sınıf etiketlerini DataFrame'e ekleyin
df['target'] = wine.target


# Veriyi inceleyin (veri başlıkları, ilk 5 satır ve eksik değer kontrolü)
print(df.head())
print(df.shape)
print(df.isna().sum())

# x, df DataFrame'i ile aynı veriyi temsil eder (özellikler)
x = df

# y, "target" sütununu ayırarak hedef sınıf etiketlerini temsil eder
y = x.pop("target")







#VERİNİN EĞİTİM VE TEST KÜMELERİNE BÖLÜNMESİ VE EĞİTİLMESİ
# Veriyi eğitim ve test kümelerine böl
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=55)


# K-Nearest Neighbors modelini oluştur
knn = KNeighborsClassifier(n_neighbors=5)  # K değeri 5 olarak belirlenmiş, isteğe bağlı olarak değiştirilebilir

# Modeli eğit
knn.fit(x_train, y_train)

# Modelin test kümesi üzerindeki performansını değerlendir
accuracy = knn.score(x_test, y_test)
print("Model Accuracy:", accuracy)










#SENSİTİVİTY TUNİNG GRAFİĞİ ÇİZİLMESİ
#Bu kod, K değerini (n_neighbors) değiştirerek
#modelin doğruluk skorlarını kaydeder ve bu skorları
#farklı K değerleri için gösteren bir grafik çizer. 
#Bu grafik, KNN modelinin hangi K değeriyle en iyi performansı
#gösterdiğini belirlemek için kullanılır. K değeri seçimi, 
#modelin doğruluğunu etkileyen önemli bir parametredir ve bu 
#grafikle optimize edilebilir..


# Sensitivity tuning grafiği için kullanılacak veriler 
k_range = range(1, 25) #k değerleri aralığı 
scores = [] #skor listesi


# K değerini değiştirerek modeli eğit ve doğruluk skorlarını kaydet
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(x_train, y_train)
    scores.append(knn.score(x_test, y_test))


# Sensitivity tuning grafiğini çiz
plt.figure(figsize=(8,6))
plt.plot(k_range, scores, marker='o', linestyle='-')
plt.xlabel("K count")
plt.ylabel("Model Accuracy")
plt.title("Sensitivity of Model to n_neighbors")
plt.grid()
plt.xticks(np.arange(1, 25))
plt.show()





#TRAİNİNG SPLİT ½ İLE ACCURACY GRAFİĞİ
#Bu kod, farklı eğitim verisi bölme oranlarına sahip 
#veri setleri üzerinde KNN modelinin doğruluğunu ölçer 
#ve sonuçları gösteren bir grafik oluşturur. Bu grafik, 
#eğitim verisi bölme oranının modelin performansını nasıl
#etkilediğini incelemenize yardımcı olacaktır.
test_sizes = [0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
accuracy_scores = []

for test_size in test_sizes:
    scores = []
    
    for i in range(1, 1000):
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=1 - test_size)
        knn.fit(x_train, y_train)
        scores.append(knn.score(x_test, y_test))
    accuracy_scores.append(np.mean(scores))

# Accuracy grafiğini çiz
plt.figure()
plt.plot(test_sizes, accuracy_scores, marker='o', linestyle='-')
plt.xlabel("Training split %")
plt.ylabel("Accuracy")
plt.title("Accuracy vs. Training Split Percentage")
plt.grid()
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




