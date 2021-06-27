# -*- coding: utf-8 -*-
"""
Created on Thu Jun 17 17:00:39 2021

@author: lenovo
"""

from sklearn.model_selection import train_test_split
#import matplotlib.pyplot as plt                            46
import scipy.io
from sklearn.svm import SVC
import numpy as np
from scipy import signal
from sklearn.preprocessing import StandardScaler
import random
import pickle
import time
a = time.time()

def gurultu(x):
    for k in range(0, int(x.size / len(x))):
        for i in range(0, len(x)): 
            x[i,k] += random.uniform(-100,100)
    
    return x

mat =scipy.io.loadmat('data.mat')
#print(mat)
x  = mat["LeftBackward1"]
x1 = mat["LeftBackward2"]
x2 = mat["LeftBackward3"]

x3 = mat["LeftForward1"]
x4 = mat["LeftForward2"]
x5 = mat["LeftForward3"]

x6 = mat["RightBackward1"]
x7 = mat["RightBackward2"]
x8 = mat["RightBackward3"]

x9 = mat["RightForward1"]
x10 = mat["RightForward2"]
x11 = mat["RightForward3"]


liste = []
orijinalVeri = [x,x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11]
# EEG cihazının çalışma frekansı
fs=500

# np.array([sho, sho1, sho2, sho3, sho4, sho5, sho6, sho7, sho8, sho9, sho10, sho11])

gurultuluData = []
for i in orijinalVeri:
    for j in range(0,20):
        gurultuluData.append(gurultu(i))
# alınan gürültülü veriye sınıflarını ekle ve eğitime at

gurultuluDataLabels = []
cnt = 0
for i in range(0,240):
    gurultuluDataLabels.append(cnt)
    if (i % (3*20) == 0) and (i != 0) :
        cnt +=1

"""
print(gurultuluDataLabels)
si sg sai sag
3   3   3   3
20  20  20  20
60  60  60  60
"""
def filtre(x):
    for k in range(0, int(x.size / len(x))):
        for i in range(0, len(x)):
            if ((i % 500) >= 0 and (i % 500) <= 48): # Beyinde motor verileri belirli promplarda 0-48 Hz aralığında ölçülür.
                liste.append(x[i, k])

    #sonuc = liste
    sonuc1 = np.array(liste)
    # sonuc2=sonuc.astype(np.int32)
    
    sonuc3 = sonuc1.reshape((-1, 1)).transpose().reshape(int(x.size / len(x)), int(sonuc1.size / (int(x.size / len(x)))))
    f, Pxx = signal.welch(sonuc3, fs)
    ho = []
    
    # bi ara da ortalamasına almadan dene!!!!
    ho = np.mean(Pxx, axis=0)
    
    # plt.figure(2)
    # plt.plot(ho)

    return ho


"""
sho = filtre(x)
sho1 = filtre(x1)
sho2 = filtre(x2)

sho3 = filtre(x3)
sho4 = filtre(x4)
sho5 = filtre(x5)

sho6 = filtre(x6)
sho7 = filtre(x7)
sho8 = filtre(x8)

sho9 = filtre(x9)
sho10 = filtre(x10)
sho11 = filtre(x11)
# sonucc=filtre(x1)
"""
def main():
    """filteredData = []
    for data in gurultuluData:
        filteredData.append(filtre(data))"""
    """
    # Veriler bir listede toplandı 
    features = np.array(filteredData)
    #np.array([sho, sho1, sho2, sho3, sho4, sho5, sho6, sho7, sho8, sho9, sho10, sho11])

    #features = np.array(features)
    #clas=[1,1,1,1,1,1,0,0,0,0,0,0]
    #clas = [[0, 0], [0, 0], [0, 0], [0, 1], [0, 1], [0, 1], [1, 0], [1, 0], [1, 0], [1, 1], [1, 1], [1, 1]]

    # Verilerin sınıfların değerleri eklendi
    #label = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3])
    label = np.array(gurultuluDataLabels)
    # özniteliği çıkartılmış veriler ve sınıfları -> veri-sınıfı olarak yan yana eklendi 
    #featuress = np.c_[features, label]
    #print(featuress)
    # for i in range(0,len(features)):
    #    features.append()


    # veriler X'e atandı
    X = features

    # sınıflar Y'ye atandı
    Y = label


    #X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.5)

    # Makinenin eğitilmesi için nesne oluşturuldu
    sc = StandardScaler()

    #min_max_scaler= preprocessing.MinMaxScaler() # NORMALİZASYON
    #X_scale = min_max_scaler.fit_transform(X)




    # verileri Eğitim ve test olarak test_size'a göre ayarlandı
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.7)


    #min_max_scaler= preprocessing.MinMaxScaler() # NORMALİZASYON
    #X_scale = min_max_scaler.fit_transform(X)

    #X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)
    #X_val, X_test, Y_val, Y_test = train_test_split(X_val_and_test, Y_val_and_test, test_size=0.5)

    # Makineyi eğitiyor
    X_train = sc.fit_transform(X_train)

    # Test verilerini hazırlıyor
    X_test = sc.transform(X_test)


    #kernel{‘linear = lineer’, ‘poly = polimetrik’, ‘rbf’, ‘sigmoid = sigmoid’, ‘precomputed’}, default=’rbf’     , probability=True 168-129
    #SVC(C=1.0, kernel='linear', degree=3, gamma='auto',probability=True)
    #svc = SVC(kernel = '‘linear', probability=True)
    svc = SVC(C=1.0, kernel='linear')


    svc = OneVsRestClassifier(SVC()).fit(X_train, Y_train)

    svc.fit(X_train, Y_train)"""
    svc = SVC(C=1.0, kernel='linear')

    X_train = np.load('x_train.npy')

    Y_train = np.load('y_train.npy')
    X_test = np.load('x_test.npy')

    svc.fit(X_train, Y_train)

    # Oluşturulan modeli kaydediyor
    filename = 'svmEegModel.sav'
    pickle.dump(svc, open(filename, 'wb'))
    Y_test = np.load('y_test.npy')


    loaded_model = pickle.load(open(filename, 'rb'))

    # test verilerini makineye soruyor
    #for  i in range(10):
    # Kayıtlı modele soru sormak için alttakini aktif et
    i = random.randint(0,100)
    Y_val = loaded_model.predict(X_test[i].reshape((-1, 129)))

    # Program çalışırken eğitilen modele soru somak için alttkini aktif et
    #Y_val =y loaded_model.predict(X_test[i].reshape((-1, 129)))
    np.save('y_test',Y_test)
    print("Tahmin : ",Y_val," Gerçek Değer : ",Y_test[i])
    print("----")
    print(Y_test[i])
    print("----")

    return(Y_test[i])
    
    """
    # verilerin doğruluk tablosunu çıkartır
    cm = confusion_matrix(Y_test,Y_val)
    print(cm)
    """

    """
    sonuçta köşegenler doğru, diğerleri yanlış bildikleri
    1 0 0
    0 1 0
    0 0 1

   0 """

