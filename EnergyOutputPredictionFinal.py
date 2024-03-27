import pandas as pd
import random
import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression


#Reading Data
print ("Reading Data from CSV file")
EnergyDataFile = pd.read_csv(r"CCPP_data.csv")

#X = EnergyDataFile

#kf = KFold(n_splits=12)
#for train, test in kf.split(X):
#    print("%s %s" % (train, test))

#for i in range(10):
#    print("Current Index of the file is: %s" %i)
#    print(len(EnergyDataFile))
#    print(EnergyDataFile.loc[i])

X_EnergyDataFile = EnergyDataFile.iloc[:,:-1]
Y_EnergyDataFile = EnergyDataFile.iloc[:, -1]

X_train, X_test = train_test_split(X_EnergyDataFile, test_size= 0.15, shuffle=False)
Y_train, Y_test = train_test_split(Y_EnergyDataFile, test_size= 0.15, shuffle=False)

X_train.shape, X_test.shape, Y_train.shape, Y_test.shape
print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)
#print ("Training: ")
#print (Y_train)
#print ("Testing: ")
#print (Y_test)

#print ("logistic regression")
#lr = LogisticRegression()
#lr.fit(X_train, Y_train)
#lr.score(X_test, Y_test)

X_train = X_train.to_numpy()
X_test = X_test.to_numpy()
Y_train = Y_train.to_numpy()
Y_test = Y_test.to_numpy()

lr = LinearRegression()
lr.fit(X_train, Y_train)

target_predict = lr.predict(X_test)
scores = lr.score(X_test, Y_test)

print (target_predict)
print (scores)