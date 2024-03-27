import pandas as pd
import random
import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_predict

#Reading Data
print ("Log: Reading Data from CSV file")
EnergyDataFile = pd.read_csv(r"CCPP_data.csv")

print ("Log: Splitting data by the columns, with Y/target being PE")
X_EnergyDataFile = EnergyDataFile.iloc[:,:-1]
Y_EnergyDataFile = EnergyDataFile.iloc[:, -1]

print ("Log: Features/Variables")
print (X_EnergyDataFile)
print ("Log: Observation or target to be predicted")
print (Y_EnergyDataFile)

X_train, X_test = train_test_split(X_EnergyDataFile, test_size= 0.15, shuffle=False)
Y_train, Y_test = train_test_split(Y_EnergyDataFile, test_size= 0.15, shuffle=False)

print ("Log: X/variables for training")
print (X_train)
print ("Log: Observation/target for training")
print (Y_train)

print ("Log: X/variables for testing")
print (X_test)
print ("Log: Observation/target for testing")
print (Y_test)

X_train.shape, X_test.shape, Y_train.shape, Y_test.shape
print("Log: X/variable training shape - ", X_train.shape)
print("Log: X/variable testing shape - ",X_test.shape)
print("Log: Y/observation/target training shape - ",Y_train.shape)
print("Log: Y/observation/target testing shape - ",Y_test.shape)

X_train = X_train.to_numpy()
X_test = X_test.to_numpy()
Y_train = Y_train.to_numpy()
Y_test = Y_test.to_numpy()

print ("Log: Linear Regression")

lr = LinearRegression()
lr.fit(X_train, Y_train)

target_predict = lr.predict(X_test)
scores = lr.score(X_test, Y_test)

print ("Log: Predicting values using linear regression")
print (target_predict)
print ("Log: Score of using linear progression")
print (scores)

#Prediction using test data
kf_predict = cross_val_predict(lr, X_test, Y_test)
print ("Cross Validation Prediction using test data")
print (kf_predict)

#Cross validation K-Fold
print ("Log: K-Fold Validation")

FoldCount = 12

while FoldCount > 0:
        print ("Log: %i Fold" %FoldCount)

        kf_score_r2 = cross_val_score(lr, X_train, Y_train, scoring = 'r2', cv=FoldCount)
        print("Log: R2 (Coefficient of Determination)")
        print (kf_score_r2)
        print ("Log: Average R2 Score: ", np.mean(kf_score_r2))

        kf_score_e_variance = cross_val_score(lr, X_train, Y_train, scoring = 'explained_variance', cv=FoldCount)
        print("Log: Explained Variance")
        print (kf_score_e_variance)
        print ("Log: Average EV Score: ", np.mean(kf_score_e_variance))

        kf_score_MSE = cross_val_score(lr, X_train, Y_train, scoring = 'neg_mean_squared_error', cv=FoldCount)
        print("Log: Mean Squared Error (MSE)")
        print (kf_score_MSE)
        print ("Log: Average MSE: ", np.mean(kf_score_MSE))

        kf_score_MAE = cross_val_score(lr, X_train, Y_train, scoring = 'neg_mean_absolute_error', cv=FoldCount)
        print("Log: Mean Absolute Error (MAE)")
        print (kf_score_MAE)
        print ("Log: Average MAE: ", np.mean(kf_score_MAE))

        kf_score_RMSE = cross_val_score(lr, X_train, Y_train, scoring = 'neg_root_mean_squared_error', cv=FoldCount)
        print("Log: Root Mean Squared Error (RMSE)")
        print (kf_score_RMSE)
        print ("Log: Average RMSE: ", np.mean(kf_score_RMSE))

        FoldCount = int(input("K fold count: "))

        if FoldCount <= 0:
                print ("End of K Fold Cross Validation")
                break