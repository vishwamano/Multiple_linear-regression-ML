# -*- coding: utf-8 -*-
"""
Created on Tue May 19 19:57:46 2020

@author: sarav
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


data_set = pd.read_csv("50_startups.csv")
x = data_set.iloc[:,:-1].values
y = data_set.iloc[:,4].values

data_set.isnull().sum()

from  sklearn.preprocessing import LabelEncoder , OneHotEncoder
le_x = LabelEncoder()
x[:,3] = le_x.fit_transform(x[:,3])

ohe_x = OneHotEncoder(categorical_features = [3])
x = ohe_x.fit_transform(x).toarray()

x = x[:,1:]

from sklearn.model_selection import train_test_split
x_train , x_test , y_train ,y_test = train_test_split(x,y,test_size=.30,random_state =0)

from sklearn.linear_model import LinearRegression
regress =LinearRegression()
regress.fit(x_train,y_train)


#pred
y_pred = regress.predict(x_test)

regress.score(x_train,y_train)
regress.score(x_test,y_test)

#starsmodel lib
#lest build the model using backwrd elimination

import statsmodels.api as sm
x = np.append(arr = np.ones(shape =[50,1] , dtype = int), values =x, axis = 1) #y= c + bx

#iteration 1
x_ov = x[:,[0,1,2,3,4,5]]
regress_ols = sm.OLS(endog = y ,exog = x_ov).fit()
regress_ols.summary()

x_ovc = x_ov[:,1:]
x_ov_train ,x_ov_test,y_ov_train ,y_ov_test =train_test_split(x_ovc,y,test_size =.30,random_state=0)
regress_ov1 = LinearRegression()
regress_ov1.fit(x_ov_train,y_ov_train)
regress_ov1.score(x_ov_train,y_ov_train)
regress_ov1.score(x_ov_test,y_ov_test)


#iteration 2regress_ov1.score(x_ov_train,y_ov_train)
regress_ov1.score(x_ov_test,y_ov_test)
x_ov = x[:,[0,1,3,4,5]]
regress_ols = sm.OLS(endog = y ,exog = x_ov).fit()
regress_ols.summary()

x_ovc = x_ov[:,1:]
x_ov_train ,x_ov_test,y_ov_train ,y_ov_test =train_test_split(x_ovc,y,test_size =.30,random_state=0)
regress_ov2 = LinearRegression()
regress_ov2.fit(x_ov_train,y_ov_train)
regress_ov2.score(x_ov_train,y_ov_train)
regress_ov2.score(x_ov_test,y_ov_test)

#iteration 3
x_ov = x[:,[0,3,4,5]]
regress_ols = sm.OLS(endog = y ,exog = x_ov).fit()
regress_ols.summary()

x_ovc = x_ov[:,1:]
x_ov_train ,x_ov_test,y_ov_train ,y_ov_test =train_test_split(x_ovc,y,test_size =.30,random_state=0)
regress_ov3 = LinearRegression()
regress_ov3.fit(x_ov_train,y_ov_train)
regress_ov3.score(x_ov_train,y_ov_train)
regress_ov3.score(x_ov_test,y_ov_test)

#iteration 4
x_ov = x[:,[0,3,5]]
regress_ols = sm.OLS(endog = y ,exog = x_ov).fit()
regress_ols.summary()

x_ovc = x_ov[:,1:]
x_ov_train ,x_ov_test,y_ov_train ,y_ov_test =train_test_split(x_ovc,y,test_size =.30,random_state=0)
regress_ov4 = LinearRegression()
regress_ov4.fit(x_ov_train,y_ov_train)
regress_ov4.score(x_ov_train,y_ov_train)
regress_ov4.score(x_ov_test,y_ov_test)

#iteration 5
x_ov = x[:,[0,3]]
regress_ols = sm.OLS(endog = y ,exog = x_ov).fit()
regress_ols.summary()

x_ovc = x_ov[:,1:]
x_ov_train ,x_ov_test,y_ov_train ,y_ov_test =train_test_split(x_ovc,y,test_size =.30,random_state=0)
regress_ov5 = LinearRegression()
regress_ov5.fit(x_ov_train,y_ov_train)
regress_ov5.score(x_ov_train,y_ov_train)
regress_ov5.score(x_ov_test,y_ov_test)

