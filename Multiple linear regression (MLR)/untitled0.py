import numpy as np 	#Array		
import matplotlib.pyplot as plt		
import pandas as pd	
# load the dataset
dataset=pd.read_csv(r"D:\Naresh IT foundation\Python project\Multiple linear regression (MLR)\Investment.csv")
# INDEPENDENT VARIABLE
X = dataset.iloc[:, :-1].values	
# DEPENDENT VARIABLE
y = dataset.iloc[:,4].values  

X=pd.get_dummies(X,dtype=int)


# SPLIT THE DATA 
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)


# multiple liner regression 

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(X_train,y_train)

y_pred=regressor.predict(X_test)

