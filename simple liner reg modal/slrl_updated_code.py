import numpy as np 	#Array		
import matplotlib.pyplot as plt		
import pandas as pd	
# load the dataset
dataset=pd.read_csv(r"D:\Naresh IT foundation\Python project\simple liner reg modal\Salary_Data.csv")
# INDEPENDENT VARIABLE
x = dataset.iloc[:, :-1].values	
# DEPENDENT VARIABLE
y = dataset.iloc[:,-1].values  

# there is no missing value we can check by using(dataset.inull() all is false value so will not apply )
'''
from sklearn.impute import SimpleImputer # spyder4

imputer= SimpleImputer()
imputer.fit.transform(x[:,-1])

'''
# this is used for standerd split ration

# Split the Data ratio(80-20)

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(x_train,y_train)

y_pred=regressor.predict(x_test)



plt.scatter(x_test,y_test,color='red')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title('Salary vs Experience (test set')
plt.xlabel("year of experience ")
plt.ylabel("Salary")
plt.show()

m_slop=regressor.coef_
print(m_slop)

c_intercept= regressor.intercept_
print(c_intercept)

y_12=m_slop*12+c_intercept
print(y_12)

# compare prediction and actual salaries from the test set
comparison=pd.DataFrame({'Actual':y_test,'predicted':y_pred})

print(comparison)

bais= regressor.score(x_train, y_train)
print(bais)

varience=regressor.score(x_test,y_test)
print(varience)
# stat concept need to add into this code 

# means 
dataset.mean()
dataset['Salary'].mean()

# median
dataset.median()
dataset['Salary'].median()


# mode 
dataset.mode()
dataset['Salary'].mode()

# varience (𝑠**2=∑(𝑥𝑖−𝑥ˉ)2/𝑛−1)

dataset.var() # this will give varience of entire dataframne
dataset['Salary'].var()

# Standerd deviation( Root of S**2)

dataset.std() # this will give standard varience of entire dataframne
dataset['Salary'].std()

# cofficent of variance (cv)

# for calculating the cv we need to import the library from scipy.stats import varience

from scipy.stats import variation

variation(dataset.values)
variation(dataset['Salary'])


# Correlation 
dataset.corr()# this will Give correlation of entire data frame 
#this will give the correlation between 
dataset['Salary'].corr(dataset['YearsExperience'])

# Skewness 
dataset.skew()

dataset['Salary'].skew()

# santadard error 

dataset.sem()

# Zscore (Standarded normal deviation

import scipy.stats as stats
dataset.apply(stats.zscore)

stats.zscore(dataset['Salary'])

# degree of freedom 

a=dataset.shape[0] # this will give us no of rows 
b=dataset.shape[1] # this will give us no of coloumns 

degree_of_freedom=a-b
print(degree_of_freedom)

#SSR

y_mean=np.mean(y)
SSR=np.sum((y_pred-y_mean)**2)
print(SSR)

#SSE

y=y[0:6] # this is 6 here because we are taking testing point(y_test)
SSE=np.sum((y-y_pred)**2)
print(SSE)

#SST
mean_total=np.mean(dataset.values)
SST=np.sum((dataset.values-mean_total)**2)
print(SST)


# R2 Square 

r_square=1-(SSR/SST)
print(r_square)


# save the trained model to disk
import pickle

filename='Linear_regression_model.pkl'
with open(filename, 'wb') as file:
    pickle.dump(regressor, file)

print('Model has been picked and saved successfully!')

import os
print(os.getcwd())  # Print the current working directory to verify where the model is saved    
    