import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import pandas as pd
import seaborn as sns

from sklearn import preprocessing
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.metrics import r2_score

data=pd.read_csv(r"D:\Naresh IT foundation\Python project\lesso-ridge-Regulization\car-mpg.csv")
data = data.drop(['car_name'], axis=1)
data['origin'] = data['origin'].replace({1: 'america', 2: 'europe', 3: 'asia'})
data = pd.get_dummies(data, columns=['origin'], dtype=int)
data =data.replace('?',np.NAN)
data.head()
#data=data.apply(lambda x: x.fillna(x.median()),axis=0)


data = data.apply(pd.to_numeric, errors='ignore')

# Fill missing values with median only for numeric columns
numeric_cols = data.select_dtypes(include=[np.number]).columns
data[numeric_cols] = data[numeric_cols].apply(lambda x: x.fillna(x.median()))   

x=data.drop(['mpg'],axis=1) # independent variable

y=data[['mpg']] # dependent variable 

x_s=preprocessing.scale(x)
x_s=pd.DataFrame(x_s,columns=x.columns)  #converting scale data into dataframe

y_s=preprocessing.scale(y)
y_s=pd.DataFrame(y_s,columns=y.columns)   #ideally train ,teat data should be a columns 


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.20, random_state = 0)                    

### 2.a Simple Linear Model
#Fit simple linear model and find coefficients
regression_model = LinearRegression()
regression_model.fit(x_train, y_train)

for idx, col_name in enumerate(x_train.columns):
    print('The coefficient for {} is {}'.format(col_name, regression_model.coef_[0][idx]))
    
intercept = regression_model.intercept_[0]
print('The intercept is {}'.format(intercept))


#Regularized Ridge Regression
#alpha factor here is lambda (penalty term) which helps to reduce the magnitude of coeff

ridge_model = Ridge(alpha = 0.3)
ridge_model.fit(x_train, y_train)

print('Ridge model coef: {}'.format(ridge_model.coef_))
#As the data has 10 columns hence 10 coefficients appear here    