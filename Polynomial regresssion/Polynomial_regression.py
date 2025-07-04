import numpy as np 	#Array		
import matplotlib.pyplot as plt		
import pandas as pd	

dataset=pd.read_csv(r"D:\Naresh IT foundation\Python project\Polynomial regresssion\emp_sal.csv")

X = dataset.iloc[:, 1:2].values	
# DEPENDENT VARIABLE
y = dataset.iloc[:,2].values 

from sklearn.linear_model import LinearRegression
lin_reg=LinearRegression()
lin_reg.fit(X,y)

plt.scatter(X,y,color='red')
plt.plot(X,lin_reg.predict(X),color='Blue')
plt.title("Liner regression model(linear regression)")
plt.xlabel("position level")
plt.ylabel("salary")
plt.show()

lin_model_pred=lin_reg.predict([[6.5]])

lin_model_pred 


from sklearn.preprocessing import PolynomialFeatures

poly_reg=PolynomialFeatures(degree=5)
X_poly=poly_reg.fit_transform(X)

poly_reg.fit(X_poly,y)

lin_reg_2=LinearRegression()
lin_reg_2.fit(X_poly,y)


# ist model lin_reg_2 (liner model)

# 2nd model lin_reg(polynomial model)

plt.scatter(X,y,color='red')
plt.plot(X,lin_reg_2.predict(poly_reg.fit_transform(X)),color='Blue')
plt.title("Liner regression model(linear regression)")
plt.xlabel("position level")
plt.ylabel("salary")
plt.show()


poly_model_pred = lin_reg_2.predict(poly_reg.transform([[6.5]]))

poly_model_pred


