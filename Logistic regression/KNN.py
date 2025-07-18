import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset=pd.read_csv(r"D:\Naresh IT foundation\Python project\Logistic regression\logit classification.csv")



X = dataset.iloc[:, [2,3]].values
# DEPENDENT VARIABLE
y = dataset.iloc[:,-1].values  

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)


from sklearn.preprocessing  import StandardScaler

sc=StandardScaler() 
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)

from sklearn.neighbors import KNeighborsClassifier

classifier=KNeighborsClassifier(n_neighbors=5) 
classifier.fit(X_train, y_train)

y_pred=classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test, y_pred)
print(cm)

from sklearn.metrics import accuracy_score
ac=accuracy_score(y_test, y_pred)
print(ac)

from sklearn.metrics import classification_report
cr=classification_report(y_test, y_pred)
print(cr)


bias=classifier.score(X_train,y_train)
print(bias)

variance=classifier.score(X_test,y_test)
print(variance)


## we need to pass future records to the 
dataset1=pd.read_csv(r"D:\Naresh IT foundation\Python project\Logistic regression\final1.csv")


d2=dataset1.copy()

dataset1=dataset1.iloc[:, [3, 4]].values

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
M=sc.fit_transform(dataset1)

y_pred1=pd.DataFrame()
d2['y_pred1']=classifier.predict(M)
d2.to_csv('final_csv')








