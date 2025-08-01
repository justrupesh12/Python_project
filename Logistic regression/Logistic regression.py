import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset=pd.read_csv(r"C:\Users\RuKumar\OneDrive - DXC Production\python\python projects\logit classification.csv")

X=dataset.iloc[:,[2,3]].values
y=dataset.iloc[:,-1].values

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=51)


# this will Santdarlaized between _3 to 3
from sklearn.preprocessing import StandardScaler

SC=StandardScaler()
X_train=SC.fit_transform(X_train)
X_test=SC.transform(X_test)


from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(X_train,y_train)

#model-classifier
#algorithm-LogisticRegression

y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
print(cm)

from sklearn.metrics import accuracy_score
ac = accuracy_score(y_test,y_pred)
print(ac)

# to get classification report

from sklearn.metrics import classification_report
cr = classification_report(y_test,y_pred)
print(cr)
