import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from datapreprocess2 import *

dt = DecisionTreeClassifier()
dt.fit(X_train,Y_train)

y_pred4 = dt.predict(X_test)

print((accuracy_score(Y_test,y_pred4))*100)