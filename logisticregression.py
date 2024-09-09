import pandas as pd
from predictor import *
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Logistic Regression

log = LogisticRegression()
log.fit(X_train,Y_train)
# print(log)
y_pred1 = log.predict(X_test)

print((accuracy_score(Y_test,y_pred1))*100)