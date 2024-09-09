import pandas as pd
from predictor import *
from sklearn import svm
from sklearn.metrics import accuracy_score

svm = svm.SVC()

svm.fit(X_train,Y_train)

y_pred2 = svm.predict(X_test)

print((accuracy_score(Y_test,y_pred2))*100)
