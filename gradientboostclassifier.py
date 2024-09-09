from datapreprocess2 import *
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
import pandas as pd

gb = GradientBoostingClassifier()
gb.fit(X_train,Y_train)

y_pred6 = gb.predict(X_test)

print((accuracy_score(Y_test,y_pred6))*100)