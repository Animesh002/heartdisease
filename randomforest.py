from datapreprocess2 import *
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

rf = RandomForestClassifier()
rf.fit(X_train,Y_train)

y_pred5 = rf.predict(X_test)

print((accuracy_score(Y_test,y_pred5))*100)