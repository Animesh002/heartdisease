import pandas as pd
from predictor import *
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix

knn = KNeighborsClassifier(n_neighbors=2)

knn.fit(X_train,Y_train)

y_pred3 = knn.predict(X_test)
print((accuracy_score(Y_test,y_pred3))*100)
conf_matrix = confusion_matrix(Y_test,y_pred3)
classification_rep = (classification_report(Y_test,y_pred3))
#
# print(conf_matrix)
# print(classification_rep)



#
#
# best_k = 0
# best_accuracy = 0
#
# # Iterate over different values of k
# for k in range(1, 40):
#     knn = KNeighborsClassifier(n_neighbors=k)
#     knn.fit(X_train, Y_train)
#     y_pred = knn.predict(X_test)
#     current_accuracy = accuracy_score(Y_test, y_pred)
#
#     # Check if the current k gives a higher accuracy
#     if current_accuracy > best_accuracy:
#         best_k = k
#         best_accuracy = current_accuracy
#
#
# print(best_accuracy)