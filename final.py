import pandas as pd
from sklearn.metrics import accuracy_score
from datapreprocess2 import *
from predictor import *
from decisiontree import *
from gradientboostclassifier import *
from KNN import *
from logisticregression import *
from randomforest import *
from SVM import *
import seaborn as sns


final_data = pd.DataFrame({'Models':['LR','SVM','KNN','DT','RF','GBC'],
                           'ACC':[accuracy_score(Y_test,y_pred1),
                                  accuracy_score(Y_test,y_pred2),
                                  accuracy_score(Y_test,y_pred3),
                                  accuracy_score(Y_test,y_pred4),
                                  accuracy_score(Y_test,y_pred5),
                                  accuracy_score(Y_test,y_pred6)]})

# print(final_data)

x = sns.barplot(final_data['Models'],final_data['ACC'])
print(x)