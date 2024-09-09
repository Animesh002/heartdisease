# import pandas as pd
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.impute import SimpleImputer
# from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
# from sklearn.model_selection import train_test_split
# #
# # from ucimlrepo import fetch_ucirepo
# #
# # # fetch dataset
# # heart_disease = fetch_ucirepo(id=45)
# #
# # # data (as pandas dataframes)
# # X = heart_disease.data.features
# # Y = heart_disease.data.targets
# #
# # # metadata
# # print(heart_disease.metadata)
# #
# # # variable information
# # print(heart_disease.variables)
# #
# data = pd.read_csv('dataset.csv')
# data = data.drop_duplicates()
# # data_dup = data.duplicated().any()
# # print(data_dup)
#
# X = data.drop('target',axis = 1)
# Y = data['target']
#
# # X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=42)
# #
# #
#
# # import pandas as pd
# # from sklearn.model_selection import train_test_split
# # from sklearn.ensemble import RandomForestClassifier
# # from sklearn.impute import SimpleImputer
# # from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# #
# # # Assuming you have fetched the dataset using fetch_ucirepo
# # from ucimlrepo import fetch_ucirepo
# #
# # heart_disease = fetch_ucirepo(id=45)
# #
# # # data (as pandas dataframes)
# # X = heart_disease.data.features
# # Y = heart_disease.data.targets
# #
# # # Handling missing values using SimpleImputer
# imputer = SimpleImputer(strategy='mean')  # You can choose other strategies as well
# X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
# #
# # # Split the data into training and testing sets
# X_train, X_test, Y_train, Y_test = train_test_split(X_imputed, Y, test_size=0.2, random_state=42)
# #
# # # Initialize and train the Random Forest model
# random_forest_model = RandomForestClassifier(n_estimators=100, random_state=42)
# random_forest_model.fit(X_train, Y_train)
#
# # Make predictions on the test set
# y_pred = random_forest_model.predict(X_test)
#
# # Evaluate the model
# accuracy = accuracy_score(Y_test, y_pred)
# conf_matrix = confusion_matrix(Y_test, y_pred)
# classification_rep = classification_report(Y_test, y_pred)
#
# print(f"Test Accuracy: {accuracy * 100:.2f}%")
# print("Confusion Matrix:\n", conf_matrix)
# print("Classification Report:\n", classification_rep)

# import pandas as pd
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.impute import SimpleImputer
# from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
# from sklearn.model_selection import train_test_split
#
# # Load your dataset
# data = pd.read_csv('dataset.csv')
#
# # Drop duplicates
# data = data.drop_duplicates()
#
# #invalid data removal
# data = data.dropna()
#
#
# # Separate features and target variable
# X = data.drop('target', axis=1)
# Y = data['target']
#
# # Handling missing values using SimpleImputer
# imputer = SimpleImputer(strategy='mean')
# X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
#
# # Split the data into training and testing sets
# X_train, X_test, Y_train, Y_test = train_test_split(X_imputed, Y, test_size=0.2, random_state=42)
#
# # Initialize and train the Random Forest model
# random_forest_model = RandomForestClassifier(n_estimators=100, random_state=42)
# random_forest_model.fit(X_train, Y_train)
#
# # Make predictions on the test set
# y_pred = random_forest_model.predict(X_test)
#
# # Evaluate the model
# accuracy = accuracy_score(Y_test, y_pred)
# conf_matrix = confusion_matrix(Y_test, y_pred)
# classification_rep = classification_report(Y_test, y_pred)
#
# print(f"Test Accuracy: {accuracy * 100:.2f}%")
# print("Confusion Matrix:\n", conf_matrix)
# print("Classification Report:\n", classification_rep)

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import numpy as np

# Load your dataset
data = pd.read_csv('disease.csv')

# Drop duplicates
data = data.drop_duplicates()

# Invalid data removal (remove rows with missing values)
data = data.dropna()

# Check if there are samples in the dataset
if data.shape[0] == 0:
    print("Error: No samples in the dataset.")
    exit()

# Check for outliers and remove if present
target_column = 'target' if 'target' in data.columns else 'Disease'
if target_column in data.columns:
    features = data.drop(target_column, axis=1)
    z_scores = np.abs((features - features.mean()) / features.std())
    outliers_mask = (z_scores >= 3).any(axis=1)
    data_outliers = data[outliers_mask]
    data_no_outliers = data[~outliers_mask]
else:
    data_no_outliers = data
    data_outliers = pd.DataFrame()

# Print the outliers
if not data_outliers.empty:
    print("Outliers:")
    print(data_outliers)
else:
    print("No outliers found.")

# Dynamically identify categorical columns and perform encoding
label_encoder = LabelEncoder()
categorical_columns = data_no_outliers.select_dtypes(include=['object']).columns
for col in categorical_columns:
    data_no_outliers[col] = label_encoder.fit_transform(data_no_outliers[col])

# Check if there are samples after outlier removal
if data_no_outliers.shape[0] == 0:
    print("Error: No samples after outlier removal.")
    exit()

# Separate features and target variable
X = data_no_outliers.drop(target_column, axis=1)
Y = data_no_outliers[target_column]

# Handling missing values using SimpleImputer
imputer = SimpleImputer(strategy='mean')
X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X_imputed, Y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize and train the Random Forest model
random_forest_model = RandomForestClassifier(n_estimators=100, random_state=42)
random_forest_model.fit(X_train, Y_train)

# Make predictions on the test set
y_pred = random_forest_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(Y_test, y_pred)
conf_matrix = confusion_matrix(Y_test, y_pred)
classification_rep = classification_report(Y_test, y_pred)

print(f"Test Accuracy: {accuracy * 100:.2f}%")
print("Confusion Matrix:\n", conf_matrix)
print("Classification Report:\n", classification_rep)

