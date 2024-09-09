import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn import svm
from sklearn.metrics import confusion_matrix
import tensorflow as tf

# np.random.seed(42)
# tf.random.set_seed(42)

data = pd.read_csv('heart.csv')

# data.isnull().sum()
# print(x)

# data_dup = data.duplicated().any()
# print(data_dup)

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

# Encode categorical variables
label_encoder = LabelEncoder()
categorical_columns = data_no_outliers.select_dtypes(include=['object']).columns
for col in categorical_columns:
    data_no_outliers[col] = label_encoder.fit_transform(data_no_outliers[col])

# data_dup = data.duplicated().any()
# print(data_dup)

#data preprocessing
# cat_val = []
# num_val=[]
#
# for column in data.columns:
#     if data[column].nunique()<=10:
#         cat_val.append(column)
#     else:
#         num_val.append(column)

# print(cat_val)
# print(num_val)

#encoding the cateogrical data

#first removing dummy variable trap
#
# cat_val.remove('sex')
# cat_val.remove('target')
# data = pd.get_dummies(data,columns=cat_val,drop_first=True)

# print(data.head())


# print(data.head())

#Split the data
X = data_no_outliers.drop('target',axis=1)
Y = data_no_outliers['target']

# Handle missing values using SimpleImputer
imputer = SimpleImputer(strategy='mean')
X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=42)
# print(X_test)

#Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)



# # Standardize the features
# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)

#Logistic Regression
log = LogisticRegression()
log.fit(X_train,Y_train)
# print(log)
y_pred1 = log.predict(X_test)
print("Logistic Regression:")
print((accuracy_score(Y_test,y_pred1))*100)
conf_matrix1 = confusion_matrix(Y_test,y_pred1)
classification_rep1 = (classification_report(Y_test,y_pred1))
print("Confusion Matrix:\n", conf_matrix1)
print("Classification Report:\n", classification_rep1)
print("\n")

#SVM
svm = svm.SVC()

svm.fit(X_train,Y_train)

y_pred2 = svm.predict(X_test)

print("SVM accuracy:")
print((accuracy_score(Y_test,y_pred2))*100)
conf_matrix1 = confusion_matrix(Y_test,y_pred2)
classification_rep1 = (classification_report(Y_test,y_pred2))
print("Confusion Matrix:\n", conf_matrix1)
print("Classification Report:\n", classification_rep1)
print("\n")

#KNN
knn = KNeighborsClassifier(n_neighbors=2)

knn.fit(X_train,Y_train)

y_pred3 = knn.predict(X_test)
print("KNN accuracy:")
print((accuracy_score(Y_test,y_pred3))*100)
conf_matrix2 = confusion_matrix(Y_test,y_pred3)
classification_rep2 = (classification_report(Y_test,y_pred3))
print("Confusion Matrix:\n", conf_matrix2)
print("Classification Report:\n", classification_rep2)
print("\n")


#random forest
# Initialize and train the Random Forest model
random_forest_model = RandomForestClassifier(n_estimators=100, random_state=42)
random_forest_model.fit(X_train, Y_train)

# Make predictions on the test set
y_pred = random_forest_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(Y_test, y_pred)
conf_matrix = confusion_matrix(Y_test, y_pred)
classification_rep = classification_report(Y_test, y_pred)

print("Random Forest:")
print(f"Test Accuracy: {accuracy * 100:.2f}%")
print("Confusion Matrix:\n", conf_matrix)
print("Classification Report:\n", classification_rep)



#My Neural Network

# import pandas as pd
# import numpy as np
# import tensorflow as tf
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.impute import SimpleImputer
# from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support

# Set seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Load your dataset

# from ucimlrepo import fetch_ucirepo
#
# fetch dataset
# heart_disease = fetch_ucirepo(id=45)
#
# data (as pandas dataframes)
# X = heart_disease.data.features
# Y = heart_disease.data.targets

# data = pd.read_csv('dataset.csv')
#
# # Check for and handle duplicate values
# data = data.drop_duplicates()
#
# # Invalid data removal (remove rows with missing values)
# data = data.dropna()
#
# # Remove outliers using Z-score
# z_scores = np.abs((data - data.mean()) / data.std())
# data_no_outliers = data[(z_scores < 3).all(axis=1)]
#
# # Data preprocessing
# X = data_no_outliers.drop('target', axis=1)
# y = data_no_outliers['target']
#
# # Handle missing values using SimpleImputer
# imputer = SimpleImputer(strategy='mean')
# X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
#
# # Split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)
#
# # Standardize the features
# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)

# Build the neural network model with further adjustments
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(256, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model with a lower learning rate
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# Implement early stopping
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train the model with more epochs and early stopping
model.fit(X_train, Y_train, epochs=150, batch_size=128, validation_split=0.2, callbacks=[early_stopping])

# Evaluate the model on the test set
y_pred = (model.predict(X_test) > 0.5).astype("int32")
accuracy = accuracy_score(Y_test, y_pred)
conf_matrix = confusion_matrix(Y_test, y_pred)
classification_rep = classification_report(Y_test, y_pred, zero_division=1)

print(f"Test Accuracy: {accuracy * 100:.2f}%")
print("Confusion Matrix:\n", conf_matrix)
print("Classification Report:\n", classification_rep)


# new_data = pd.dataframe({
#     'age':40,
#     'sex': 1,
#     'cp': 0,
#     'trestbps':125,
#     'chol': 233,
#     'fbs': 0,
#     'restecg': 1,
#     'thalach': 178,
#     'exang': 0,
#     'oldpeak': 1,
#     'slope': 2,
#     'ca': 2,
#     'thal': 3
#
# },index = [0])
