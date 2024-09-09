from flask.Neural import *
from datapreprocess2 import *

import numpy as np
import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.preprocessing import LabelEncoder

# Assuming 'model' is your original neural network model

# Function to create a KerasClassifier
def create_keras_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Create a KerasClassifier
keras_classifier = KerasClassifier(build_fn=create_keras_model, epochs=50, batch_size=32, verbose=0)

# Train the Keras model
keras_classifier.fit(X_train, y_train)

# Train the RandomForest model
random_forest_model = RandomForestClassifier()
random_forest_model.fit(X_train, y_train)

# Make predictions on the test set
keras_predictions = keras_classifier.predict(X_test)
rf_predictions = random_forest_model.predict(X_test)

# Convert multi-label format to binary format
y_test_binary = LabelEncoder().fit_transform(y_test)

# Combine predictions using a simple averaging approach
ensemble_predictions = np.round((keras_predictions + rf_predictions) / 2).astype(int)

# Evaluate the ensemble model
accuracy = accuracy_score(y_test_binary, ensemble_predictions)
conf_matrix = confusion_matrix(y_test_binary, ensemble_predictions)
classification_rep = classification_report(y_test_binary, ensemble_predictions)

print(f"Ensemble Test Accuracy: {accuracy * 100:.2f}%")
print("Confusion Matrix:\n", conf_matrix)
print("Classification Report:\n", classification_rep)
