# Artificial Intelligence Project 2
# Data Classification Using AI
# DecodeLabs


# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix



# LOAD DATASET


# Load the Iris dataset
iris = load_iris()

# Create DataFrame
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = iris.target

# Display first 5 rows
print("\nFirst 5 Rows of Dataset:\n")
print(X.head())

# Display dataset information
print("\nDataset Information:\n")
print(X.info())

# Display statistical summary
print("\nStatistical Summary:\n")
print(X.describe())



# VISUALIZE DATA


# Pairplot visualization
sns.pairplot(
    pd.concat([X, pd.DataFrame(y, columns=['target'])], axis=1),
    hue='target'
)

plt.suptitle("Iris Dataset Visualization", y=1.02)
plt.show()



# SPLIT DATA INTO TRAINING AND TESTING SETS


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining Data Shape:", X_train.shape)
print("Testing Data Shape:", X_test.shape)



# FEATURE SCALING


scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)



# TRAIN CLASSIFICATION MODEL


# Create model
model = KNeighborsClassifier(n_neighbors=3)

# Train model
model.fit(X_train_scaled, y_train)

print("\nModel training completed successfully.")



# MAKE PREDICTIONS


# Predict test data
predictions = model.predict(X_test_scaled)

print("\nPredictions:\n")
print(predictions)



# EVALUATE MODEL


# Accuracy
accuracy = accuracy_score(y_test, predictions)

print(f"\nModel Accuracy: {accuracy * 100:.2f}%")

# Classification report
print("\nClassification Report:\n")
print(classification_report(y_test, predictions))

# Confusion matrix
cm = confusion_matrix(y_test, predictions)

print("\nConfusion Matrix:\n")
print(cm)



# CONFUSION MATRIX VISUALIZATION


plt.figure(figsize=(6, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=iris.target_names,
    yticklabels=iris.target_names
)

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.show()

# TEST WITH NEW DATA


# Example flower measurements
new_data = np.array([[5.1, 3.5, 1.4, 0.2]])

# Scale new data
new_data_scaled = scaler.transform(new_data)

# Predict class
prediction = model.predict(new_data_scaled)

print("\nPrediction for New Data:")
print("Predicted Flower Type:", iris.target_names[prediction[0]])

