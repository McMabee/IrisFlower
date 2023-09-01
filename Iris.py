# Iris Flower Project
# Ty Mabee
# Understanding Machine Learning w/ Python

from sklearn.datasets import load_iris

# Load the Iris dataset
iris = load_iris()

# Separate features (X) and target variable (y)
X = iris.data
y = iris.target

from sklearn.preprocessing import StandardScaler

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

import matplotlib.pyplot as plt

# Create scatter plots
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.title("Sepal Length vs Sepal Width")
plt.show()

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

from sklearn.linear_model import LogisticRegression

# Initialize the model
model = LogisticRegression()
model.fit(X_train, y_train)

from sklearn.metrics import accuracy_score, classification_report

# Make predictions
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Generate a classification report
report = classification_report(y_test, y_pred, target_names=iris.target_names)
print("Classificationi Report:")
print (report)

# Prediction for a new data point
new_data_point = [[5.1, 3.5, 1.4, 0.2]]
new_data_point_scaled = scaler.transform(new_data_point)
predicted_class = model.predict(new_data_point_scaled)
predicted_species = iris.target_names[predicted_class[0]]
print(f"Predictied Speciesi: {predicted_species}")