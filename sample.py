# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

# Load dataset
iris = pd.read_csv('/content/IRIS.csv')

# Display shape and first few rows
print("Dataset Shape:", iris.shape)
print(iris.head())

# Basic visualization
px.scatter(iris, x='species', y='petal_width', size='petal_width', color='species')

# Bar Plot - Average Petal Width by Species
iris_avg = iris.groupby('species')['petal_width'].mean().reset_index()
plt.bar(iris_avg['species'], iris_avg['petal_width'], color=['orange', 'skyblue', 'green'])
plt.title('Average Petal Width by Species')
plt.xlabel('Species')
plt.ylabel('Petal Width')
plt.show()

# Pair plot
sns.pairplot(iris, hue='species')
plt.show()

# Label Encoding
le = LabelEncoder()
iris['species'] = le.fit_transform(iris['species'])

# Features & Labels
X = iris.drop('species', axis=1)
y = iris['species']

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Models dictionary
models = {
    'Logistic Regression': LogisticRegression(max_iter=200),
    'Decision Tree': DecisionTreeClassifier(),
    'KNN': KNeighborsClassifier(),
    'SVM': SVC()
}

# Train all models
for name, model in models.items():
    model.fit(X_train, y_train)

# Evaluate KNN model only, matching your output format
knn = models['KNN']
preds = knn.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, preds)

# Confusion Matrix
cm = confusion_matrix(y_test, preds)

# Class names from label encoder
class_names = le.classes_  # ['Setosa', 'Versicolor', 'Virginica']

print("\nSample Output:")
print("\nModel Evaluation Metrics:\n")
print(f"Accuracy: {accuracy * 100:.0f}%")
print("Confusion Matrix:\n")

print("        Predicted")
print("       S  Vc  Vg")
for i, row in enumerate(cm):
    print(f"Actual {class_names[i][:2]} ", end='')
    for val in row:
        print(f"{val:3}", end=' ')
    print()

print("\nClassification Report:\n")
print(classification_report(y_test, preds, target_names=class_names))
