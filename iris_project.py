import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

iris = load_iris()
X = iris.data
Y = iris.target

df = pd.DataFrame(X, columns=iris.feature_names)
df['species'] = Y
print(df.head())
print("\nShape of data:", df.shape)
print("\nSpecies count:")
print(df['species'].value_counts())

plt.figure(figsize=(8, 5))
colors = ['red' , 'green', 'blue']
for i in range(3):
    subset = df[df['species'] == i]
    plt.scatter(subset['sepal length (cm)'],
                subset['sepal width (cm)'],
                c=colors[i],
                label=iris.target_names[i])
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Sepal Width (cm)')
plt.title('Iris Dataset - Sepal Dimensions by Species')
plt.legend()
plt.savefig('iris_scatter_plot.png')
plt.show()
print("Plot saved as iris_scatter_plot.png")

X_train, X_test , Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
print("\nTraining samples:",len(X_train))
print("Testing samples:",len(X_test))

# Train the model
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, Y_train)
print("\nModel trained successfully!")

# Test the model
predictions = model.predict(X_test)
accuracy = accuracy_score(Y_test, predictions)
print("\nModel Accuracy:", round(accuracy*100, 2), "%")

# Predict the new flower
new_flower = [[5.1, 3.5, 1.4, 0.2]]
predicted_species = model.predict(new_flower)
print("\nPredicted species for the new flower:")
print("Species:",iris.target_names[predicted_species][0])

input("Press Enter to exit...")