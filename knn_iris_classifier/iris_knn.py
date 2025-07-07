import pandas as pd
import math
from collections import Counter

# Load the dataset
data = pd.read_csv("resources/iris.csv")

# Convert to list of tuples [(features), label]
dataset = list(zip(data["SepalLength"], data["SepalWidth"], 
                   data["PetalLength"], data["PetalWidth"], data["Name"]))

# Function to calculate Euclidean distance
def euclidean_distance(p1, p2):
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(p1, p2)))

# Function to get the predicted class
def knn_predict(test_point, k=3):
    distances = []
    i = 0
    while i < len(dataset):
        train_point = dataset[i][:4]
        label = dataset[i][4]
        distance = euclidean_distance(test_point, train_point)
        distances.append((distance, label))
        i += 1

    # Sort by distance
    distances.sort(key=lambda x: x[0])

    # Get top k labels
    top_k_labels = [label for _, label in distances[:k]]

    # Get most common label
    most_common = Counter(top_k_labels).most_common(1)[0][0]

    return most_common

# Accept input
if __name__ == "__main__":
    print("Enter sepal length, sepal width, petal length, and petal width:")
    try:
        sl = float(input("Sepal Length (cm): "))
        sw = float(input("Sepal Width (cm): "))
        pl = float(input("Petal Length (cm): "))
        pw = float(input("Petal Width (cm): "))
        specify_k_bool = input("Do you want to specify the value of k? (yes/no): ").strip().lower()
        if specify_k_bool == 'yes' or specify_k_bool == 'y':
            k = int(input("Value of k (default is 3): ") or "3")

        test_input = (sl, sw, pl, pw)
        if specify_k_bool != 'yes':
            k = 3
        prediction = knn_predict(test_input, k)
        print(f"\nPredicted Iris type: {prediction}")
    except ValueError:
        print("Invalid input. Please enter floating point numbers.")
