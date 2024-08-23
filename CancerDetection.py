
from sklearn.datasets import load_breast_cancer
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score

# load the data set

data = load_breast_cancer()
label_names = data["target_names"]
labels = data["target"]
feature_names = data["feature_names"]
features = data["data"]

# Print dataset information

print(label_names)
print("Class label :", labels[0])
print(feature_names)
print(features[0], "\n")

# Split the dataset into training and testing sets

train, test, train_labels, test_labels = train_test_split(features, labels,
                                                          test_size=0.2,
                                                          random_state=42)

# Initialize and train the classifier

gnb = GaussianNB()
gnb.fit(train, train_labels)
preds = gnb.predict(test)

# Print predictions and accuracy score

print(preds, "\n")
print(accuracy_score(test_labels, preds))