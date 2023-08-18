# import required libraries
from sklearn.datasets import load_iris
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score

# load iris dataset
iris = load_iris()

# split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.3, random_state=0)

# create Naive Bayes classifier
classifier = GaussianNB()

# train the classifier using the training data
classifier.fit(X_train, y_train)

# predict the target values for the testing data
y_pred = classifier.predict(X_test)

# display confusion matrix and accuracy score
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

acc = accuracy_score(y_test, y_pred)
print("Accuracy Score:", acc)
