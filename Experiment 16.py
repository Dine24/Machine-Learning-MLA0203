import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import PassiveAggressiveClassifier

iris = pd.read_csv("IRIS.csv")

x = iris.drop("species", axis=1)
y = iris["species"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

decisiontree = DecisionTreeClassifier()
logisticregression = LogisticRegression()
knearestclassifier = KNeighborsClassifier()
bernoulli_naiveBayes = BernoulliNB()
passiveAggressive = PassiveAggressiveClassifier()

knearestclassifier.fit(x_train, y_train)
decisiontree.fit(x_train, y_train)
logisticregression.fit(x_train, y_train)
passiveAggressive.fit(x_train, y_train)

data1 = {"Classification Algorithms": ["KNN Classifier", "Decision Tree Classifier", 
                                       "Logistic Regression", "Passive Aggressive Classifier"],
         "Score": [knearestclassifier.score(x_test, y_test), decisiontree.score(x_test, y_test), 
                   logisticregression.score(x_test, y_test), passiveAggressive.score(x_test, y_test)]}
score = pd.DataFrame(data1)
print(score)
1
