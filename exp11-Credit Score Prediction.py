import pandas as pd
import numpy as np
import plotly.express as px   
import plotly.graph_objects as go
import plotly.io as pio
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

pio.templates.default = "plotly_white"

data = pd.read_csv("CREDITSCORE.csv")
print(data.head())

print(data.info())

x = data[["Annual_Income", "Monthly_Inhand_Salary", 
          "Num_Bank_Accounts", "Num_Credit_Card", 
          "Interest_Rate", "Num_of_Loan", 
          "Delay_from_due_date", "Num_of_Delayed_Payment", 
          "Credit_Mix", "Outstanding_Debt", 
          "Credit_History_Age", "Monthly_Balance"]]
y = data[["Credit_Score"]]

# Convert "Credit_Mix" to numerical representation using .loc method
x.loc[:, "Credit_Mix"] = x["Credit_Mix"].map({"Bad": 0, "Standard": 1, "Good": 3})

# Handle missing values (if any)
# x.fillna(0, inplace=True)

# Scale numerical features
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

xtrain, xtest, ytrain, ytest = train_test_split(x_scaled, y,test_size=0.33,random_state=42)

model = RandomForestClassifier()
model.fit(xtrain, ytrain.values.ravel())

print("Credit Score Prediction : ")
a = float(input("Annual Income: "))
b = float(input("Monthly Inhand Salary: "))
c = float(input("Number of Bank Accounts: "))
d = float(input("Number of Credit cards: "))
e = float(input("Interest rate: "))
f = float(input("Number of Loans: "))
g = float(input("Average number of days delayed by the person: "))
h = float(input("Number of delayed payments: "))
i = input("Credit Mix (Bad: 0, Standard: 1, Good: 3) : ")
j = float(input("Outstanding Debt: "))
k = float(input("Credit History Age: "))
l = float(input("Monthly Balance: "))

# Convert user input for Credit Mix to integer
credit_mix_map = {"Bad": 0, "Standard": 1, "Good": 3}
i = credit_mix_map.get(i)

features = np.array([[a, b, c, d, e, f, g, h, i, j, k, l]])
print("Predicted Credit Score = ", model.predict(features))
