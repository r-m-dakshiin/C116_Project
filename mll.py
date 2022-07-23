import pandas as pd 
import plotly.express as px 
df = pd.read_csv('Admission_Predict.csv')
gre = df["GRE Score"].tolist()
toefl = df["TOEFL Score"].tolist()
coa = df["Chance of admit"]
fig = px.scatter(x=gre, y=coa)
fig.show()

import plotly.graph_objects as go
colours = []
for data in coa:
  if data==1:
    colours.append("green")
  else:
    colours.append("red")
fig = go.Figure(go.Scatter(x=toefl,y=gre,mode='markers', marker=dict(color = colours)))
fig.show()

score = df[["GRE Score", "TOEFL Score"]]
results = df["Chance of admit"]
from sklearn.model_selection import train_test_split
score_train, score_test, results_train, results_test = train_test_split(score,results,test_size = 0.25, random_state = 0)
print(score_train)

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(score_train, results_train)

results_pred = classifier.predict(score_test)
from sklearn.metrics import accuracy_score
print(f"Accuracy {accuracy_score(results_test, results_pred)}")


user_gre_score = int(input("Enter GRE Score : "))
user_toefl_score = int(input("Enter TOEFL Score : "))
user_test = [[user_gre_score, user_toefl_score]]
user_result_pred = classifier.predict(user_test)
if user_result_pred[0]==1:
  print("This user may be admitted")
else:
  print("This user may not be admitted")