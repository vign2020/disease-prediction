# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

df = pd.read_csv('heart.csv')

df2=df.copy()
df2.drop('slope',axis=1,inplace=True)

X3=df2.drop('target',axis=1)
y3=df2['target']

from sklearn.model_selection import train_test_split
X3_train,X3_test,y3_train,y3_test=train_test_split(X3,y3,test_size=0.3,random_state=42)

from sklearn.ensemble import RandomForestClassifier
model3 = RandomForestClassifier(n_estimators=20)
model3.fit(X3_train, y3_train)

print(model3.score(X3_test,y3_test))

model3gs=RandomForestClassifier(criterion= 'gini',max_depth= 5, max_features= 'auto', n_estimators= 200,random_state=42)


model3gs.fit(X3_train,y3_train)
print(model3gs.score(X3_test,y3_test))


def predict_disease():
    age=int(input())
    sex=int(input())
    cp=int(input())
    trestbps=int(input())
    chol=int(input())
    fbs=int(input())
    restcg=int(input())
    thalach=int(input())
    exang=int(input())
    oldpeak=float(input())
    ca=int(input())
    thal=int(input())
    
    ans=model3gs.predict([[age,sex,cp,trestbps,chol,fbs,restcg,thalach,exang,oldpeak,ca,thal]])[0]

    return ans

# Saving model to disk
pickle.dump(model3gs, open('model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
# print(model.predict([[2, 9, 6]]))