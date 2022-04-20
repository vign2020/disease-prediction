import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

df=pd.read_csv('diabetes.csv')

upper_mark=df['Pregnancies'].quantile(0.95)


df2=df.copy()
df2=df[df['Pregnancies']<upper_mark]

X=df2.drop('Outcome',axis=1)
y=df2['Outcome']

df3=df2.copy()
df3.drop(['Pregnancies','SkinThickness','BloodPressure'],axis=1,inplace=True)

X1=df3.drop('Outcome',axis=1)
y1=df3['Outcome']

from sklearn.model_selection import train_test_split
X1_train,X1_test,y1_train,y1_test=train_test_split(X1,y1,test_size=0.3)

df4=df3.copy()
df4=df4[df4['BMI']!=0]
maxi=df4['BMI'].max()
df4=df4[df4['BMI']!=maxi]

upper_mark=df4['DiabetesPedigreeFunction'].quantile(0.95)
df4=df4[df4['DiabetesPedigreeFunction']<=upper_mark]

lower_mark=df['Glucose'].quantile(0.04)
df4=df4[df4['Glucose']>=lower_mark]


X2=df4.drop('Outcome',axis=1)
y2=df4['Outcome']
X2_train,X2_test,y2_train,y2_test=train_test_split(X2,y2,test_size=0.3)

from sklearn.ensemble import RandomForestClassifier
model2 = RandomForestClassifier(n_estimators=20)
model2.fit(X2_train, y2_train)

print(model2.score(X2_test, y2_test))

def predict_disease():
    gluc=int(input())
    insulin=int(input())
    bmi=float(input())
    dpf=float(input())
    age=int(input())

    ans=model2.predict([[gluc,insulin,bmi,dpf,age]])[0]

    return ans

pickle.dump(model2, open('model2.pkl','wb'))

# Loading model to compare the results
model2 = pickle.load(open('model2.pkl','rb'))