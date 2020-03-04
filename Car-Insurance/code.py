# --------------
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Code starts here
df = pd.read_csv(path)
print(df.head())
print(df.info())
a = ['INCOME','HOME_VAL','BLUEBOOK','OLDCLAIM','CLM_AMT']
for i in range(0,4):
    df[a[i]] = df[a[i]].str.replace('\W','')
print(df.head())
X = df.iloc[:,:-1]
y = df.iloc[:,-1]
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0,random_state=6)

# Code ends here


# --------------
# Code starts here
X_train[a] = X_train[a].astype(float)
X_test[a] = X_test[a].astype(float)
print(X_train.info())
print(X_test.info())

# Code ends here


# --------------
# Code starts here
X_train.dropna(axis=0,subset=['YOJ','OCCUPATION'],inplace=True)
X_test.dropna(axis=0,subset=['YOJ','OCCUPATION'],inplace=True)
print(X_train.shape)
y_train = y_train[X_train.index]
y_test = y_test[X_test.index]
X_train[['AGE','CAR_AGE','INCOME','HOME_VAL']].fillna(X_train.mean(),inplace=True)
X_test[['AGE','CAR_AGE','INCOME','HOME_VAL']].fillna(X_train.mean(),inplace=True)
# Code ends here


# --------------
from sklearn.preprocessing import LabelEncoder
columns = ["PARENT1","MSTATUS","GENDER","EDUCATION","OCCUPATION","CAR_USE","CAR_TYPE","RED_CAR","REVOKED"]

# Code starts here
for i in range (0,len(columns)):
    le = LabelEncoder()
    X_train[columns[i]] = le.fit_transform(X_train[columns[i]].astype(str))
    X_test[columns[i]] = le.transform(X_test[columns[i]].astype(str))

# Code ends here



# --------------
from sklearn.metrics import precision_score 
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression



# code starts here 
model = LogisticRegression(random_state=6)
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
score = accuracy_score(y_test,y_pred)
print(score)

# Code ends here


# --------------
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

# code starts here
smote = SMOTE(random_state=9)
X_train, y_train = smote.fit_sample(X_train,y_train)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
# Code ends here


# --------------
# Code Starts here
model = LogisticRegression(random_state=6)
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
score = accuracy_score(y_test,y_pred)
print(score)

# Code ends here


