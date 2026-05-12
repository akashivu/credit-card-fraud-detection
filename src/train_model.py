import pandas as pd
from sklearn.preprocessing import  StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_auc_score
    )

#Load dataset

df = pd.read_csv("data/creditcard.csv")

#split dataset

X=df.drop("Class", axis=1)
y=df["Class"]

#Scaling
scaler=StandardScaler()

X["Amount"]=scaler.fit_transform(X[["Amount"]])
X["Time"]=scaler.fit_transform(X[["Time"]])

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20,random_state=42, stratify=y)

#model
model=LogisticRegression()

#Train
model.fit(X_train,y_train)

#predict
y_pred=model.predict(X_test)

#evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:\n")
print(confusion_matrix(y_test, y_pred))

print("\nROC-AUC Score:")
print(roc_auc_score(y_test, y_pred))


#confusion matrix

import seaborn as sns
import matplotlib.pyplot as plt

cm=confusion_matrix(y_test,y_pred)

sns.heatmap(cm,annot=True,fmt="d", cmap="Blues")

plt.title("confusion matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.show()

#Save 

import joblib
joblib.dump(model, "models/fraud_model.pkl")