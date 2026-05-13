import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score
)

from imblearn.over_sampling import SMOTE

import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("data/creditcard.csv")

# Reduce size for faster training
df = df.sample(50000, random_state=42)

# Features and target
X = df.drop("Class", axis=1)
y = df["Class"]

# Scaling
scaler = StandardScaler()

X["Amount"] = scaler.fit_transform(X[["Amount"]])
X["Time"] = scaler.fit_transform(X[["Time"]])

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# SMOTE
smote = SMOTE(random_state=42)

X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# Random Forest Model
model = RandomForestClassifier(
    n_estimators=20,
    random_state=42,
    n_jobs=-1
)

# Train
model.fit(X_train_smote, y_train_smote)

# Predict
y_pred = model.predict(X_test)

# Metrics
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

print("\nROC-AUC Score:")
print(roc_auc_score(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

sns.heatmap(cm, annot=True, fmt="d", cmap="Greens")

plt.title("Random Forest Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.close()



#feature importances

importance=pd.Series(model.feature_importances_, index=X.columns)
importance =importance.sort_values(ascending=False)
print(importance.head(10))
importance.head(10).plot(kind="bar")
plt.title("feature Importances")
plt.tight_layout()
plt.savefig("feature_importance.png")
plt.close()


#save model
import joblib
joblib.dump(model, "models/random_forest_fraud.pkl")