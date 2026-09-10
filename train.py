import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Load dataset
df = pd.read_csv("data/titanic.csv")

# Fill missing values
df["Age"] = df["Age"].fillna(df["Age"].median())
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

# Preprocessing & artifact creation
le_sex = LabelEncoder()
le_embarked = LabelEncoder()

df["Sex"] = le_sex.fit_transform(df["Sex"])
df["Embarked"] = le_embarked.fit_transform(df["Embarked"])

# Define features and target
X = df[["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]]
y = df["Survived"]

# 80-20 Train-Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression(max_iter=500).fit(X_train, y_train)

# Save preprocessing and model artifacts
joblib.dump(model, "models/logistic_model.pkl")
joblib.dump(le_sex, "models/le_sex.pkl")
joblib.dump(le_embarked, "models/le_embarked.pkl")

print(f"Test Accuracy: {model.score(X_test, y_test):.2%}")
print("Artifacts saved: 'logistic_model.pkl', 'le_sex.pkl', 'le_embarked.pkl'")