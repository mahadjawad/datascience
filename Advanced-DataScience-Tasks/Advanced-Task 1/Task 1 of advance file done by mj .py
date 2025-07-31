#!/usr/bin/env python
# coding: utf-8

# # task1

# In[6]:


import pandas as pd  # For handling data

# Step 1: Load CSV with correct separator (;)
df = pd.read_csv(r'C:\Users\mahad\Downloads\bank+marketing\bank-additional\bank-additional\bank-additional-full.csv', sep=';')

# Step 2: Clean column names (remove spaces and quotes)
df.columns = [col.strip().replace('"', '') for col in df.columns]

# Step 3: Clean all text values in the table
for col in df.columns:
    if df[col].dtype == 'object':  # only clean string columns
        df[col] = df[col].str.strip().str.replace('"', '')

# Step 4: Remove any duplicate rows
df.drop_duplicates(inplace=True)

# Step 5: Reset index after cleaning
df.reset_index(drop=True, inplace=True)

# Step 6: Show basic info
print("✅ Cleaned data shape:", df.shape)
print("\n✅ Column names:", df.columns.tolist())
print("\n✅ Missing values:\n", df.isnull().sum())
print("\n✅ First 5 rows:\n", df.head())


# In[7]:


print(df)


# In[8]:


df.shape


# In[9]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# One-Hot Encode all categorical variables
df_encoded = pd.get_dummies(df, drop_first=True)  # drop_first avoids dummy variable trap

# Split into features and target
X = df_encoded.drop('y_yes', axis=1)  # Because 'y' becomes 'y_no' and 'y_yes' after get_dummies
y = df_encoded['y_yes']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print("✅ Accuracy:", accuracy_score(y_test, y_pred))
print("\n✅ Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\n✅ Classification Report:\n", classification_report(y_test, y_pred))


# In[10]:


import matplotlib.pyplot as plt

feature_importances = model.feature_importances_
features = X.columns

plt.figure(figsize=(10, 6))
plt.barh(features, feature_importances)
plt.title("Feature Importance")
plt.xlabel("Importance")
plt.ylabel("Features")
plt.show()


# In[12]:


# ✅ 1. Random Forest Model
from sklearn.linear_model import LogisticRegression
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
rf_prob = rf_model.predict_proba(X_test)[:, 1]

# ✅ 2. Logistic Regression Model
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)
lr_prob = lr_model.predict_proba(X_test)[:, 1]

# ✅ Evaluation Function
def evaluate_model(name, y_true, y_pred, y_prob):
    print(f"\n✅ {name} Accuracy:", accuracy_score(y_true, y_pred))
    print(f"✅ Confusion Matrix:\n", confusion_matrix(y_true, y_pred))
    print(f"✅ Classification Report:\n", classification_report(y_true, y_pred))
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc_score(y_true, y_prob):.2f})')

# ✅ Evaluate Both Models
plt.figure(figsize=(8,6))
evaluate_model("Random Forest", y_test, rf_pred, rf_prob)
evaluate_model("Logistic Regression", y_test, lr_pred, lr_prob)
plt.plot([0,1], [0,1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.grid()
plt.show()

# ✅ SHAP Explainability on Random Forest
explainer = shap.Explainer(rf_model, X_train)
shap_values = explainer(X_test)

# ✅ Beeswarm plot
shap.plots.beeswarm(shap_values)

# ✅ Explain 5 individual predictions using waterfall plots
shap.initjs()
for i in range(5):
    print(f"🔎 Explaining prediction {i+1}")
    shap.plots.waterfall(shap_values[i])


# In[19]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import shap
import numpy as np

# Load Bank Marketing dataset
df = pd.read_csv(r'C:\Users\mahad\Downloads\bank+marketing\bank-additional\bank-additional\bank-additional-full.csv', sep=';')

# Clean string columns
df.columns = [col.strip().replace('"', '') for col in df.columns]
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = df[col].str.strip().str.replace('"', '')

# One-hot encode categorical features
df_encoded = pd.get_dummies(df, drop_first=True)

# Separate features and target
X = df_encoded.drop('y_yes', axis=1)
y = df_encoded['y_yes']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ✅ 1. Random Forest Model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
rf_prob = rf_model.predict_proba(X_test)[:, 1]

# ✅ 2. Logistic Regression Model
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)
lr_prob = lr_model.predict_proba(X_test)[:, 1]

# ✅ Evaluation Function
def evaluate_model(name, y_true, y_pred, y_prob):
    print(f"\n✅ {name} Accuracy:", accuracy_score(y_true, y_pred))
    print(f"✅ Confusion Matrix:\n", confusion_matrix(y_true, y_pred))
    print(f"✅ Classification Report:\n", classification_report(y_true, y_pred))
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc_score(y_true, y_prob):.2f})')

# ✅ Evaluate Both Models
plt.figure(figsize=(8,6))
evaluate_model("Random Forest", y_test, rf_pred, rf_prob)
evaluate_model("Logistic Regression", y_test, lr_pred, lr_prob)
plt.plot([0,1], [0,1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.grid()
plt.show()

# ✅ SHAP Explainability on Random Forest
explainer = shap.Explainer(rf_model, X_train)
shap_values = explainer(X_test)

# ✅ Beeswarm plot
shap.plots.beeswarm(shap_values)

# ✅ Explain 5 individual predictions using waterfall plots
shap.initjs()
for i in range(5):
    print(f"🔎 Explaining prediction {i+1}")
    shap.plots.waterfall(shap_values[i])


# In[ ]:




