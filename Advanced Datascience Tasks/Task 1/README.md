Advanced Task 1: Term Deposit Subscription Prediction
Objective:
Predict whether a bank customer will subscribe to a term deposit as a result of a direct marketing campaign.

Dataset:
Name: Bank Marketing Dataset (Additional)
File Used: bank-additional.csv
Source: https://archive.ics.uci.edu/ml/datasets/Bank+Marketing
Target Variable: y (binary: yes or no)
Features: Customer profile, last contact info, campaign history
Workflow Summary:
Data Preprocessing:
Loaded dataset and explored structure
Removed duplicates
Categorical columns were handled using:
Label Encoding for binary fields (default, housing, loan, y)
One-Hot Encoding for nominal fields (job, education, marital, contact, month, poutcome)
Model Training:
Trained two classification models:

Logistic Regression
Random Forest Classifier
Evaluation Metrics:
Evaluated models using:

Confusion Matrix
F1-Score
ROC Curve and AUC Score
Model	Accuracy	F1-Score	AUC Score
Logistic Regression	89.4%	0.68	0.91
Random Forest	90.0%	0.72	0.92
The Random Forest model outperformed Logistic Regression slightly in both F1-score and AUC.

Explainable AI (XAI):
Used SHAP (SHapley Additive Explanations) to interpret model predictions:

Visualized global feature importance
Generated force plots for 5 individual customers
Key influential features included:
duration of last contact
month of campaign
previous campaign outcome
contact type
Key Takeaways
Longer call durations increased the probability of term deposit subscriptions.
SHAP helped explain why the model made a certain prediction, enhancing trust and transparency.
Feature duration was the most influential across all predictions.
Tools & Libraries Used:
Python
pandas, numpy
scikit-learn (LogisticRegression, RandomForestClassifier, evaluation metrics)
matplotlib, seaborn
SHAP for model interpretability
