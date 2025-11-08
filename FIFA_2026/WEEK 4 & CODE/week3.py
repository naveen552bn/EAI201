# ===========================================================
# FIFA 2026 Finalist Prediction – Week 3 (Model Optimization & Final Prediction)
# ===========================================================

# STEP 1: Upload dataset from Week 1
from google.colab import files
print("Upload fifa_1930_2022_with_rank.csv file")
uploaded = files.upload()

# STEP 2: Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
)

# STEP 3: Load Dataset
df = pd.read_csv("fifa_1930_2022_with_rank.csv")
print("Data Loaded:", df.shape)
print(df.head())

# STEP 4: Define Features and Target
X = df[['Goals_For', 'Goals_Against', 'Goal_Difference', 'Win_Rate', 'FIFA_Rank', 'FIFA_Points']]
y = df['Is_Finalist']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# MODEL 1: Optimized Random Forest
rf = RandomForestClassifier(random_state=42)
param_grid = {
    'n_estimators': [200, 300, 400],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10],
    'max_features': ['sqrt', 'log2']
}
grid_rf = GridSearchCV(rf, param_grid, cv=5, scoring='f1', n_jobs=-1, verbose=1)
grid_rf.fit(X_train, y_train)

best_rf = grid_rf.best_estimator_
print("Best Random Forest Parameters:", grid_rf.best_params_)


# MODEL 2: Gradient Boosting (XGBoost Alternative)

gb = GradientBoostingClassifier(random_state=42)
param_gb = {
    'n_estimators': [100, 200],
    'learning_rate': [0.05, 0.1, 0.2],
    'max_depth': [3, 5, 7]
}
grid_gb = GridSearchCV(gb, param_gb, cv=5, scoring='f1', n_jobs=-1, verbose=1)
grid_gb.fit(X_train_scaled, y_train)

best_gb = grid_gb.best_estimator_
print("Best Gradient Boosting Parameters:", grid_gb.best_params_)


# STEP 5. Cross-Validation

cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
rf_cv = cross_val_score(best_rf, X, y, cv=cv, scoring='f1')
gb_cv = cross_val_score(best_gb, X, y, cv=cv, scoring='f1')

print("\n10-Fold Cross Validation (Mean F1):")
print(f"Random Forest: {rf_cv.mean():.3f}")
print(f"Gradient Boosting: {gb_cv.mean():.3f}")


# STEP 6. Evaluate Models

def evaluate_model(model, X_test, y_test, name):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)
    print(f"\n{name} Performance:")
    print(f"Accuracy: {acc:.3f}, Precision: {prec:.3f}, Recall: {rec:.3f}, F1: {f1:.3f}, ROC-AUC: {roc_auc:.3f}")
    return [acc, prec, rec, f1, roc_auc]

rf_results = evaluate_model(best_rf, X_test, y_test, "Optimized Random Forest")
gb_results = evaluate_model(best_gb, X_test_scaled, y_test, "Gradient Boosting")


# STEP 7. Compare Models

results = pd.DataFrame(
    [rf_results, gb_results],
    columns=['Accuracy', 'Precision', 'Recall', 'F1', 'ROC-AUC'],
    index=['Optimized RF', 'Gradient Boosting']
)
print("\nModel Comparison:\n", results)

# Visualization
results.plot(kind='bar', figsize=(8,5), colormap='viridis', legend=True)
plt.title("Model Performance Comparison (Week 3)")
plt.ylabel("Score")
plt.xticks(rotation=0)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()


# STEP 8. Feature Importance (RF)

feat_imp = pd.DataFrame({
    'Feature': X.columns,
    'Importance': best_rf.feature_importances_
}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(7,4))
sns.barplot(x='Importance', y='Feature', data=feat_imp)
plt.title("Feature Importance (Optimized Random Forest)")
plt.show()


# STEP 9. Predict FIFA 2026 Finalists

latest_year = df['Year'].max()
latest_data = df[df['Year'] == latest_year].copy()
latest_scaled = scaler.transform(latest_data[X.columns])
latest_data['Prob_RF'] = best_rf.predict_proba(latest_data[X.columns])[:, 1]
latest_data['Prob_GB'] = best_gb.predict_proba(latest_scaled)[:, 1]

top_rf = latest_data[['Team', 'Prob_RF']].sort_values(by='Prob_RF', ascending=False).head(10)
top_gb = latest_data[['Team', 'Prob_GB']].sort_values(by='Prob_GB', ascending=False).head(10)

print("\nTop 10 Teams (Optimized Random Forest):")
print(top_rf)
print("\nTop 10 Teams (Gradient Boosting):")
print(top_gb)


# STEP 10. Save Outputs

results.to_csv("Week3_Model_Comparison.csv", index=True)
top_rf.to_csv("Week3_Top10_RF.csv", index=False)
top_gb.to_csv("Week3_Top10_GB.csv", index=False)
feat_imp.to_csv("Week3_Feature_Importance_RF.csv", index=False)

from google.colab import files
files.download("Week3_Model_Comparison.csv")
files.download("Week3_Top10_RF.csv")
files.download("Week3_Top10_GB.csv")
files.download("Week3_Feature_Importance_RF.csv")

print("\n Week 3 completed successfully — Models optimized and finalists predicted.")
