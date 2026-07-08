"""
train_model.py
Run this script once to train all 4 ML models and save them.
Usage: python train_model.py
"""

import pandas as pd
import numpy as np
import os
import json
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

print("Loading dataset...")
df = pd.read_csv('student_placement_synthetic.csv')
print(f"Dataset shape: {df.shape}")
print(f"Placement distribution:\n{df['placement_status'].value_counts()}\n")

df = df.drop(columns=['salary_package_lpa'])
df = df.dropna()

le_branch = LabelEncoder()
le_tier = LabelEncoder()
df['branch'] = le_branch.fit_transform(df['branch'])
df['college_tier'] = le_tier.fit_transform(df['college_tier'])

X = df.drop('placement_status', axis=1)
y = df['placement_status']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}\n")

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Decision Tree':       DecisionTreeClassifier(max_depth=10, random_state=42),
    'Random Forest':       RandomForestClassifier(n_estimators=100, random_state=42),
    'KNN':                 KNeighborsClassifier(n_neighbors=5),
}

accuracies = {}
class_metrics = {}
confusion_matrices_data = {}

for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = round(accuracy_score(y_test, preds) * 100, 2)
    accuracies[name] = acc
    print(f"  Accuracy: {acc}%")
    print(f"  Classification Report:\n{classification_report(y_test, preds)}")

    cm = confusion_matrix(y_test, preds)
    tn, fp, fn, tp = cm.ravel()
    not_placed_acc = round(tn / (tn + fp) * 100, 2) if (tn + fp) > 0 else 0.0
    placed_acc     = round(tp / (tp + fn) * 100, 2) if (tp + fn) > 0 else 0.0

    class_metrics[name] = {
        'placed_accuracy': placed_acc,
        'not_placed_accuracy': not_placed_acc,
    }
    confusion_matrices_data[name] = cm.tolist()
    print(f"  Placed Accuracy (recall): {placed_acc}%  |  Not Placed Accuracy (specificity): {not_placed_acc}%")

best_model = max(accuracies, key=accuracies.get)
print(f"\n✅ Best Model: {best_model} ({accuracies[best_model]}%)")

os.makedirs('ml_models', exist_ok=True)

with open('ml_models/models.pkl', 'wb') as f:
    pickle.dump(models, f)

with open('ml_models/encoders.pkl', 'wb') as f:
    pickle.dump({'branch': le_branch, 'college_tier': le_tier}, f)

with open('ml_models/results.json', 'w') as f:
    json.dump({
        'accuracies': accuracies,
        'best': best_model,
        'class_metrics': class_metrics,
        'confusion_matrices': confusion_matrices_data,
    }, f, indent=2)

print("\n✅ Models saved to ml_models/ folder.")
print("You can now run: python manage.py runserver")
