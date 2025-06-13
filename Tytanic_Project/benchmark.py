import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import VotingClassifier

# ===============================
# --- Benchmark na surowych danych ---
# ===============================

df = pd.read_csv('/Users/michalcukrowski/Projekt_Uczenie_Maszynowe/Tytanic_Project/train.csv')
# Wybór kolumn
df = df[['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]

# Proste czyszczenie
df['Age'] = df['Age'].fillna(df['Age'].median())

df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

df = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)

X_raw = df.drop('Survived', axis=1)
y_raw = df['Survived']

models_simple = {
    'LogReg': LogisticRegression(max_iter=1000),
    'KNN': KNeighborsClassifier(),
    'SVM': SVC(),
    'NaiveBayes': GaussianNB(),
    'DecisionTree': DecisionTreeClassifier()
}

print("\n🧪 Benchmark na surowych danych:\n")
benchmark_simple = {}

for name, model in models_simple.items():
    score = cross_val_score(model, X_raw, y_raw, cv=5, scoring='accuracy')
    benchmark_simple[name] = (score.mean(), score.std())

simple_df = pd.DataFrame(benchmark_simple, index=['Mean Accuracy', 'Std Dev']).T
print(simple_df.sort_values('Mean Accuracy', ascending=False))

# ===============================
# --- Pipeline z feature engineeringiem ---
# ===============================

df = pd.read_csv('/Users/michalcukrowski/Projekt_Uczenie_Maszynowe/Tytanic_Project/train.csv')

# Wypełnianie braków
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
df['Cabin'] = df['Cabin'].fillna('Unknown')
df['FamilySize'] = df['SibSp'] + df['Parch']
df['Title'] = df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
# Inżynieria cech
df['Title'] = df['Title'].replace(['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
df['Title'] = df['Title'].replace('Mlle', 'Miss')
df['Title'] = df['Title'].replace('Ms', 'Miss')
df['Title'] = df['Title'].replace('Mme', 'Mrs')

# Kodowanie
label_enc_cols = ['Sex', 'Embarked', 'Title']
for col in label_enc_cols:
    df[col] = LabelEncoder().fit_transform(df[col])

features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'FamilySize', 'Title']
X = df[features]
y = df['Survived']

# SMOTE
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X, y)

# Selektor cech
selector = SelectFromModel(RandomForestClassifier(n_estimators=100, random_state=42))
selector.fit(X_res, y_res)
X_selected = selector.transform(X_res)

# ===============================
# --- Benchmark na przetworzonych danych ---
# ===============================

models_full = {
    'LogReg': LogisticRegression(max_iter=1000),
    'KNN': KNeighborsClassifier(),
    'SVM': SVC(),
    'NaiveBayes': GaussianNB(),
    'DecisionTree': DecisionTreeClassifier()
}

print("\n🧪 Benchmark na danych po pełnym pipeline:\n")
benchmark_full = {}

for name, model in models_full.items():
    score = cross_val_score(model, X_selected, y_res, cv=5, scoring='accuracy')
    benchmark_full[name] = (score.mean(), score.std())

full_df = pd.DataFrame(benchmark_full, index=['Mean Accuracy', 'Std Dev']).T
print(full_df.sort_values('Mean Accuracy', ascending=False))

from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier

# Random Forest z GridSearchCV
param_grid_rf = {
    'n_estimators': [100, 200],
    'max_depth': [4, 6, 8],
    'min_samples_split': [2, 5],
}
grid_rf = GridSearchCV(RandomForestClassifier(random_state=42), param_grid_rf, cv=5, scoring='accuracy')
grid_rf.fit(X_selected, y_res)
rf_best = grid_rf.best_estimator_
rf_score = cross_val_score(rf_best, X_selected, y_res, cv=5, scoring='accuracy')

# Gradient Boosting
gb = GradientBoostingClassifier(random_state=42)
gb_score = cross_val_score(gb, X_selected, y_res, cv=5, scoring='accuracy')

# XGBoost (upewnij się, że masz zainstalowane `xgboost`)
xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
xgb_score = cross_val_score(xgb, X_selected, y_res, cv=5, scoring='accuracy')

# Voting Classifier (ensemble z LogReg, RF i XGB)
voting = VotingClassifier(estimators=[
    ('lr', LogisticRegression(max_iter=1000)),
    ('rf', rf_best),
    ('xgb', xgb)
], voting='soft')
voting_score = cross_val_score(voting, X_selected, y_res, cv=5, scoring='accuracy')

# Wyświetlanie wyników
print("\n🧪 Benchmark - Dodatkowe modele:\n")
print(f"Random Forest (GridSearch):  {rf_score.mean():.6f}  (std: {rf_score.std():.6f})")
print(f"Gradient Boosting:          {gb_score.mean():.6f}  (std: {gb_score.std():.6f})")
print(f"XGBoost:                     {xgb_score.mean():.6f}  (std: {xgb_score.std():.6f})")
print(f"Voting Classifier:           {voting_score.mean():.6f}  (std: {voting_score.std():.6f})")