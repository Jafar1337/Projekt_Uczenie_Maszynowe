import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve, precision_recall_curve
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import shap
import joblib
import warnings
warnings.filterwarnings('ignore')
import os

# ----------- Funkcja preprocessing danych (z automatycznym tworzeniem interakcji) -------------
def preprocess_data(df, encoders=None, scaler=None, fit=True):
    df = df.copy()

    # 1. Braki danych i podstawowe cechy
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

    # 2. Inżynieria cech
    df['HasCabin'] = df['Cabin'].apply(lambda x: 0 if pd.isna(x) else 1)
    df['CabinLetter'] = df['Cabin'].apply(lambda x: str(x)[0] if pd.notna(x) else 'M')  # M = missing
    df['TicketPrefix'] = df['Ticket'].apply(lambda x: x.split()[0] if not x.isdigit() else 'None')
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
    df['Title'] = df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
    rare_titles = ['Lady', 'Countess','Capt','Col','Don', 'Dr', 'Major','Rev','Sir','Jonkheer','Dona']
    df['Title'] = df['Title'].replace(rare_titles, 'Rare')
    df['Title'] = df['Title'].replace({'Mlle': 'Miss', 'Ms': 'Miss', 'Mme': 'Mrs'})
    df['NameLength'] = df['Name'].apply(len)
    bins = [0, 12, 20, 40, 60, 120]
    labels = ['Child', 'Teen', 'Adult', 'Senior', 'Elder']
    df['AgeGroup'] = pd.cut(df['Age'], bins=bins, labels=labels)

    # --- Nowe cechy - interakcje ---
    # Uwaga: 'Sex' jest kategoryczne, więc mapujemy na liczby przed mnożeniem
    if 'Sex' in df.columns:
        sex_map = {'male': 0, 'female': 1}
        df['Sex_num'] = df['Sex'].map(sex_map)
    else:
        df['Sex_num'] = 0
    df['Age_x_Sex'] = df['Age'] * df['Sex_num']
    df['Fare_per_FamilySize'] = df['Fare'] / df['FamilySize']
    df['Age_x_Fare'] = df['Age'] * df['Fare']
    df['FamilySize_x_IsAlone'] = df['FamilySize'] * df['IsAlone']

    # Usuwamy niepotrzebne kolumny
    df.drop(['Ticket', 'Name', 'PassengerId', 'Cabin', 'Sex_num'], axis=1, inplace=True)

    # Kategoryczne do label encoding
    categorical_cols = ['Sex', 'Embarked', 'Title', 'CabinLetter', 'TicketPrefix', 'AgeGroup']

    if fit:
        encoders = {}
        for col in categorical_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            encoders[col] = le
    else:
        for col in categorical_cols:
            le = encoders[col]
            df[col] = df[col].map(lambda x: le.transform([x])[0] if x in le.classes_ else -1)

    # Skalowanie cech numerycznych
    numeric_cols = ['Age', 'FamilySize', 'NameLength', 'Fare', 'Age_x_Sex', 'Fare_per_FamilySize', 'Age_x_Fare', 'FamilySize_x_IsAlone']
    if fit:
        scaler = StandardScaler()
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    else:
        df[numeric_cols] = scaler.transform(df[numeric_cols])

    return df, encoders, scaler

# ----------- Funkcja selekcji cech -------------
def feature_selection(X, y, k=15):
    selector = SelectKBest(score_func=f_classif, k=k)
    X_new = selector.fit_transform(X, y)
    selected_features = X.columns[selector.get_support()]
    print(f"Wybrane cechy (top {k}): {list(selected_features)}")
    return pd.DataFrame(X_new, columns=selected_features), selector

# ----------- Wczytanie danych -------------
df = pd.read_csv('/Users/michalcukrowski/Projekt_Uczenie_Maszynowe/Tytanic_Project/train.csv')

print("Rozmiar zbioru:", df.shape)
print(df.head())
print(df.info())
print(df.describe())

# ----------- Wizualizacja braków danych -------------
sns.heatmap(df.isnull(), cmap='coolwarm')
plt.title("Braki danych")
plt.show()
print("Liczba braków danych:\n", df.isnull().sum())

# ----------- Preprocessing danych -------------
df_processed, encoders, scaler = preprocess_data(df, fit=True)

X = df_processed.drop('Survived', axis=1)
y = df_processed['Survived']

# ----------- Selekcja cech -------------
X_selected, selector = feature_selection(X, y, k=15)

# ----------- Balansowanie klas - SMOTE -------------
print(f"Przed SMOTE: klasa 1 = {sum(y==1)}, klasa 0 = {sum(y==0)}")
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X_selected, y)
print(f"Po SMOTE: klasa 1 = {sum(y_res==1)}, klasa 0 = {sum(y_res==0)}")

# ----------- Podział na zbiór treningowy i testowy -------------
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42, stratify=y_res)

# ----------- GridSearchCV dla Random Forest i XGBoost -------------
rf = RandomForestClassifier(random_state=42)
xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)

param_grid_rf = {
    'n_estimators': [100, 200],
    'max_depth': [5, 10, None],
    'min_samples_split': [2, 5],
}

param_grid_xgb = {
    'n_estimators': [100, 200],
    'max_depth': [3, 6, 10],
    'learning_rate': [0.01, 0.1],
}

print("Tuning Random Forest...")
grid_rf = GridSearchCV(rf, param_grid_rf, cv=3, scoring='accuracy', n_jobs=-1)
grid_rf.fit(X_train, y_train)
print("Best RF params:", grid_rf.best_params_)

print("Tuning XGBoost...")
grid_xgb = GridSearchCV(xgb, param_grid_xgb, cv=3, scoring='accuracy', n_jobs=-1)
grid_xgb.fit(X_train, y_train)
print("Best XGB params:", grid_xgb.best_params_)

# ----------- Modele do trenowania -------------
models = {
    'Logistic Regression': LogisticRegression(),
    'Random Forest': grid_rf.best_estimator_,
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42),
    'XGBoost': grid_xgb.best_estimator_
}

results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"{name} - Accuracy: {acc:.4f}")
    print(classification_report(y_test, preds))
    results[name] = acc

# ----------- Walidacja krzyżowa -------------
print("\nWalidacja krzyżowa (5-krotna):")
X_all = pd.concat([X_train, X_test])
y_all = pd.concat([y_train, y_test])

for name, model in models.items():
    scores = cross_val_score(model, X_all, y_all, cv=5)
    print(f"{name} - Średnia dokładność (CV): {scores.mean():.4f}")

# ----------- Voting Classifier -------------
voting_model = VotingClassifier(estimators=[
    ('lr', LogisticRegression()),
    ('rf', grid_rf.best_estimator_),
    ('xgb', grid_xgb.best_estimator_)
], voting='soft')
voting_model.fit(X_train, y_train)
y_probs = voting_model.predict_proba(X_test)[:, 1]

# ----------- ROC & Precision-Recall -------------
fpr, tpr, _ = roc_curve(y_test, y_probs)
plt.figure(figsize=(8,6))
plt.plot(fpr, tpr, label="Voting Classifier")
plt.plot([0,1], [0,1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.grid()
plt.show()

prec, rec, _ = precision_recall_curve(y_test, y_probs)
plt.plot(rec, prec)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.grid()
plt.show()

# ----------- Macierz pomyłek -------------
cm = confusion_matrix(y_test, voting_model.predict(X_test))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Macierz pomyłek - Voting Classifier")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()

# ----------- SHAP dla modelu Random Forest -------------
explainer = shap.TreeExplainer(grid_rf.best_estimator_)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test)

# ----------- Zapisanie modelu i obiektów preprocessingu -------------
os.makedirs('model', exist_ok=True)
joblib.dump(voting_model, 'model/voting_model.joblib')
joblib.dump(encoders, 'model/encoders.joblib')
joblib.dump(scaler, 'model/scaler.joblib')
joblib.dump(selector, 'model/selector.joblib')

# ----------- Funkcja do predykcji na nowych danych -------------
def predict_new(data_df):
    """
    Przykładowa funkcja do predykcji nowych danych:
    - Wczytuje encodery, scaler i selector
    - Preprocessuje dane
    - Wybiera cechy zgodne z wybranymi w selekcji
    - Zwraca predykcję
    """
    encoders = joblib.load('model/encoders.joblib')
    scaler = joblib.load('model/scaler.joblib')
    selector = joblib.load('model/selector.joblib')
    model = joblib.load('model/voting_model.joblib')

    df_proc, _, _ = preprocess_data(data_df, encoders=encoders, scaler=scaler, fit=False)

    # Wybieramy tylko cechy wybrane przez selector
    df_selected = df_proc[selector.get_feature_names_out()]

    preds = model.predict(df_selected)
    return preds

