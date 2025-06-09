import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import warnings 
warnings.filterwarnings('ignore')

# 1. Wczytanie danych
df = pd.read_csv('train.csv')
print("Rozmiar zbioru:", df.shape)
print(df.head())
print(df.info())
print(df.describe())

# 2. Wizualizacja braków danych
sns.heatmap(df.isnull(), cmap='coolwarm')
plt.title("Braki danych")
plt.show()

print("Liczba braków danych:\n", df.isnull().sum())

# 3. Inżynieria cech

# Uzupełnianie braków wieku medianą
df['Age'] = df['Age'].fillna(df['Age'].median())

# Uzupełnianie braków w 'Embarked' najczęstszą wartością
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

# Tworzenie nowej cechy FamilySize
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1

# Wyodrębnianie tytułów z nazwisk
df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)

# Grupowanie rzadkich tytułów
rare_titles = ['Lady', 'Countess','Capt','Col','Don', 'Dr', 'Major', 
               'Rev','Sir','Jonkheer','Dona']
df['Title'] = df['Title'].replace(rare_titles, 'Rare')

# Konsolidacja podobnych tytułów
df['Title'] = df['Title'].replace({'Mlle': 'Miss', 'Ms': 'Miss', 'Mme': 'Mrs'})

# 4. Kodowanie cech kategorycznych
label_enc = LabelEncoder()
df['Sex'] = label_enc.fit_transform(df['Sex'])
df['Embarked'] = label_enc.fit_transform(df['Embarked'])
df['Title'] = label_enc.fit_transform(df['Title'])

# 5. Usuwanie niepotrzebnych kolumn
df.drop(['Cabin', 'Ticket', 'Name', 'PassengerId'], axis=1, inplace=True)

# 6. Przygotowanie danych
X = df.drop('Survived', axis=1)
y = df['Survived']

# Skalowanie cech
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Podział na zbiór treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 7. Trenowanie kilku modeli
models = {
    'Logistic Regression': LogisticRegression(),
    'Random Forest': RandomForestClassifier(random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42)
}

results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    results[name] = acc
    print(f"{name} - Accuracy: {acc:.4f}")

# 8. Walidacja krzyżowa
print("\nWalidacja krzyżowa (5-krotna):")
for name, model in models.items():
    scores = cross_val_score(model, X_scaled, y, cv=5)
    print(f"{name} - Średnia dokładność (CV): {scores.mean():.4f}")

# 9. Optymalizacja hiperparametrów dla Random Forest
param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [4, 6, 8],
    'min_samples_split': [2, 5]
}

grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=3)
grid_search.fit(X_train, y_train)

print("\nNajlepsze parametry dla Random Forest:", grid_search.best_params_)
best_model = grid_search.best_estimator_

# 10. Wyświetlanie krzywej ROC i macierzy pomyłek
y_probs = best_model.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_probs)

plt.figure(figsize=(8,6))
plt.plot(fpr, tpr, label="Random Forest (best)")
plt.plot([0,1], [0,1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.grid()
plt.show()

# Macierz pomyłek
cm = confusion_matrix(y_test, best_model.predict(X_test))
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Macierz pomyłek - Random Forest")
plt.xlabel("Predykcja")
plt.ylabel("Prawdziwa klasa")
plt.show()

# 11. Zapis i wczytanie modelu
joblib.dump(best_model, 'best_titanic_model.pkl')
print("Model zapisany do 'best_titanic_model.pkl'")

loaded_model = joblib.load('best_titanic_model.pkl')
pred = loaded_model.predict(X_test)
print("Dokładność wczytanego modelu:", accuracy_score(y_test, pred))

# 12. Podsumowanie wyników
print("\nPodsumowanie dokładności modeli:")
for name, acc in results.items():
    print(f"{name}: {acc:.4f}")
