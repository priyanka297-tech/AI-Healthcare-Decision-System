# PATIENT SURVIVAL PREDICTION

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split

# ==========================================================
# DISPLAY SETTINGS
# ==========================================================

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

print("="*70)
print("PATIENT SURVIVAL PREDICTION")
print("="*70)

# ==========================================================
# LOAD DATASET
# ==========================================================

df = pd.read_csv("support2_dataset.csv")

print("\nDataset Loaded Successfully.")

# ==========================================================
# BASIC INFORMATION
# ==========================================================

print("\nDataset Shape")

print(df.shape)

print("\nFirst 5 Rows")

print(df.head())

print("\nLast 5 Rows")

print(df.tail())

print("\nColumn Names")

print(df.columns.tolist())

print("\nData Types")

print(df.dtypes)

print("\nDataset Information")

df.info()

print("\nStatistical Summary")

print(df.describe(include="all").transpose())

# ==========================================================
# MISSING VALUES
# ==========================================================

missing = pd.DataFrame({

    "Missing Values": df.isnull().sum(),

    "Percentage": round(
        df.isnull().mean()*100,
        2
    )

})

missing = missing[missing["Missing Values"] > 0]

missing = missing.sort_values(

    by="Percentage",

    ascending=False

)

print("\nMissing Values")

print(missing)

# ==========================================================
# DUPLICATES
# ==========================================================

duplicates = df.duplicated().sum()

print("\nDuplicate Rows :", duplicates)

# ==========================================================
# PART 2
# Target Selection
# ==========================================================

print("="*70)
print("TARGET COLUMN")
print("="*70)

# Our target
target = "death"

print(df[target].value_counts())

print("\nPercentage")

print(df[target].value_counts(normalize=True)*100)

# ==========================================================
# Remove Leakage Columns
# ==========================================================

drop_columns = [

    "surv2m",
    "surv6m",
    "prg2m",
    "prg6m",
    "dnrday",
    "totmcst",
    "totcst",
    "charges"

]

# Drop only if present

drop_columns = [

    col for col in drop_columns

    if col in df.columns

]

df.drop(

    columns=drop_columns,

    inplace=True

)

print("\nDropped Columns")

print(drop_columns)

print("\nRemaining Shape")

print(df.shape)

# ==========================================================
# Features and Target
# ==========================================================

X = df.drop("death", axis=1)

y = df["death"]

print("\nFeatures Shape :", X.shape)

print("Target Shape :", y.shape)

# ==========================================================
# PART 3
# Missing Value Handling
# ==========================================================

print("="*70)
print("MISSING VALUE HANDLING")
print("="*70)

# Separate numerical and categorical columns

numeric_cols = X.select_dtypes(
    include=["int64", "float64"]
).columns

categorical_cols = X.select_dtypes(
    include=["object", "category", "bool"]
).columns

print("\nNumerical Columns :", len(numeric_cols))
print("Categorical Columns :", len(categorical_cols))

print("Categorical Columns")
print(categorical_cols)

print()

print("Numerical Columns")
print(numeric_cols)

# Fill missing values

for col in numeric_cols:
    X[col] = X[col].fillna(X[col].median())

for col in categorical_cols:
    X[col] = X[col].fillna(X[col].mode()[0])

print("\nRemaining Missing Values")

print(X.isnull().sum().sum())

# ==========================================================
# PART 4
# EXPLORATORY DATA ANALYSIS (EDA)
# ==========================================================

print("="*70)
print("EXPLORATORY DATA ANALYSIS")
print("="*70)

import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")

# ==========================================================
# Target Variable Distribution
# ==========================================================

plt.figure(figsize=(6,5))

sns.countplot(
    x=df["death"],
    palette="viridis"
)

plt.title("Target Variable Distribution")
plt.xlabel("Death")
plt.ylabel("Count")

plt.show()

# ==========================================================
# Age Distribution
# ==========================================================

plt.figure(figsize=(8,5))

sns.histplot(
    df["age"],
    bins=30,
    kde=True,
    color="royalblue"
)

plt.title("Age Distribution")

plt.show()

# ==========================================================
# Gender Distribution
# ==========================================================

plt.figure(figsize=(6,5))

sns.countplot(
    x=df["sex"],
    palette="Set2"
)

plt.title("Gender Distribution")

plt.show()

# ==========================================================
# Disease Group Distribution
# ==========================================================

plt.figure(figsize=(10,5))

sns.countplot(
    x=df["dzgroup"],
    order=df["dzgroup"].value_counts().index,
    palette="tab20"
)

plt.xticks(rotation=45)

plt.title("Disease Group Distribution")

plt.show()

# ==========================================================
# Numerical Features Distribution
# ==========================================================

numerical_features = [

    "age",
    "hrt",
    "meanbp",
    "resp",
    "temp",
    "wblc",
    "glucose",
    "crea"

]

for feature in numerical_features:

    if feature in df.columns:

        plt.figure(figsize=(7,4))

        sns.histplot(

            df[feature],

            kde=True,

            color="steelblue"

        )

        plt.title(f"{feature} Distribution")

        plt.show()

# ==========================================================
# Boxplots
# ==========================================================

for feature in numerical_features:

    if feature in df.columns:

        plt.figure(figsize=(7,4))

        sns.boxplot(

            x=df[feature],

            color="orange"

        )

        plt.title(f"{feature} Boxplot")

        plt.show()

# ==========================================================
# Correlation Heatmap
# ==========================================================

numeric_df = df.select_dtypes(include=["number"])

corr = numeric_df.corr()

plt.figure(figsize=(15,10))

sns.heatmap(
    corr,
    cmap="coolwarm",
    linewidths=0.3
)

plt.title("Correlation Heatmap")

plt.show()

# ==========================================================
# Survival vs Important Features
# ==========================================================

important_features = [

    "age",
    "sfdm2",
    "hrt",
    "meanbp",
    "glucose"

]

for feature in important_features:

    if feature in df.columns:

        plt.figure(figsize=(7,5))

        sns.boxplot(

            x=df["death"],

            y=df[feature]

        )

        plt.title(f"Death vs {feature}")

        plt.show()        

# ==========================================================
# Pair Plot
# ==========================================================

pair_features = [

    "age",
    "hrt",
    "meanbp",
    "temp",
    "death"

]

pair_features = [

    col for col in pair_features

    if col in df.columns

]

sns.pairplot(

    df[pair_features],

    hue="death"

)

plt.show()

# ==========================================================
# Missing Value Heatmap
# ==========================================================

plt.figure(figsize=(14,6))

sns.heatmap(

    df.isnull(),

    cbar=False,

    cmap="viridis"

)

plt.title("Missing Values Heatmap")

plt.show()

# ==========================================================
# Encode Categorical Variables
# ==========================================================

from sklearn.preprocessing import LabelEncoder

encoders = {}

for col in categorical_cols:

    le = LabelEncoder()

    X[col] = le.fit_transform(
        X[col].astype(str)
    )

    encoders[col] = le

print("\nEncoding Completed.")

print("\nDataset Shape")

print(X.shape)

# ==========================================================
# Train Test Split
# ==========================================================

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(

    X,

    y,

    test_size=0.20,

    random_state=42,

    stratify=y

)

print("\nTraining Shape")

print(X_train.shape)

print("\nTesting Shape")

print(X_test.shape)

# ==========================================================
# PART 4
# Model Training
# ==========================================================

print("="*70)
print("MODEL TRAINING")
print("="*70)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
    HistGradientBoostingClassifier
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

models = {

    "Logistic Regression": LogisticRegression(
        max_iter=500,
        class_weight="balanced",
        random_state=42
    ),

    "Decision Tree": DecisionTreeClassifier(
        random_state=42,
        class_weight="balanced"
    ),

    "Random Forest": RandomForestClassifier(
        random_state=42,
        class_weight="balanced"
    ),

    "Extra Trees": ExtraTreesClassifier(
        random_state=42,
        class_weight="balanced"
    ),

    "Gradient Boosting": GradientBoostingClassifier(
        random_state=42
    ),

    "AdaBoost": AdaBoostClassifier(
        random_state=42
    ),

    "Hist Gradient Boosting": HistGradientBoostingClassifier(
        random_state=42
    )

}

results = []

trained_models = {}

for name, model in models.items():

    print(f"Training {name}...")

    model.fit(X_train, y_train)

    trained_models[name] = model

    pred = model.predict(X_test)

    acc = accuracy_score(y_test, pred)

    precision = precision_score(y_test, pred)

    recall = recall_score(y_test, pred)

    f1 = f1_score(y_test, pred)

    results.append([

        name,

        acc,

        precision,

        recall,

        f1

    ])

print("\nTraining Completed!")

# ==========================================================
# Results
# ==========================================================

result_df = pd.DataFrame(

    results,

    columns=[

        "Model",

        "Accuracy",

        "Precision",

        "Recall",

        "F1 Score"

    ]

)

result_df = result_df.sort_values(

    by="Accuracy",

    ascending=False

)

print(result_df)

# ==========================================================
# Best Model
# ==========================================================

best_model_name = result_df.iloc[0]["Model"]

best_model = trained_models[best_model_name]

print("\nBest Model :", best_model_name)

# ==========================================================
# PART 5
# FEATURE IMPORTANCE
# ==========================================================

print("="*70)
print("FEATURE IMPORTANCE")
print("="*70)

from sklearn.ensemble import ExtraTreesClassifier

feature_model = ExtraTreesClassifier(
    n_estimators=300,
    random_state=42
)

feature_model.fit(X_train, y_train)

importance_df = pd.DataFrame({

    "Feature": X_train.columns,
    "Importance": feature_model.feature_importances_

})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

print("\nTop 20 Important Features\n")

print(importance_df.head(20))

# ==========================================================
# Feature Importance Plot
# ==========================================================

plt.figure(figsize=(10,8))

sns.barplot(

    data=importance_df.head(20),

    x="Importance",

    y="Feature"

)

plt.title("Top 20 Important Features")

plt.tight_layout()

plt.show()

# ==========================================================
# PART 6
# FEATURE CORRELATION
# ==========================================================

print("="*70)
print("FEATURE CORRELATION WITH TARGET")
print("="*70)

corr_df = X.copy()

corr_df["death"] = y

correlation = corr_df.corr(numeric_only=True)["death"]

correlation = correlation.sort_values(ascending=False)

print(correlation)

# ==========================================================
# PART 7
# HIGHLY CORRELATED FEATURES
# ==========================================================

corr_matrix = X.corr().abs()

upper = corr_matrix.where(

    np.triu(

        np.ones(corr_matrix.shape),

        k=1

    ).astype(bool)

)

high_corr = [

    column

    for column in upper.columns

    if any(upper[column] > 0.95)

]

print("="*70)
print("HIGHLY CORRELATED FEATURES")
print("="*70)

print(high_corr)

# ==========================================================
# PART 8
# HYPERPARAMETER TUNING
# ==========================================================

print("="*70)
print("HYPERPARAMETER TUNING")
print("="*70)

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier

gb = GradientBoostingClassifier(
    random_state=42
)

param_grid = {

    "n_estimators": [100, 200],

    "learning_rate": [0.05, 0.1],

    "max_depth": [3, 5],

    "subsample": [0.8, 1.0]

}

grid = GridSearchCV(

    estimator=gb,

    param_grid=param_grid,

    cv=5,

    scoring="accuracy",

    n_jobs=-1,

    verbose=2

)

grid.fit(X_train, y_train)

best_model = grid.best_estimator_

print("\nBest Parameters")

print(grid.best_params_)

print("\nBest Cross Validation Accuracy")

print(grid.best_score_)

# ==========================================================
# PART 9
# MODEL EVALUATION
# ==========================================================

from sklearn.metrics import (

    accuracy_score,

    precision_score,

    recall_score,

    f1_score,

    confusion_matrix,

    classification_report

)

prediction = best_model.predict(X_test)

accuracy = accuracy_score(y_test, prediction)

precision = precision_score(y_test, prediction)

recall = recall_score(y_test, prediction)

f1 = f1_score(y_test, prediction)

print("="*70)
print("FINAL MODEL PERFORMANCE")
print("="*70)

print("Accuracy :", round(accuracy,4))

print("Precision :", round(precision,4))

print("Recall :", round(recall,4))

print("F1 Score :", round(f1,4))

print("\nClassification Report\n")

print(classification_report(y_test, prediction))

# ==========================================================
# CONFUSION MATRIX
# ==========================================================

cm = confusion_matrix(

    y_test,

    prediction

)

plt.figure(figsize=(6,5))

sns.heatmap(

    cm,

    annot=True,

    fmt="d",

    cmap="Blues"

)

plt.xlabel("Predicted")

plt.ylabel("Actual")

plt.title("Confusion Matrix")

plt.show()

# ==========================================================
# ROC CURVE
# ==========================================================

from sklearn.metrics import roc_curve, auc

probability = best_model.predict_proba(X_test)[:,1]

fpr, tpr, threshold = roc_curve(

    y_test,

    probability

)

roc_auc = auc(

    fpr,

    tpr

)

plt.figure(figsize=(7,5))

plt.plot(

    fpr,

    tpr,

    label=f"AUC = {roc_auc:.3f}"

)

plt.plot([0,1],[0,1],"--")

plt.xlabel("False Positive Rate")

plt.ylabel("True Positive Rate")

plt.title("ROC Curve")

plt.legend()

plt.show()

print("ROC AUC Score :", roc_auc)

# ==========================================================
# CROSS VALIDATION
# ==========================================================

from sklearn.model_selection import cross_val_score

scores = cross_val_score(

    best_model,

    X,

    y,

    cv=5,

    scoring="accuracy"

)

print("="*70)
print("CROSS VALIDATION")
print("="*70)

print(scores)

print("\nAverage Accuracy :", scores.mean())

# ==========================================================
# SAVE MODEL AND PREPROCESSING OBJECTS
# ==========================================================

import pickle

# Save trained model
with open("patient_survival_model.pkl", "wb") as file:
    pickle.dump(best_model, file)

# Save Label Encoders
with open("label_encoders.pkl", "wb") as file:
    pickle.dump(encoders, file)

# Save Feature Column Order
with open("feature_columns.pkl", "wb") as file:
    pickle.dump(list(X.columns), file)

print("="*70)
print("MODEL SAVED SUCCESSFULLY")
print("LABEL ENCODERS SAVED")
print("FEATURE COLUMNS SAVED")
print("="*70)

import pickle

with open("feature_columns.pkl", "rb") as f:
    feature_columns = pickle.load(f)

print(feature_columns)

import pickle

with open("label_encoders.pkl", "rb") as f:
    encoders = pickle.load(f)

for col, encoder in encoders.items():
    print(f"{col} : {list(encoder.classes_)}")

print(df[["death", "hospdead"]].head(20))
print(df[["death", "hospdead"]].value_counts())    