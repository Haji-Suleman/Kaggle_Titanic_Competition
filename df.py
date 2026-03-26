# -*- coding: utf-8 -*-
"""
Titanic Survival Prediction - Complete Pipeline
Using CatBoost with Feature Engineering, Hyperparameter Tuning, and Ensemble Methods
"""

import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")

from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from category_encoders import TargetEncoder, WOEEncoder
import optuna
from lightgbm import LGBMClassifier

print("=" * 60)
print("TITANIC SURVIVAL PREDICTION PIPELINE - ENHANCED VERSION")
print("=" * 60)

# ============================================================================
# 1. DATA LOADING AND ADVANCED FEATURE ENGINEERING
# ============================================================================
print("\n[1/5] Loading data and engineering advanced features...")

# Load data
df = pd.read_csv("./train.csv")
test = pd.read_csv("./test.csv")

# Create a combined dataset for consistent feature engineering
combined = pd.concat([df, test], sort=False)
combined["IsTrain"] = combined["Survived"].notna()

# 1. Advanced Title extraction with more categories
combined["Title"] = combined["Name"].str.extract(" ([A-Za-z]+)\.", expand=False)
combined["Title"] = combined["Title"].replace(
    [
        "Lady",
        "Countess",
        "Capt",
        "Col",
        "Don",
        "Dr",
        "Major",
        "Rev",
        "Sir",
        "Jonkheer",
        "Dona",
    ],
    "Rare",
)
combined["Title"] = combined["Title"].replace(["Mlle", "Ms"], "Miss")
combined["Title"] = combined["Title"].replace("Mme", "Mrs")

# Title frequency encoding
title_freq = combined.groupby("Title")["IsTrain"].transform("count")
combined["TitleFreq"] = title_freq

# 2. Family features - Enhanced
combined["FamilySize"] = combined["SibSp"] + combined["Parch"] + 1
combined["IsAlone"] = (combined["FamilySize"] == 1).astype(int)

# Family survival rate (using training data only)
family_survival = (
    combined[combined["IsTrain"]].groupby("FamilySize")["Survived"].mean().to_dict()
)
combined["FamilySurvivalRate"] = combined["FamilySize"].map(family_survival).fillna(0.5)

# Family size categories with more granularity
combined["FamilySizeCategory"] = pd.cut(
    combined["FamilySize"],
    bins=[0, 1, 2, 3, 4, 5, 20],
    labels=["Alone", "Couple", "Small", "Medium", "Large", "VeryLarge"],
)

# 3. Cabin features - Enhanced
combined["HasCabin"] = combined["Cabin"].notna().astype(int)
combined["CabinLetter"] = combined["Cabin"].str[0].fillna("U")
# Group rare cabin letters into 'Other'
rare_letters = combined[combined["IsTrain"]]["CabinLetter"].value_counts()
rare_letters = rare_letters[rare_letters < 5].index.tolist()
combined["CabinLetter"] = combined["CabinLetter"].apply(
    lambda x: "Other" if x in rare_letters else x
)
combined["CabinNum"] = (
    combined["Cabin"].str.extract("(\d+)", expand=False).astype(float)
)
combined["CabinNum"] = combined["CabinNum"].fillna(0)

# 4. Age features - Enhanced
# Age imputation with more granular groups
combined["Age"] = combined.groupby(["Title", "Pclass", "Sex"])["Age"].transform(
    lambda x: x.fillna(x.median())
)
combined["Age"] = combined["Age"].fillna(combined["Age"].median())

combined["AgeGroup"] = pd.cut(
    combined["Age"],
    bins=[0, 5, 12, 18, 25, 35, 50, 65, 100],
    labels=[
        "Infant",
        "Child",
        "Teen",
        "YoungAdult",
        "Adult",
        "Middle",
        "Senior",
        "Elder",
    ],
)

# Age bin encoding
age_bins = [0, 12, 18, 35, 50, 80]
age_labels = [0, 1, 2, 3, 4]
combined["AgeBin"] = pd.cut(combined["Age"], bins=age_bins, labels=age_labels).astype(
    float
)

# 5. Fare features - Enhanced
combined["Fare"] = combined.groupby(["Pclass", "Embarked"])["Fare"].transform(
    lambda x: x.fillna(x.median())
)
combined["Fare"] = combined["Fare"].fillna(combined["Fare"].median())

combined["FarePerPerson"] = combined["Fare"] / combined["FamilySize"]
combined["FarePerPerson"] = combined["FarePerPerson"].fillna(combined["Fare"])

# Fare categories
combined["FareGroup"] = pd.qcut(
    combined["Fare"],
    q=5,
    labels=["VeryLow", "Low", "Medium", "High", "VeryHigh"],
    duplicates="drop",
)

# Log transform for fare
combined["FareLog"] = np.log1p(combined["Fare"])

# 6. Ticket features - Enhanced
combined["TicketPrefix"] = (
    combined["Ticket"].str.extract("([A-Za-z]+)\.?", expand=False).fillna("NUM")
)
# Group rare ticket prefixes
rare_prefixes = combined[combined["IsTrain"]]["TicketPrefix"].value_counts()
rare_prefixes = rare_prefixes[rare_prefixes < 5].index.tolist()
combined["TicketPrefix"] = combined["TicketPrefix"].apply(
    lambda x: "Other" if x in rare_prefixes else x
)

combined["TicketLength"] = combined["Ticket"].str.len()
combined["TicketNumber"] = (
    combined["Ticket"].str.extract("(\d+)", expand=False).astype(float)
)
combined["TicketNumber"] = combined["TicketNumber"].fillna(0)

# Ticket prefix frequency
prefix_freq = combined.groupby("TicketPrefix")["IsTrain"].transform("count")
combined["TicketPrefixFreq"] = prefix_freq

# 7. Interaction features - Enhanced
combined["Age*Pclass"] = combined["Age"] * combined["Pclass"]
combined["Age*Fare"] = combined["Age"] * combined["FareLog"]
combined["Sex_Pclass"] = combined["Sex"] + "_" + combined["Pclass"].astype(str)
combined["Sex_AgeGroup"] = combined["Sex"] + "_" + combined["AgeGroup"].astype(str)
combined["Pclass_FareGroup"] = (
    combined["Pclass"].astype(str) + "_" + combined["FareGroup"].astype(str)
)

# 8. Embarked features
combined["Embarked"] = combined["Embarked"].fillna(combined["Embarked"].mode()[0])
combined["IsSouthampton"] = (combined["Embarked"] == "S").astype(int)
combined["IsCherbourg"] = (combined["Embarked"] == "C").astype(int)
combined["IsQueenstown"] = (combined["Embarked"] == "Q").astype(int)

# 9. Pclass features
combined["IsFirstClass"] = (combined["Pclass"] == 1).astype(int)
combined["IsSecondClass"] = (combined["Pclass"] == 2).astype(int)
combined["IsThirdClass"] = (combined["Pclass"] == 3).astype(int)

# 10. Sex features with encoding
combined["IsFemale"] = (combined["Sex"] == "female").astype(int)
combined["IsMale"] = (combined["Sex"] == "male").astype(int)

# 11. Missing value indicators
combined["AgeMissing"] = combined["Age"].isna().astype(int)
combined["FareMissing"] = combined["Fare"].isna().astype(int)
combined["CabinMissing"] = combined["Cabin"].isna().astype(int)

# Fill any remaining missing values
combined["Age"] = combined["Age"].fillna(combined["Age"].median())
combined["Fare"] = combined["Fare"].fillna(combined["Fare"].median())

# Drop unnecessary columns
drop_cols = ["Name", "Ticket", "Cabin", "PassengerId"]
if "Survived" in combined.columns:
    drop_cols.append("Survived")

# Split back
df_train = combined[combined["IsTrain"]].copy()
df_test = combined[~combined["IsTrain"]].copy()

# Prepare features
X = df_train.drop(columns=["Survived", "IsTrain"] + drop_cols)
y = df_train["Survived"]

# Define categorical features
cat_features = [
    "Sex",
    "Embarked",
    "Title",
    "FamilySizeCategory",
    "CabinLetter",
    "AgeGroup",
    "FareGroup",
    "TicketPrefix",
    "Sex_Pclass",
    "Sex_AgeGroup",
    "Pclass_FareGroup",
]

print(f"✅ Training data shape: {X.shape}")
print(f"✅ Test data shape: {df_test.shape}")
print(f"✅ Total features: {len(X.columns)}")
print(f"✅ Categorical features: {len(cat_features)}")

# ============================================================================
# 2. HYPERPARAMETER TUNING WITH OPTUNA
# ============================================================================
print("\n[2/5] Hyperparameter tuning with Optuna (CatBoost only)...")


def objective(trial):
    params = {
        "iterations": trial.suggest_int("iterations", 500, 1200),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "depth": trial.suggest_int("depth", 4, 10),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1, 15),
        "random_strength": trial.suggest_float("random_strength", 0.1, 8),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0, 1),
        "border_count": trial.suggest_int("border_count", 32, 255),
        "subsample": trial.suggest_float("subsample", 0.5, 1),
        "loss_function": "Logloss",
        "eval_metric": "Accuracy",
        "random_seed": 42,
        "verbose": 0,
        "early_stopping_rounds": 50,
    }

    # Cross-validation
    cv_scores = []
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for train_idx, val_idx in skf.split(X, y):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = CatBoostClassifier(**params)
        model.fit(
            X_train,
            y_train,
            cat_features=cat_features,
            eval_set=(X_val, y_val),
            verbose=0,
        )

        preds = model.predict(X_val)
        cv_scores.append(accuracy_score(y_val, preds))

    return np.mean(cv_scores)


# Run optimization
print("   Running Optuna optimization (20 trials)...")
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=20, timeout=1800)

print("\n✅ Best CatBoost parameters found:")
for key, value in study.best_params.items():
    print(f"   {key}: {value}")
print(f"✅ Best CV score: {study.best_value:.4f}")

# Train CatBoost with best parameters
catboost_params = study.best_params.copy()
catboost_params.update(
    {
        "loss_function": "Logloss",
        "eval_metric": "Accuracy",
        "random_seed": 42,
        "verbose": 100,
    }
)

best_catboost = CatBoostClassifier(**catboost_params)
best_catboost.fit(X, y, cat_features=cat_features)

# ============================================================================
# 3. LIGHTGBM MODEL WITH PROPER ENCODING
# ============================================================================
print("\n[3/5] Training LightGBM model...")


# Function to safely encode categorical features
def safe_encode(train, test, columns):
    """Encode categorical features handling unseen categories"""
    encoded_train = train.copy()
    encoded_test = test.copy()

    for col in columns:
        # Create a mapping from training categories
        unique_train = train[col].unique()

        # Encode training data
        le = LabelEncoder()
        encoded_train[col] = le.fit_transform(train[col].astype(str))

        # Encode test data, setting unseen categories to -1
        test_values = test[col].astype(str)
        encoded_test[col] = test_values.map(
            lambda x: le.transform([x])[0] if x in le.classes_ else -1
        )

    return encoded_train, encoded_test


# Prepare data for LightGBM
X_lgbm_train = X.copy()
X_lgbm_test = df_test.drop(columns=["IsTrain"] + drop_cols).copy()

# Apply safe encoding
X_lgbm_train, X_lgbm_test = safe_encode(X_lgbm_train, X_lgbm_test, cat_features)

print("   Running LightGBM optimization (15 trials)...")


def objective_lgbm(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 300, 1000),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "num_leaves": trial.suggest_int("num_leaves", 20, 150),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.01, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.01, 10.0, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
        "random_state": 42,
        "verbose": -1,
        "boosting_type": "gbdt",
    }

    cv_scores = []
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for train_idx, val_idx in skf.split(X_lgbm_train, y):
        X_train, X_val = X_lgbm_train.iloc[train_idx], X_lgbm_train.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = LGBMClassifier(**params)
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            eval_metric="binary_logloss",
            callbacks=[
                LGBMClassifier.early_stopping(50),
                LGBMClassifier.log_evaluation(0),
            ],
        )

        preds = model.predict(X_val)
        cv_scores.append(accuracy_score(y_val, preds))

    return np.mean(cv_scores)


study_lgbm = optuna.create_study(direction="maximize")
study_lgbm.optimize(objective_lgbm, n_trials=15, timeout=1200)

print("\n✅ Best LightGBM parameters found:")
for key, value in study_lgbm.best_params.items():
    print(f"   {key}: {value}")
print(f"✅ Best CV score: {study_lgbm.best_value:.4f}")

# Train LightGBM with best parameters
lgbm_params = study_lgbm.best_params.copy()
lgbm_params.update({"random_state": 42, "verbose": -1, "boosting_type": "gbdt"})

best_lgbm = LGBMClassifier(**lgbm_params)
best_lgbm.fit(X_lgbm_train, y)

# ============================================================================
# 4. ENSEMBLE METHODS
# ============================================================================
print("\n[4/5] Creating ensemble models...")

# Prepare data for ensemble (encode categorical features with one-hot)
X_encoded = pd.get_dummies(X, columns=cat_features, drop_first=True)
X_test_encoded = pd.get_dummies(
    df_test.drop(columns=["IsTrain"] + drop_cols), columns=cat_features, drop_first=True
)

# Align columns
X_test_encoded = X_test_encoded.reindex(columns=X_encoded.columns, fill_value=0)

# Create individual models
model_catboost = CatBoostClassifier(**catboost_params, verbose=0)
model_lgbm_final = LGBMClassifier(**lgbm_params, verbose=-1)

# Voting Classifier (soft voting)
print("   Training Voting Classifier...")
voting_clf = VotingClassifier(
    estimators=[("catboost", model_catboost), ("lgbm", model_lgbm_final)],
    voting="soft",
    weights=[0.6, 0.4],  # Give more weight to CatBoost
)
voting_clf.fit(X_encoded, y)

# Stacking Classifier
print("   Training Stacking Classifier...")
stacking_clf = StackingClassifier(
    estimators=[("catboost", model_catboost), ("lgbm", model_lgbm_final)],
    final_estimator=LogisticRegression(C=1.0, max_iter=1000),
    cv=5,
    passthrough=True,
)
stacking_clf.fit(X_encoded, y)

# ============================================================================
# 5. ADVANCED TECHNIQUES
# ============================================================================
print("\n[5/5] Applying advanced techniques...")

# 1. Target encoding for high-cardinality features
print("   Applying target encoding...")
te = TargetEncoder()
X_te = X.copy()
X_test_te = df_test.drop(columns=["IsTrain"] + drop_cols).copy()

for col in ["TicketPrefix", "Title", "CabinLetter"]:
    X_te[f"{col}_te"] = te.fit_transform(X_te[col], y)
    X_test_te[f"{col}_te"] = te.transform(X_test_te[col])

# 2. Create additional interaction features
X_interaction = X.copy()
X_test_interaction = df_test.drop(columns=["IsTrain"] + drop_cols).copy()

# Polynomial features
X_interaction["Age_Squared"] = X_interaction["Age"] ** 2
X_test_interaction["Age_Squared"] = X_test_interaction["Age"] ** 2

X_interaction["Fare_Squared"] = X_interaction["Fare"] ** 2
X_test_interaction["Fare_Squared"] = X_test_interaction["Fare"] ** 2

# Ratio features
X_interaction["Age_Fare_Ratio"] = X_interaction["Age"] / (X_interaction["Fare"] + 1)
X_test_interaction["Age_Fare_Ratio"] = X_test_interaction["Age"] / (
    X_test_interaction["Fare"] + 1
)

# 3. Probability calibration
print("   Calibrating model probabilities...")
calibrated_model = CalibratedClassifierCV(best_catboost, method="isotonic", cv=5)
calibrated_model.fit(X_encoded, y)

# 4. Find optimal threshold
print("   Finding optimal threshold...")
thresholds = np.arange(0.2, 0.8, 0.01)
best_threshold = 0.5
best_f1 = 0

probs = best_catboost.predict_proba(X_encoded)[:, 1]
for threshold in thresholds:
    preds_threshold = (probs > threshold).astype(int)
    f1 = f1_score(y, preds_threshold)
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold

print(f"✅ Optimal threshold: {best_threshold:.3f}")

# Use optimal threshold for final predictions
probs_test = best_catboost.predict_proba(X_test_encoded)[:, 1]
preds_optimized = (probs_test > best_threshold).astype(int)

# ============================================================================
# 6. CREATE SUBMISSIONS
# ============================================================================
print("\n[6/6] Creating submission files...")

# Method 1: CatBoost with optimal threshold
submission_catboost_opt = pd.DataFrame(
    {"PassengerId": df_test["PassengerId"], "Survived": preds_optimized}
)
submission_catboost_opt.to_csv("submission_catboost_optimized.csv", index=False)
print("✅ submission_catboost_optimized.csv saved")

# Method 2: CatBoost (standard threshold)
preds_catboost = best_catboost.predict(X_test_encoded)
submission_catboost = pd.DataFrame(
    {"PassengerId": df_test["PassengerId"], "Survived": preds_catboost}
)
submission_catboost.to_csv("submission_catboost.csv", index=False)
print("✅ submission_catboost.csv saved")

# Method 3: LightGBM
preds_lgbm = best_lgbm.predict(X_lgbm_test)
submission_lgbm = pd.DataFrame(
    {"PassengerId": df_test["PassengerId"], "Survived": preds_lgbm}
)
submission_lgbm.to_csv("submission_lgbm.csv", index=False)
print("✅ submission_lgbm.csv saved")

# Method 4: Voting ensemble
preds_voting = voting_clf.predict(X_test_encoded)
submission_voting = pd.DataFrame(
    {"PassengerId": df_test["PassengerId"], "Survived": preds_voting}
)
submission_voting.to_csv("submission_voting.csv", index=False)
print("✅ submission_voting.csv saved")

# Method 5: Stacking ensemble
preds_stacking = stacking_clf.predict(X_test_encoded)
submission_stacking = pd.DataFrame(
    {"PassengerId": df_test["PassengerId"], "Survived": preds_stacking}
)
submission_stacking.to_csv("submission_stacking.csv", index=False)
print("✅ submission_stacking.csv saved")

# Method 6: Calibrated model
preds_calibrated = calibrated_model.predict(X_test_encoded)
submission_calibrated = pd.DataFrame(
    {"PassengerId": df_test["PassengerId"], "Survived": preds_calibrated}
)
submission_calibrated.to_csv("submission_calibrated.csv", index=False)
print("✅ submission_calibrated.csv saved")

# ============================================================================
# 7. PERFORMANCE SUMMARY
# ============================================================================
print("\n" + "=" * 60)
print("PERFORMANCE SUMMARY")
print("=" * 60)

# Calculate cross-validation scores
print("\n📊 Model Performance (5-fold CV):")
print("-" * 40)

# CatBoost CV score
catboost_cv = cross_val_score(best_catboost, X, y, cv=5, scoring="accuracy")
print(f"CatBoost CV Accuracy: {catboost_cv.mean():.4f} (+/- {catboost_cv.std():.4f})")

# LightGBM CV score
lgbm_cv = cross_val_score(best_lgbm, X_lgbm_train, y, cv=5, scoring="accuracy")
print(f"LightGBM CV Accuracy: {lgbm_cv.mean():.4f} (+/- {lgbm_cv.std():.4f})")

# Voting ensemble CV score
voting_cv = cross_val_score(voting_clf, X_encoded, y, cv=5, scoring="accuracy")
print(
    f"Voting Ensemble CV Accuracy: {voting_cv.mean():.4f} (+/- {voting_cv.std():.4f})"
)

# Stacking ensemble CV score
stacking_cv = cross_val_score(stacking_clf, X_encoded, y, cv=5, scoring="accuracy")
print(
    f"Stacking Ensemble CV Accuracy: {stacking_cv.mean():.4f} (+/- {stacking_cv.std():.4f})"
)

# Feature importance for CatBoost
print("\n📊 Top 15 Most Important Features (CatBoost):")
print("-" * 40)
feature_importance = (
    pd.DataFrame(
        {"feature": X.columns, "importance": best_catboost.feature_importances_}
    )
    .sort_values("importance", ascending=False)
    .head(15)
)

for idx, row in feature_importance.iterrows():
    print(f"   {row['feature']:25s}: {row['importance']:.4f}")

print("\n✅ All submissions created successfully!")
print("\n🎯 Recommended submissions to try (in order):")
print("   1. submission_stacking.csv - Stacking ensemble (usually best)")
print("   2. submission_voting.csv - Soft voting ensemble")
print("   3. submission_catboost_optimized.csv - CatBoost with optimized threshold")
print("   4. submission_calibrated.csv - Probability calibrated model")

print("\n" + "=" * 60)
print("✅ Pipeline completed successfully!")
print("=" * 60)
