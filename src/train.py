from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd

from xgboost import XGBClassifier
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_score,    
    train_test_split,
    GridSearchCV,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC


# --------------------------------------------------
# PROJECT PATHS
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "heart_disease_clean.csv"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "best_model.joblib"
)

CV_RESULTS_PATH = (
    PROJECT_ROOT
    / "reports"
    / "cv_results.csv"
)

FINAL_RESULTS_PATH = (
    PROJECT_ROOT
    / "reports"
    / "final_test_results.csv"
)

FIGURES_PATH = (
    PROJECT_ROOT
    / "reports"
    / "figures"
)


# --------------------------------------------------
# DATASET COLUMNS
# --------------------------------------------------

COLUMNS = [
    "age",
    "sex",
    "chest_pain_type",
    "resting_blood_pressure",
    "cholesterol",
    "fasting_blood_sugar",
    "resting_ecg",
    "max_heart_rate",
    "exercise_induced_angina",
    "st_depression",
    "st_slope",
    "major_vessels",
    "thalassemia",
    "target",
]


# --------------------------------------------------
# FEATURE GROUPS
# --------------------------------------------------

NUMERICAL_FEATURES = [
    "age",
    "resting_blood_pressure",
    "cholesterol",
    "max_heart_rate",
    "st_depression",
    "major_vessels",
]


CATEGORICAL_FEATURES = [
    "sex",
    "chest_pain_type",
    "fasting_blood_sugar",
    "resting_ecg",
    "exercise_induced_angina",
    "st_slope",
    "thalassemia",
]


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

def load_data():
    df = pd.read_csv(DATA_PATH)
    return df


# --------------------------------------------------
# TRAIN / TEST SPLIT
# --------------------------------------------------

def split_data(df):

    X = df.drop(columns=["target"])

    y = df["target"]

    return train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )


# --------------------------------------------------
# PREPROCESSING PIPELINE
# --------------------------------------------------

def build_preprocessor():

    numerical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median",
                ),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent",
                ),
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numerical",
                numerical_pipeline,
                NUMERICAL_FEATURES,
            ),
            (
                "categorical",
                categorical_pipeline,
                CATEGORICAL_FEATURES,
            ),
        ]
    )

    return preprocessor


# --------------------------------------------------
# CANDIDATE MODELS
# --------------------------------------------------

def build_models():

    models = {
        "Logistic Regression": Pipeline(
            steps=[
                (
                    "preprocessor",
                    build_preprocessor(),
                ),
                (
                    "model",
                    LogisticRegression(
                        C=0.5,
                        penalty="l2",
                        solver="liblinear",
                        class_weight="balanced",
                        max_iter=5000,
                        random_state=42,
                    )
                ),
            ]
        ),

        "Random Forest": Pipeline(
            steps=[
                (
                    "preprocessor",
                    build_preprocessor(),
                ),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=1000,
                        max_depth=None,
                        min_samples_split=2,
                        min_samples_leaf=1,
                        max_features="sqrt",
                        bootstrap=True,
                        class_weight="balanced",
                        random_state=42,
                    )
                ),
            ]
        ),

        "Support Vector Machine": Pipeline(
            steps=[
                (
                    "preprocessor",
                    build_preprocessor(),
                ),
                (
                    "model",
                    SVC(
                        C=2,
                        kernel="rbf",
                        gamma="scale",
                        probability=True,
                        random_state=42,
                    )
                ),
            ]
        ),
    
        "XGBoost": Pipeline(
            steps=[
                (
                    "preprocessor",
                    build_preprocessor(),
                ),
                (
                    "model",
                    XGBClassifier(
                        n_estimators=300,
                        learning_rate=0.05,
                        max_depth=4,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        eval_metric="logloss",
                        random_state=42,
                    ),
                ),
            ]
        ),
    }

    return models


# --------------------------------------------------
# CROSS-VALIDATION MODEL COMPARISON
# --------------------------------------------------

def compare_models_with_cv(
    models,
    X_train,
    y_train,
):

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    results = []

    print("\n" + "=" * 65)

    print("5-FOLD STRATIFIED CROSS-VALIDATION")

    print("=" * 65)

    for model_name, model in models.items():

        print(f"\nEvaluating: {model_name}")

        scores = cross_val_score(
            model,
            X_train,
            y_train,
            scoring="roc_auc",
            cv=cv,
            n_jobs=-1,
        )

        result = {
            "model": model_name,
            "mean_cv_roc_auc": scores.mean(),
            "std_cv_roc_auc": scores.std(),
        }

        results.append(result)

        print(
            f"Fold ROC-AUC Scores: "
            f"{scores.round(4)}"
        )

        print(
            f"Mean CV ROC-AUC: "
            f"{scores.mean():.4f}"
        )

        print(
            f"Std CV ROC-AUC : "
            f"{scores.std():.4f}"
        )

    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(
        by="mean_cv_roc_auc",
        ascending=False,
    ).reset_index(drop=True)

    return results_df


# --------------------------------------------------
# TRAIN SELECTED MODEL
# --------------------------------------------------

def train_selected_model(
    best_model_name,
    models,
    X_train,
    y_train,
):

    print("\n" + "=" * 65)
    print("HYPERPARAMETER TUNING")
    print("=" * 65)

    pipeline = models[best_model_name]

    param_grid = {}

    if best_model_name == "Logistic Regression":
        param_grid = {
            "model__C": [0.01, 0.1, 0.5, 1, 2, 5, 10],
            "model__solver": ["liblinear", "lbfgs"],
            "model__penalty": ["l2"],
            "model__class_weight": [None, "balanced"],
            "model__max_iter": [1000, 3000, 5000],
        }

    elif best_model_name == "Random Forest":
        param_grid = {
            "model__n_estimators": [200, 300, 500],
            "model__max_depth": [5, 8, 10, None],
            "model__min_samples_split": [2, 4, 6],
            "model__min_samples_leaf": [1, 2],
        }

    elif best_model_name == "Support Vector Machine":
        param_grid = {
            "model__C": [0.01, 0.1, 1, 2, 5, 10, 20, 50],
            "model__kernel": ["rbf"],
            "model__gamma": [
                "scale",
                "auto",
                0.001,
                0.01,
                0.1,
            ],
        }
    elif best_model_name == "XGBoost":
        param_grid = {
            "model__n_estimators": [200, 300, 500],
            "model__max_depth": [3, 4, 5],
            "model__learning_rate": [0.01, 0.05, 0.1],
            "model__subsample": [0.8, 1.0],
            "model__colsample_bytree": [0.8, 1.0],
        }
    cv = StratifiedKFold(
        n_splits=15,
        shuffle=True,
        random_state=42,
    )

    search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="roc_auc",
        cv=cv,
        n_jobs=-1,
        verbose=2,
    )

    search.fit(X_train, y_train)

    print("\nBest Parameters:")
    print(search.best_params_)

    print(f"\nBest CV ROC-AUC: {search.best_score_:.4f}")

    return search.best_estimator_

# --------------------------------------------------
# FINAL TEST EVALUATION
# --------------------------------------------------

def evaluate_final_model(
    model,
    model_name,
    X_test,
    y_test,
):

    predictions = model.predict(X_test)

    probabilities = (
        model.predict_proba(X_test)[:, 1]
    )

    metrics = {
        "model": model_name,
        "accuracy": accuracy_score(
            y_test,
            predictions,
        ),
        "precision": precision_score(
            y_test,
            predictions,
            zero_division=0,
        ),
        "recall": recall_score(
            y_test,
            predictions,
            zero_division=0,
        ),
        "f1_score": f1_score(
            y_test,
            predictions,
            zero_division=0,
        ),
        "roc_auc": roc_auc_score(
            y_test,
            probabilities,
        ),
    }

    print("\n" + "=" * 65)

    print("FINAL HELD-OUT TEST RESULTS")

    print("=" * 65)

    print(f"\nModel    : {model_name}")

    print(
        f"Accuracy : "
        f"{metrics['accuracy']:.4f}"
    )

    print(
        f"Precision: "
        f"{metrics['precision']:.4f}"
    )

    print(
        f"Recall   : "
        f"{metrics['recall']:.4f}"
    )

    print(
        f"F1 Score : "
        f"{metrics['f1_score']:.4f}"
    )

    print(
        f"ROC-AUC  : "
        f"{metrics['roc_auc']:.4f}"
    )

    return metrics, predictions, probabilities


# --------------------------------------------------
# SAVE RESULTS
# --------------------------------------------------

def save_results(
    cv_results_df,
    final_metrics,
):

    CV_RESULTS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    cv_results_df.to_csv(
        CV_RESULTS_PATH,
        index=False,
    )

    final_results_df = pd.DataFrame(
        [final_metrics]
    )

    final_results_df.to_csv(
        FINAL_RESULTS_PATH,
        index=False,
    )

    print(
        f"\nCV results saved to:\n"
        f"{CV_RESULTS_PATH}"
    )

    print(
        f"\nFinal test results saved to:\n"
        f"{FINAL_RESULTS_PATH}"
    )


# --------------------------------------------------
# CONFUSION MATRIX
# --------------------------------------------------

def save_confusion_matrix(
    model,
    X_test,
    y_test,
):

    FIGURES_PATH.mkdir(
        parents=True,
        exist_ok=True,
    )

    ConfusionMatrixDisplay.from_estimator(
        model,
        X_test,
        y_test,
        display_labels=[
            "No Heart Disease",
            "Heart Disease",
        ],
        cmap="Blues",
    )

    plt.title(
        "Final Model Confusion Matrix"
    )

    plt.tight_layout()

    output_path = (
        FIGURES_PATH
        / "07_final_confusion_matrix.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print(
        f"\nConfusion matrix saved to:\n"
        f"{output_path}"
    )


# --------------------------------------------------
# ROC CURVE
# --------------------------------------------------

def save_roc_curve(
    model,
    X_test,
    y_test,
):

    RocCurveDisplay.from_estimator(
        model,
        X_test,
        y_test,
    )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Random Classifier",
    )

    plt.title(
        "Final Model ROC Curve"
    )

    plt.legend()

    plt.tight_layout()

    output_path = (
        FIGURES_PATH
        / "08_final_roc_curve.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print(
        f"\nROC curve saved to:\n"
        f"{output_path}"
    )


# --------------------------------------------------
# SAVE FINAL MODEL
# --------------------------------------------------

def save_model(model):

    MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        model,
        MODEL_PATH,
    )

    print(
        f"\nFinal model saved to:\n"
        f"{MODEL_PATH}"
    )


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():

    print("=" * 65)

    print(
        "BIOINSIGHT AI — "
        "CROSS-VALIDATION & FINAL MODEL TRAINING"
    )

    print("=" * 65)

    df = load_data()

    print(f"\nDataset Shape: {df.shape}")

    X_train, X_test, y_train, y_test = (
        split_data(df)
    )

    print(
        f"\nTraining Patients: "
        f"{len(X_train)}"
    )

    print(
        f"Testing Patients : "
        f"{len(X_test)}"
    )

    models = build_models()

    cv_results_df = compare_models_with_cv(
        models,
        X_train,
        y_train,
    )

    print("\n" + "=" * 65)

    print("CROSS-VALIDATION MODEL COMPARISON")

    print("=" * 65)

    print(
        cv_results_df.to_string(
            index=False,
        )
    )

    best_model_name = (
        cv_results_df.iloc[0]["model"]
    )

    best_model = train_selected_model(
        best_model_name,
        models,
        X_train,
        y_train,
    )

    (
        final_metrics,
        predictions,
        probabilities,
    ) = evaluate_final_model(
        best_model,
        best_model_name,
        X_test,
        y_test,
    )

    save_results(
        cv_results_df,
        final_metrics,
    )

    save_confusion_matrix(
        best_model,
        X_test,
        y_test,
    )

    save_roc_curve(
        best_model,
        X_test,
        y_test,
    )

    save_model(best_model)

    print("\n" + "=" * 65)

    print(
        "DAY 3 MODEL PIPELINE "
        "COMPLETED SUCCESSFULLY"
    )

    print("=" * 65)


if __name__ == "__main__":
    main()