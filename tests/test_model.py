"""
BioInsight-AI
Automated Tests for the Trained Machine Learning Pipeline
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = PROJECT_ROOT / "models" / "best_model.joblib"


# ============================================================
# PATIENT FEATURE CONFIGURATION
# ============================================================

FEATURE_COLUMNS = [
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
]


# ============================================================
# TEST PATIENTS
# ============================================================

VALID_PATIENT = {
    "age": 54,
    "sex": 1,
    "chest_pain_type": 2,
    "resting_blood_pressure": 130,
    "cholesterol": 246,
    "fasting_blood_sugar": 0,
    "resting_ecg": 1,
    "max_heart_rate": 150,
    "exercise_induced_angina": 0,
    "st_depression": 1.2,
    "st_slope": 2,
    "major_vessels": 0,
    "thalassemia": 2,
}


SECOND_VALID_PATIENT = {
    "age": 63,
    "sex": 1,
    "chest_pain_type": 4,
    "resting_blood_pressure": 145,
    "cholesterol": 233,
    "fasting_blood_sugar": 1,
    "resting_ecg": 0,
    "max_heart_rate": 150,
    "exercise_induced_angina": 0,
    "st_depression": 2.3,
    "st_slope": 3,
    "major_vessels": 0,
    "thalassemia": 6,
}


# ============================================================
# FIXTURES
# ============================================================

@pytest.fixture(scope="module")
def model():

    assert MODEL_PATH.exists(), (
        f"Saved model was not found at: {MODEL_PATH}"
    )

    loaded_model = joblib.load(MODEL_PATH)

    return loaded_model


@pytest.fixture
def patient_dataframe():

    return pd.DataFrame(
        [VALID_PATIENT],
        columns=FEATURE_COLUMNS,
    )


@pytest.fixture
def two_patient_dataframe():

    return pd.DataFrame(
        [
            VALID_PATIENT,
            SECOND_VALID_PATIENT,
        ],
        columns=FEATURE_COLUMNS,
    )


# ============================================================
# MODEL FILE TESTS
# ============================================================

def test_model_file_exists():

    assert MODEL_PATH.exists()

    assert MODEL_PATH.is_file()


def test_model_file_is_not_empty():

    assert MODEL_PATH.exists()

    assert MODEL_PATH.stat().st_size > 0


def test_model_loads_successfully():

    loaded_model = joblib.load(MODEL_PATH)

    assert loaded_model is not None


# ============================================================
# MODEL INTERFACE TESTS
# ============================================================

def test_model_has_predict_method(model):

    assert hasattr(model, "predict")

    assert callable(model.predict)


def test_model_has_predict_proba_method(model):

    assert hasattr(model, "predict_proba")

    assert callable(model.predict_proba)


# ============================================================
# SINGLE PATIENT PREDICTION TESTS
# ============================================================

def test_model_accepts_patient_dataframe(
    model,
    patient_dataframe,
):

    prediction = model.predict(patient_dataframe)

    assert prediction is not None


def test_single_patient_prediction_shape(
    model,
    patient_dataframe,
):

    prediction = model.predict(patient_dataframe)

    assert prediction.shape == (1,)


def test_single_patient_prediction_is_valid_class(
    model,
    patient_dataframe,
):

    prediction = model.predict(patient_dataframe)

    assert prediction[0] in [0, 1]


def test_single_patient_prediction_is_integer_like(
    model,
    patient_dataframe,
):

    prediction = model.predict(patient_dataframe)

    assert isinstance(
        prediction[0],
        (int, np.integer),
    )


# ============================================================
# SINGLE PATIENT PROBABILITY TESTS
# ============================================================

def test_predict_proba_returns_two_class_probabilities(
    model,
    patient_dataframe,
):

    probabilities = model.predict_proba(patient_dataframe)

    assert probabilities.shape == (1, 2)


def test_probabilities_are_between_zero_and_one(
    model,
    patient_dataframe,
):

    probabilities = model.predict_proba(patient_dataframe)

    assert np.all(probabilities >= 0)

    assert np.all(probabilities <= 1)


def test_probabilities_sum_to_one(
    model,
    patient_dataframe,
):

    probabilities = model.predict_proba(patient_dataframe)

    assert np.isclose(
        probabilities[0].sum(),
        1.0,
    )


def test_prediction_matches_highest_probability_class(
    model,
    patient_dataframe,
):

    prediction = model.predict(patient_dataframe)

    probabilities = model.predict_proba(patient_dataframe)

    highest_probability_class = np.argmax(
        probabilities,
        axis=1,
    )

    assert np.array_equal(
        prediction,
        highest_probability_class,
    )


# ============================================================
# MULTIPLE PATIENT PREDICTION TESTS
# ============================================================

def test_model_accepts_multiple_patients(
    model,
    two_patient_dataframe,
):

    predictions = model.predict(two_patient_dataframe)

    assert predictions.shape == (2,)


def test_multiple_patient_predictions_are_valid_classes(
    model,
    two_patient_dataframe,
):

    predictions = model.predict(two_patient_dataframe)

    assert np.all(
        np.isin(
            predictions,
            [0, 1],
        )
    )


def test_multiple_patient_probability_shape(
    model,
    two_patient_dataframe,
):

    probabilities = model.predict_proba(
        two_patient_dataframe
    )

    assert probabilities.shape == (2, 2)


def test_each_patient_probability_sums_to_one(
    model,
    two_patient_dataframe,
):

    probabilities = model.predict_proba(
        two_patient_dataframe
    )

    probability_sums = probabilities.sum(axis=1)

    assert np.allclose(
        probability_sums,
        np.ones(2),
    )


# ============================================================
# OUTPUT STABILITY TESTS
# ============================================================

def test_prediction_is_deterministic(
    model,
    patient_dataframe,
):

    prediction_1 = model.predict(patient_dataframe)

    prediction_2 = model.predict(patient_dataframe)

    assert np.array_equal(
        prediction_1,
        prediction_2,
    )


def test_probability_output_is_deterministic(
    model,
    patient_dataframe,
):

    probabilities_1 = model.predict_proba(
        patient_dataframe
    )

    probabilities_2 = model.predict_proba(
        patient_dataframe
    )

    assert np.allclose(
        probabilities_1,
        probabilities_2,
    )


# ============================================================
# FEATURE SCHEMA TESTS
# ============================================================

def test_patient_dataframe_has_correct_number_of_features(
    patient_dataframe,
):

    assert patient_dataframe.shape[1] == 13


def test_patient_dataframe_has_correct_feature_order(
    patient_dataframe,
):

    assert patient_dataframe.columns.tolist() == FEATURE_COLUMNS


def test_no_missing_values_in_test_patient(
    patient_dataframe,
):

    assert patient_dataframe.isnull().sum().sum() == 0


# ============================================================
# BATCH OUTPUT CONSISTENCY TEST
# ============================================================

def test_batch_predictions_match_individual_predictions(
    model,
    two_patient_dataframe,
):

    batch_predictions = model.predict(
        two_patient_dataframe
    )

    first_patient = two_patient_dataframe.iloc[[0]]

    second_patient = two_patient_dataframe.iloc[[1]]

    first_prediction = model.predict(first_patient)

    second_prediction = model.predict(second_patient)

    individual_predictions = np.concatenate(
        [
            first_prediction,
            second_prediction,
        ]
    )

    assert np.array_equal(
        batch_predictions,
        individual_predictions,
    )