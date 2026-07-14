"""
BioInsight-AI
Clinical Input Validation Module

Validates patient clinical data before it is passed to the
machine-learning prediction pipeline.

This module is intended for software-level validation and does not
provide medical diagnosis or medical advice.
"""

from numbers import Real
from typing import Any, Dict, List, Tuple


# ============================================================
# VALID CATEGORICAL VALUES
# ============================================================

VALID_SEX = {0, 1}

VALID_CHEST_PAIN_TYPES = {1, 2, 3, 4}

VALID_FASTING_BLOOD_SUGAR = {0, 1}

VALID_RESTING_ECG = {0, 1, 2}

VALID_EXERCISE_INDUCED_ANGINA = {0, 1}

VALID_ST_SLOPE = {1, 2, 3}

VALID_MAJOR_VESSELS = {0, 1, 2, 3}

VALID_THALASSEMIA = {3, 6, 7}


# ============================================================
# NUMERICAL VALIDATION LIMITS
# ============================================================

NUMERICAL_LIMITS = {
    "age": (1, 120),
    "resting_blood_pressure": (50, 250),
    "cholesterol": (50, 700),
    "max_heart_rate": (30, 250),
    "st_depression": (0.0, 10.0),
}


# ============================================================
# REQUIRED MODEL FEATURES
# ============================================================

REQUIRED_FEATURES = [
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
# VALID CATEGORICAL FEATURE VALUES
# ============================================================

CATEGORICAL_VALUES = {
    "sex": VALID_SEX,
    "chest_pain_type": VALID_CHEST_PAIN_TYPES,
    "fasting_blood_sugar": VALID_FASTING_BLOOD_SUGAR,
    "resting_ecg": VALID_RESTING_ECG,
    "exercise_induced_angina": VALID_EXERCISE_INDUCED_ANGINA,
    "st_slope": VALID_ST_SLOPE,
    "major_vessels": VALID_MAJOR_VESSELS,
    "thalassemia": VALID_THALASSEMIA,
}


# ============================================================
# FRIENDLY FEATURE NAMES
# ============================================================

FEATURE_DISPLAY_NAMES = {
    "age": "Age",
    "sex": "Sex",
    "chest_pain_type": "Chest Pain Type",
    "resting_blood_pressure": "Resting Blood Pressure",
    "cholesterol": "Cholesterol",
    "fasting_blood_sugar": "Fasting Blood Sugar",
    "resting_ecg": "Resting ECG",
    "max_heart_rate": "Maximum Heart Rate",
    "exercise_induced_angina": "Exercise-Induced Angina",
    "st_depression": "ST Depression",
    "st_slope": "ST Segment Slope",
    "major_vessels": "Number of Major Vessels",
    "thalassemia": "Thalassemia",
}


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def _is_numeric(value: Any) -> bool:
    """
    Return True when the value is a valid real number.

    Boolean values are rejected because bool is a subclass of int.
    """

    return isinstance(value, Real) and not isinstance(value, bool)


def _display_name(feature_name: str) -> str:
    """
    Return a human-readable feature name.
    """

    return FEATURE_DISPLAY_NAMES.get(feature_name, feature_name)


# ============================================================
# REQUIRED FEATURE VALIDATION
# ============================================================

def validate_required_features(
    patient_data: Dict[str, Any],
) -> List[str]:

    errors = []

    for feature in REQUIRED_FEATURES:

        if feature not in patient_data:
            errors.append(
                f"Missing required feature: {_display_name(feature)}."
            )

    return errors


# ============================================================
# NUMERICAL FEATURE VALIDATION
# ============================================================

def validate_numerical_features(
    patient_data: Dict[str, Any],
) -> List[str]:

    errors = []

    for feature, (minimum, maximum) in NUMERICAL_LIMITS.items():

        if feature not in patient_data:
            continue

        value = patient_data[feature]

        if not _is_numeric(value):

            errors.append(
                f"{_display_name(feature)} must be a numeric value."
            )

            continue

        if value < minimum or value > maximum:

            errors.append(
                f"{_display_name(feature)} must be between "
                f"{minimum} and {maximum}. Received: {value}."
            )

    return errors


# ============================================================
# CATEGORICAL FEATURE VALIDATION
# ============================================================

def validate_categorical_features(
    patient_data: Dict[str, Any],
) -> List[str]:

    errors = []

    for feature, valid_values in CATEGORICAL_VALUES.items():

        if feature not in patient_data:
            continue

        value = patient_data[feature]

        if value not in valid_values:

            allowed_values = sorted(valid_values)

            errors.append(
                f"Invalid value for {_display_name(feature)}: "
                f"{value}. Allowed values: {allowed_values}."
            )

    return errors


# ============================================================
# UNKNOWN FEATURE VALIDATION
# ============================================================

def validate_unknown_features(
    patient_data: Dict[str, Any],
) -> List[str]:

    errors = []

    unknown_features = sorted(
        set(patient_data.keys()) - set(REQUIRED_FEATURES)
    )

    for feature in unknown_features:

        errors.append(
            f"Unknown patient feature received: {feature}."
        )

    return errors


# ============================================================
# COMPLETE PATIENT VALIDATION
# ============================================================

def validate_patient_data(
    patient_data: Dict[str, Any],
    reject_unknown_features: bool = True,
) -> Tuple[bool, List[str]]:
    """
    Validate a complete patient record.

    Returns
    -------
    is_valid:
        True when no validation errors are found.

    errors:
        List containing all validation errors.
    """

    if not isinstance(patient_data, dict):

        return False, [
            "Patient data must be provided as a dictionary."
        ]

    errors = []

    errors.extend(
        validate_required_features(patient_data)
    )

    errors.extend(
        validate_numerical_features(patient_data)
    )

    errors.extend(
        validate_categorical_features(patient_data)
    )

    if reject_unknown_features:

        errors.extend(
            validate_unknown_features(patient_data)
        )

    return len(errors) == 0, errors


# ============================================================
# EXCEPTION-BASED VALIDATION
# ============================================================

def validate_patient_data_or_raise(
    patient_data: Dict[str, Any],
    reject_unknown_features: bool = True,
) -> None:
    """
    Validate patient data and raise ValueError when validation fails.
    """

    is_valid, errors = validate_patient_data(
        patient_data,
        reject_unknown_features=reject_unknown_features,
    )

    if not is_valid:

        error_message = (
            "Patient data validation failed:\n- "
            + "\n- ".join(errors)
        )

        raise ValueError(error_message)


# ============================================================
# VALID TEST PATIENT
# ============================================================

def create_valid_test_patient() -> Dict[str, Any]:
    """
    Return a valid patient record for testing the validation system.
    """

    return {
        "age": 54,
        "sex": 1,
        "chest_pain_type": 4,
        "resting_blood_pressure": 130,
        "cholesterol": 246,
        "fasting_blood_sugar": 0,
        "resting_ecg": 1,
        "max_heart_rate": 150,
        "exercise_induced_angina": 0,
        "st_depression": 1.2,
        "st_slope": 2,
        "major_vessels": 0,
        "thalassemia": 3,
    }


# ============================================================
# INVALID TEST PATIENT
# ============================================================

def create_invalid_test_patient() -> Dict[str, Any]:
    """
    Return an intentionally invalid patient record.
    """

    return {
        "age": 180,
        "sex": 5,
        "chest_pain_type": 10,
        "resting_blood_pressure": 20,
        "cholesterol": 1000,
        "fasting_blood_sugar": 3,
        "resting_ecg": 8,
        "max_heart_rate": 500,
        "exercise_induced_angina": 4,
        "st_depression": -5,
        "st_slope": 9,
        "major_vessels": 10,
        "thalassemia": 100,
    }


# ============================================================
# MODULE SELF-TEST
# ============================================================

def main():

    print("=" * 70)
    print("BIOINSIGHT-AI CLINICAL INPUT VALIDATION TEST")
    print("=" * 70)

    print("\nTEST 1: VALID PATIENT")
    print("-" * 70)

    valid_patient = create_valid_test_patient()

    is_valid, errors = validate_patient_data(valid_patient)

    print(f"Validation Result : {is_valid}")
    print(f"Errors Found      : {len(errors)}")

    if errors:

        for error in errors:
            print(f"- {error}")

    print("\nTEST 2: INVALID PATIENT")
    print("-" * 70)

    invalid_patient = create_invalid_test_patient()

    is_valid, errors = validate_patient_data(invalid_patient)

    print(f"Validation Result : {is_valid}")
    print(f"Errors Found      : {len(errors)}")

    for error in errors:
        print(f"- {error}")

    print("\nTEST 3: MISSING FEATURES")
    print("-" * 70)

    incomplete_patient = {
        "age": 50,
        "sex": 1,
    }

    is_valid, errors = validate_patient_data(incomplete_patient)

    print(f"Validation Result : {is_valid}")
    print(f"Errors Found      : {len(errors)}")

    for error in errors:
        print(f"- {error}")

    print("\n" + "=" * 70)

    if validate_patient_data(valid_patient)[0]:

        print("DAY 6 PART 1 VALIDATION SYSTEM COMPLETED SUCCESSFULLY")

    else:

        print("VALIDATION SYSTEM TEST FAILED")

    print("=" * 70)


if __name__ == "__main__":
    main()