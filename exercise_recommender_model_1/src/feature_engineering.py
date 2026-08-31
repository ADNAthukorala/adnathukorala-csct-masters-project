from __future__ import annotations

import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer

from .text_utils import split_values


def encode_features(
    exercises: pd.DataFrame,
    user_profile: pd.DataFrame,
    feature_map: dict[str, str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Encode exercise and user features using multi-hot encoding."""
    exercise_parts = []
    user_parts = []

    # Each feature is encoded on its own, then the pieces are joined at the end.
    for exercise_col, user_col in feature_map.items():
        # A cell can hold more than one value, so it is split into a list first.
        exercise_values = exercises[exercise_col].apply(split_values)
        user_values = user_profile[user_col].apply(split_values)

        # The encoder is fitted on the exercises and the user together. This is the
        # important bit: it guarantees both sides end up with the same columns in the
        # same order, which is what makes the similarity comparison valid.
        combined_values = pd.concat([exercise_values, user_values], axis=0)

        encoder = MultiLabelBinarizer()
        encoder.fit(combined_values)

        # One column per possible value, holding 1 when the row has that value.
        # The feature name is added as a prefix to keep column names unique.
        exercise_encoded = pd.DataFrame(
            encoder.transform(exercise_values),
            columns=[f"{exercise_col}_{class_name}" for class_name in encoder.classes_],
            index=exercises.index,
        )

        user_encoded = pd.DataFrame(
            encoder.transform(user_values),
            columns=[f"{exercise_col}_{class_name}" for class_name in encoder.classes_],
            index=user_profile.index,
        )

        exercise_parts.append(exercise_encoded)
        user_parts.append(user_encoded)

    # Join the per-feature blocks side by side into one wide matrix each.
    exercise_matrix = pd.concat(exercise_parts, axis=1)
    user_matrix = pd.concat(user_parts, axis=1)

    return exercise_matrix, user_matrix


def apply_feature_weights(
    matrix: pd.DataFrame,
    feature_weights: dict[str, float],
) -> pd.DataFrame:
    """Apply weights so more important features influence similarity more."""
    matrix = matrix.copy()

    for feature, weight in feature_weights.items():
        # Every encoded column starts with its feature name, so the prefix is used to
        # find all the columns belonging to one feature and scale them together.
        matching_columns = [column for column in matrix.columns if column.startswith(feature + "_")]
        matrix[matching_columns] = matrix[matching_columns] * weight

    return matrix
