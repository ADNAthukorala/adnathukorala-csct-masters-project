from __future__ import annotations

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from .feature_engineering import apply_feature_weights, encode_features


# Map each exercise dataset feature to the matching user-profile field.
# These are the five features used by cosine similarity.
FEATURE_MAP = {
    "muscle_group": "target_muscle_group",
    "level": "experience_level",
    "equipment": "available_equipment",
    "mechanics": "exercise_mechanics",
    "force": "force_type",
}

# Equal weights mean that no profile feature is intentionally given more
# importance than another before cosine similarity is calculated.
FEATURE_WEIGHTS = {
    "muscle_group": 5.0,
    "level": 3.0,
    "equipment": 3.0,
    "mechanics": 2.0,
    "force": 2.0,
}

# Columns returned for display and download only. This list does not take part
# in encoding, similarity, weighting or ranking. The names simply follow the
# cleaned_exercise_dataset.csv schema.
OUTPUT_COLUMNS = [
    "rank",
    "exercise_id",
    "exercise_name",
    "similarity_score",
    "difficulty_score",
    "level",
    "equipment",
    "muscle_group",
    "muscles_worked",
    "mechanics",
    "force",
    "preparation",
    "execution",
]


def recommend_exercises(
    exercises: pd.DataFrame,
    target_muscle_group: str,
    experience_level: str,
    available_equipment: list[str],
    exercise_mechanics: str,
    force_type: str,
    top_k: int = 10,
    feature_weights: dict[str, float] | None = None,
) -> pd.DataFrame:
    """
    Generate Top-K exercise recommendations for one manually entered user profile.

    The model uses five content features: target muscle group, experience level,
    available equipment, exercise mechanics, and force type. There are no hard
    or strict filters. Every exercise remains eligible and is ranked only by its
    cosine-similarity score with the user profile.
    """
    if feature_weights is None:
        feature_weights = FEATURE_WEIGHTS

    # Convert the selected user inputs into a one-row dataframe so the user
    # profile can be encoded in exactly the same feature space as the exercises.
    user_profile = pd.DataFrame([
        {
            "target_muscle_group": target_muscle_group,
            "experience_level": experience_level,
            "available_equipment": ", ".join(available_equipment),
            "exercise_mechanics": exercise_mechanics,
            "force_type": force_type,
        }
    ])

    # Multi-hot encode exercise features and user preferences.
    exercise_matrix, user_matrix = encode_features(
        exercises,
        user_profile,
        FEATURE_MAP,
    )

    # Apply the configured feature weights before similarity is calculated.
    exercise_matrix = apply_feature_weights(exercise_matrix, feature_weights)
    user_matrix = apply_feature_weights(user_matrix, feature_weights)

    # Calculate one cosine-similarity score between the user profile and every
    # exercise in the dataset. No exercise is removed by a strict filter.
    similarity_scores = cosine_similarity(user_matrix, exercise_matrix)[0]

    recommendations = exercises.copy()
    recommendations["similarity_score"] = similarity_scores

    # Rank every eligible exercise from highest to lowest score and return only
    # the requested Top-K rows.
    #
    # Exercises that share a name are kept. They are separate exercises
    # performed with different equipment, so removing one would discard an
    # exercise the user may be able to perform.
    #
    # Rounding the score before sorting removes floating-point noise, which is
    # about 1e-16 and can otherwise separate exercises that are mathematically
    # tied. Ten decimal places is far below any real difference in score.
    # exercise_id then settles any remaining tie. Because it is unique, the
    # order is fully determined and never depends on the order the rows happen
    # to sit in the CSV file.
    recommendations = recommendations.assign(
        rank_score=recommendations["similarity_score"].round(10)
    )
    recommendations = recommendations.sort_values(
        by=["rank_score", "exercise_id"],
        ascending=[False, True],
    )
    recommendations = recommendations.drop(columns=["rank_score"]).head(top_k)
    recommendations = recommendations.reset_index(drop=True)

    # Number the results 1, 2, 3... so the ranking is obvious in the output.
    recommendations.insert(0, "rank", range(1, len(recommendations) + 1))

    # Return the display columns, skipping any the dataset happens not to have.
    output_columns = [
        column for column in OUTPUT_COLUMNS
        if column in recommendations.columns
    ]
    return recommendations[output_columns]
