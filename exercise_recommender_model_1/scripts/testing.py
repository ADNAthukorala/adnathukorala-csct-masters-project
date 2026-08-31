"""
Tests for Model 1, the content-based exercise recommender.

The tests check the main building blocks of the model: text cleaning, dataset
loading, feature encoding, feature weighting, category filtering and the
recommendation ranking itself.

Nothing in the recommender is rewritten here. Every test calls the same
functions the Streamlit app uses, so a passing test says something about the
real model and not about a copy of it.

Run from the project folder:

    python scripts/testing.py

Each test prints PASSED or FAILED and a short summary is printed at the end.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pandas as pd


# File paths
PROJECT_DIRECTORY = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIRECTORY))

from src.data_loader import load_exercise_data                 # noqa: E402
from src.feature_engineering import (                          # noqa: E402
    apply_feature_weights,
    encode_features,
)
from src.filters import ANY_OPTION, apply_filters              # noqa: E402
from src.recommender import FEATURE_MAP, recommend_exercises   # noqa: E402
from src.text_utils import clean_text, split_values            # noqa: E402


# Helper functions

def show_result(passed: bool, reason: str = "") -> bool:
    """Print the outcome of one test and return it so it can be counted."""
    if passed:
        print("PASSED")
    else:
        print("FAILED")
        print(reason)
    return passed


def save_temp_csv(dataframe: pd.DataFrame, file_name: str) -> Path:
    """Save a small dataframe as a CSV file so it can be loaded from disk.

    The loading function reads from a file, so a real file is needed. A
    temporary folder is used so the project folder stays clean.
    """
    path = Path(tempfile.gettempdir()) / file_name
    dataframe.to_csv(path, index=False)
    return path


def make_small_dataset() -> pd.DataFrame:
    """Build a tiny exercise dataset with the columns the loader expects.

    Some values are deliberately messy (capital letters, extra spaces and a
    blank preparation) so the cleaning steps can be checked.
    """
    return pd.DataFrame([
        {
            "exercise_id": "EX001",
            "exercise_name": "Bench Press",
            "difficulty_score": 3,
            "level": "  Beginner ",
            "equipment": "Barbell",
            "muscle_group": "Chest",
            "muscles_worked": "Pectoralis Major",
            "mechanics": "Compound",
            "force": "Push",
            "preparation": "Lie on the bench.",
            "execution": "Press the bar upwards.",
        },
        {
            "exercise_id": "EX002",
            "exercise_name": "Lat Pulldown",
            "difficulty_score": 2,
            "level": "Intermediate",
            "equipment": "Cable",
            "muscle_group": "Back",
            "muscles_worked": "Latissimus Dorsi",
            "mechanics": "Compound",
            "force": "Pull",
            # Left blank on purpose to check how missing instructions are handled.
            "preparation": None,
            "execution": "Pull the bar to the chest.",
        },
    ])


# Test 1: text cleaning and splitting

def test_text_cleaning() -> bool:
    """Check that text values are cleaned and split into the expected format."""
    print("Test 1: Text cleaning and splitting")

    # Capital letters and the spaces around a value should be removed.
    if clean_text("  Barbell  ") != "barbell":
        return show_result(False, "Expected 'barbell' after cleaning '  Barbell  '.")

    # Repeated spaces inside a value should collapse into a single space.
    if clean_text("Dumbbell    Press") != "dumbbell press":
        return show_result(False, "Expected repeated spaces to collapse into one space.")

    # A missing value should become empty text rather than the word 'nan'.
    if clean_text(None) != "":
        return show_result(False, "Expected a missing value to become empty text.")

    # The dataset uses several separators, so all of them should split the same way.
    if split_values("Chest & Back") != ["chest", "back"]:
        return show_result(False, "Expected '&' to be treated as a separator.")

    if split_values("Push / Pull") != ["push", "pull"]:
        return show_result(False, "Expected '/' to be treated as a separator.")

    if split_values("Push; Pull | Static") != ["push", "pull", "static"]:
        return show_result(False, "Expected ';' and '|' to be treated as separators.")

    # An empty value should give an empty list, not a list holding empty text.
    if split_values("") != []:
        return show_result(False, "Expected an empty value to give an empty list.")

    return show_result(True)


# Test 2: dataset loading and cleaning

def test_dataset_loading() -> bool:
    """Check that a small valid dataset is loaded and cleaned correctly."""
    print("Test 2: Dataset loading and cleaning")

    # Save the small dataset to a temporary file so the real loader can read it.
    path = save_temp_csv(make_small_dataset(), "test_exercises.csv")
    exercises = load_exercise_data(path)

    # Both rows should survive loading.
    if len(exercises) != 2:
        return show_result(False, f"Expected 2 exercises but got {len(exercises)}.")

    # The level value had capital letters and extra spaces, so it should be cleaned.
    if exercises.loc[0, "level"] != "beginner":
        return show_result(
            False,
            f"Expected level 'beginner' but got '{exercises.loc[0, 'level']}'.",
        )

    # Equipment and muscle group are used for matching, so they are cleaned too.
    if exercises.loc[0, "equipment"] != "barbell":
        return show_result(False, "Expected equipment to be cleaned to 'barbell'.")

    if exercises.loc[0, "muscle_group"] != "chest":
        return show_result(False, "Expected muscle group to be cleaned to 'chest'.")

    # The second exercise has no preparation text, so it should become empty text.
    # Without this the app would print the word 'nan' on the recommendation card.
    if exercises.loc[1, "preparation"] != "":
        return show_result(False, "Expected a blank preparation to become empty text.")

    # The execution text should still be there and unchanged.
    if exercises.loc[1, "execution"] != "Pull the bar to the chest.":
        return show_result(False, "Expected the execution text to be kept as it is.")

    return show_result(True)


# Test 3: missing required column

def test_missing_column() -> bool:
    """Check that the loader reports a clear error when a column is missing."""
    print("Test 3: Missing required column")

    # Remove the level column, which the recommender needs.
    broken_dataset = make_small_dataset().drop(columns=["level"])
    path = save_temp_csv(broken_dataset, "test_exercises_missing_column.csv")

    # The loader should raise a ValueError here instead of failing later in a
    # confusing way, so the error is caught and checked.
    try:
        load_exercise_data(path)
        return show_result(False, "Expected a ValueError but the file loaded without one.")
    except ValueError as error:
        # The message should name the missing column so the problem is easy to fix.
        if "level" not in str(error):
            return show_result(False, "Expected the error message to name the missing column.")
        return show_result(True)


# Test 4: feature encoding

def test_feature_encoding() -> bool:
    """Check that exercises and the user profile end up in the same feature space."""
    print("Test 4: Feature encoding")

    # Two simple exercises with clearly different features.
    exercises = pd.DataFrame([
        {
            "muscle_group": "chest",
            "level": "beginner",
            "equipment": "barbell",
            "mechanics": "compound",
            "force": "push",
        },
        {
            "muscle_group": "back",
            "level": "advanced",
            "equipment": "cable",
            "mechanics": "isolated",
            "force": "pull",
        },
    ])

    # One user profile. These column names match the user side of FEATURE_MAP.
    user_profile = pd.DataFrame([
        {
            "target_muscle_group": "chest",
            "experience_level": "beginner",
            "available_equipment": "barbell, dumbbell",
            "exercise_mechanics": "compound",
            "force_type": "push",
        },
    ])

    exercise_matrix, user_matrix = encode_features(exercises, user_profile, FEATURE_MAP)

    # This is the key check. Cosine similarity is only meaningful when both
    # sides have the same columns in the same order.
    if list(exercise_matrix.columns) != list(user_matrix.columns):
        return show_result(
            False,
            "Expected the exercises and the user profile to share the same columns.",
        )

    # One row per exercise and one row for the single user profile.
    if len(exercise_matrix) != 2 or len(user_matrix) != 1:
        return show_result(False, "Expected 2 exercise rows and 1 user row.")

    # The first exercise targets the chest, so that column should hold a 1.
    if exercise_matrix.loc[0, "muscle_group_chest"] != 1:
        return show_result(False, "Expected the chest exercise to be marked with a 1.")

    # The second exercise does not, so the same column should hold a 0.
    if exercise_matrix.loc[1, "muscle_group_chest"] != 0:
        return show_result(False, "Expected the back exercise to be marked with a 0 for chest.")

    # The user listed two pieces of equipment, so both should be marked.
    if user_matrix.loc[0, "equipment_barbell"] != 1:
        return show_result(False, "Expected barbell to be marked with a 1 for the user.")

    if user_matrix.loc[0, "equipment_dumbbell"] != 1:
        return show_result(False, "Expected dumbbell to be marked with a 1 for the user.")

    return show_result(True)


# Test 5: feature weighting

def test_feature_weighting() -> bool:
    """Check that the configured weights are applied to the encoded columns."""
    print("Test 5: Feature weighting")

    # A tiny encoded matrix, so the expected result is obvious.
    matrix = pd.DataFrame([{"muscle_group_chest": 1, "force_push": 1}])

    weighted_matrix = apply_feature_weights(matrix, {"muscle_group": 5.0, "force": 2.0})

    # Muscle group has a weight of 5, so a 1 should become a 5.
    if weighted_matrix.loc[0, "muscle_group_chest"] != 5:
        return show_result(
            False,
            f"Expected 5 but got {weighted_matrix.loc[0, 'muscle_group_chest']}.",
        )

    # Force has a weight of 2, so a 1 should become a 2.
    if weighted_matrix.loc[0, "force_push"] != 2:
        return show_result(
            False,
            f"Expected 2 but got {weighted_matrix.loc[0, 'force_push']}.",
        )

    return show_result(True)


# Test 6: exact filtering

def make_filter_dataset() -> pd.DataFrame:
    """Build three exercises that differ from each other in an obvious way."""
    return pd.DataFrame([
        {
            "exercise_id": "EX001",
            "muscle_group": "chest",
            "level": "beginner",
            "equipment": "barbell",
            "mechanics": "compound",
            "force": "push",
        },
        {
            "exercise_id": "EX002",
            "muscle_group": "chest",
            "level": "advanced",
            "equipment": "dumbbell",
            "mechanics": "isolated",
            "force": "push",
        },
        {
            "exercise_id": "EX003",
            "muscle_group": "back",
            "level": "beginner",
            "equipment": "cable",
            "mechanics": "compound",
            "force": "pull",
        },
    ])


def test_exact_filtering() -> bool:
    """Check that the filter keeps only exercises matching every selected category."""
    print("Test 6: Exact filtering")

    exercises = make_filter_dataset()

    # Only the first exercise matches all five of these choices.
    filtered = apply_filters(
        exercises=exercises,
        target_muscle_group="chest",
        experience_level="beginner",
        available_equipment=["barbell"],
        exercise_mechanics="compound",
        force_type="push",
    )

    if len(filtered) != 1:
        return show_result(False, f"Expected 1 exercise but got {len(filtered)}.")

    if filtered.iloc[0]["exercise_id"] != "EX001":
        return show_result(
            False,
            f"Expected EX001 but got {filtered.iloc[0]['exercise_id']}.",
        )

    # Filtering on the muscle group alone should keep both chest exercises.
    chest_only = apply_filters(
        exercises=exercises,
        target_muscle_group="chest",
        experience_level=ANY_OPTION,
        available_equipment=[],
        exercise_mechanics=ANY_OPTION,
        force_type=ANY_OPTION,
    )

    if list(chest_only["exercise_id"]) != ["EX001", "EX002"]:
        return show_result(False, "Expected both chest exercises to be kept.")

    return show_result(True)


# Test 7: the "Any" option

def test_any_option() -> bool:
    """Check that 'Any' and an empty equipment list do not remove exercises."""
    print("Test 7: 'Any' option")

    exercises = make_filter_dataset()

    # Every category is left open, so nothing should be removed.
    filtered = apply_filters(
        exercises=exercises,
        target_muscle_group=ANY_OPTION,
        experience_level=ANY_OPTION,
        available_equipment=[],
        exercise_mechanics=ANY_OPTION,
        force_type=ANY_OPTION,
    )

    if len(filtered) != 3:
        return show_result(
            False,
            f"Expected all 3 exercises to be kept but got {len(filtered)}.",
        )

    # Now only the experience level is set. Two exercises are beginner level,
    # and the empty equipment list should not remove either of them, even though
    # they use different equipment.
    beginner_only = apply_filters(
        exercises=exercises,
        target_muscle_group=ANY_OPTION,
        experience_level="beginner",
        available_equipment=[],
        exercise_mechanics=ANY_OPTION,
        force_type=ANY_OPTION,
    )

    if list(beginner_only["exercise_id"]) != ["EX001", "EX003"]:
        return show_result(
            False,
            "Expected the two beginner exercises to be kept when equipment is empty "
            f"but got {list(beginner_only['exercise_id'])}.",
        )

    return show_result(True)


# Test 8: recommendation ranking

def test_recommendation_ranking() -> bool:
    """Check that a close match to the profile is ranked above a poor match."""
    print("Test 8: Recommendation ranking")

    # EX001 matches the profile below on all five features.
    # EX002 matches none of them.
    exercises = pd.DataFrame([
        {
            "exercise_id": "EX001",
            "exercise_name": "Bench Press",
            "muscle_group": "chest",
            "level": "beginner",
            "equipment": "barbell",
            "mechanics": "compound",
            "force": "push",
        },
        {
            "exercise_id": "EX002",
            "exercise_name": "Cable Row",
            "muscle_group": "back",
            "level": "advanced",
            "equipment": "cable",
            "mechanics": "isolated",
            "force": "pull",
        },
    ])

    recommendations = recommend_exercises(
        exercises=exercises,
        target_muscle_group="chest",
        experience_level="beginner",
        available_equipment=["barbell"],
        exercise_mechanics="compound",
        force_type="push",
        top_k=2,
    )

    # The strong match should be first in the list.
    if recommendations.iloc[0]["exercise_id"] != "EX001":
        return show_result(False, "Expected exercise EX001 to rank above exercise EX002.")

    # It should also have the higher similarity score, which is what puts it first.
    top_score = recommendations.iloc[0]["similarity_score"]
    second_score = recommendations.iloc[1]["similarity_score"]

    if top_score <= second_score:
        return show_result(
            False,
            f"Expected the top score {top_score} to be higher than {second_score}.",
        )

    return show_result(True)


# Test 9: the same user profile gives the same recommendations

def make_ranking_dataset() -> pd.DataFrame:
    """Build five exercises with a mix of matching and non-matching features."""
    return pd.DataFrame([
        {
            "exercise_id": "EX001",
            "exercise_name": "Bench Press",
            "muscle_group": "chest",
            "level": "beginner",
            "equipment": "barbell",
            "mechanics": "compound",
            "force": "push",
        },
        {
            "exercise_id": "EX002",
            "exercise_name": "Push Up",
            "muscle_group": "chest",
            "level": "beginner",
            "equipment": "body weight",
            "mechanics": "compound",
            "force": "push",
        },
        {
            "exercise_id": "EX003",
            "exercise_name": "Dumbbell Fly",
            "muscle_group": "chest",
            "level": "advanced",
            "equipment": "dumbbell",
            "mechanics": "isolated",
            "force": "push",
        },
        {
            "exercise_id": "EX004",
            "exercise_name": "Lat Pulldown",
            "muscle_group": "back",
            "level": "intermediate",
            "equipment": "cable",
            "mechanics": "compound",
            "force": "pull",
        },
        {
            "exercise_id": "EX005",
            "exercise_name": "Leg Press",
            "muscle_group": "legs",
            "level": "beginner",
            "equipment": "machine",
            "mechanics": "compound",
            "force": "push",
        },
    ])


def test_same_profile_same_results() -> bool:
    """Check that the same profile always produces the same recommendations."""
    print("Test 9: Same user profile gives the same recommendations")

    exercises = make_ranking_dataset()

    # Run the recommender twice with the same profile.
    # The results should be identical because the input has not changed.
    first_run = recommend_exercises(
        exercises=exercises,
        target_muscle_group="chest",
        experience_level="beginner",
        available_equipment=["barbell"],
        exercise_mechanics="compound",
        force_type="push",
        top_k=3,
    )

    second_run = recommend_exercises(
        exercises=exercises,
        target_muscle_group="chest",
        experience_level="beginner",
        available_equipment=["barbell"],
        exercise_mechanics="compound",
        force_type="push",
        top_k=3,
    )

    # The same exercises should come back in the same order.
    if list(first_run["exercise_id"]) != list(second_run["exercise_id"]):
        return show_result(
            False,
            "Expected both runs to return the same exercise IDs in the same order.",
        )

    # The scores should match as well. They are rounded first because floating
    # point numbers can carry a tiny amount of noise.
    first_scores = [round(score, 10) for score in first_run["similarity_score"]]
    second_scores = [round(score, 10) for score in second_run["similarity_score"]]

    if first_scores != second_scores:
        return show_result(False, "Expected both runs to return the same similarity scores.")

    return show_result(True)


# Test 10: Top-K and ranking order

def test_top_k_and_order() -> bool:
    """Check that Top-K returns the right number of rows in a consistent order."""
    print("Test 10: Top-K and ranking order")

    # Three exercises with exactly the same features, so their scores are tied.
    # They are listed out of order on purpose, which shows that the tie is
    # settled by exercise_id and not by the order of the rows in the file.
    exercises = pd.DataFrame([
        {
            "exercise_id": "EX003",
            "exercise_name": "Incline Press",
            "muscle_group": "chest",
            "level": "beginner",
            "equipment": "barbell",
            "mechanics": "compound",
            "force": "push",
        },
        {
            "exercise_id": "EX001",
            "exercise_name": "Bench Press",
            "muscle_group": "chest",
            "level": "beginner",
            "equipment": "barbell",
            "mechanics": "compound",
            "force": "push",
        },
        {
            "exercise_id": "EX002",
            "exercise_name": "Decline Press",
            "muscle_group": "chest",
            "level": "beginner",
            "equipment": "barbell",
            "mechanics": "compound",
            "force": "push",
        },
    ])

    recommendations = recommend_exercises(
        exercises=exercises,
        target_muscle_group="chest",
        experience_level="beginner",
        available_equipment=["barbell"],
        exercise_mechanics="compound",
        force_type="push",
        top_k=2,
    )

    # Asking for 2 recommendations should return exactly 2 rows.
    if len(recommendations) != 2:
        return show_result(
            False,
            f"Expected 2 recommendations but got {len(recommendations)}.",
        )

    # The rank column should count from 1.
    if list(recommendations["rank"]) != [1, 2]:
        return show_result(
            False,
            f"Expected ranks [1, 2] but got {list(recommendations['rank'])}.",
        )

    # All three scores are tied, so the two lowest exercise IDs should be returned.
    if list(recommendations["exercise_id"]) != ["EX001", "EX002"]:
        return show_result(
            False,
            "Expected tied exercises to be returned in exercise_id order but got "
            f"{list(recommendations['exercise_id'])}.",
        )

    return show_result(True)


# Main

def main() -> None:
    """Run every test in turn and print a short summary at the end."""
    print("MODEL 1 TESTING")
    print("===============")
    print()

    # The tests are listed here so they can be run one after another.
    tests = [
        test_text_cleaning,
        test_dataset_loading,
        test_missing_column,
        test_feature_encoding,
        test_feature_weighting,
        test_exact_filtering,
        test_any_option,
        test_recommendation_ranking,
        test_same_profile_same_results,
        test_top_k_and_order,
    ]

    # Each test returns True or False and the results are collected in a list
    # so they can be counted at the end.
    results = []
    for test in tests:
        results.append(test())
        print()

    total_tests = len(results)
    passed_tests = results.count(True)
    failed_tests = results.count(False)

    print("TEST SUMMARY")
    print("============")
    print(f"Total tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {failed_tests}")


if __name__ == "__main__":
    main()
