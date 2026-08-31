"""
Tests for Model 2, the TF-IDF exercise recommender.

The tests check the main steps of the model: loading the dataset, building the
TF-IDF matrix, cleaning the user's inputs, weighting features by repeating them
in the query text, strict filtering and the ranking itself.

Nothing in the recommender is rewritten here. Every test uses the same
ExerciseRecommender class the Streamlit app uses, so a passing test says
something about the real model and not about a copy of it.

Run from the project folder:

    python scripts/testing.py

Each test prints PASSED or FAILED, and a short summary is printed at the end.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pandas as pd


# File paths

# The project folder is added to the import path so the src package can be
# found when this script is run on its own.
PROJECT_DIRECTORY = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIRECTORY))

from src.recommender import ExerciseRecommender   # noqa: E402


# Helper functions

def show_result(passed: bool, reason: str = "") -> bool:
    """Print the outcome of one test and return it so it can be counted."""
    if passed:
        print("PASSED")
    else:
        print("FAILED")
        print(reason)
    return passed


def build_recommender(dataframe: pd.DataFrame, file_name: str) -> ExerciseRecommender:
    """Save a small dataset to a file and build a recommender from it.

    Model 2 loads its data inside the class, so it needs a real CSV file rather
    than a dataframe. A temporary folder is used so the project folder stays clean.
    """
    path = Path(tempfile.gettempdir()) / file_name
    dataframe.to_csv(path, index=False)
    return ExerciseRecommender(path)


def make_small_dataset() -> pd.DataFrame:
    """Build three exercises that differ from each other in an obvious way.

    Two work the chest and one works the back, and they use different levels,
    equipment, mechanics and force types, so the filters are easy to check.
    """
    return pd.DataFrame([
        {
            "exercise_id": "EX001",
            "exercise_name": "Bench Press",
            "difficulty_score": 3,
            "level": "Beginner",
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
            "exercise_name": "Dumbbell Fly",
            "difficulty_score": 4,
            "level": "Advanced",
            "equipment": "Dumbbell",
            "muscle_group": "Chest",
            "muscles_worked": "Pectoralis Major",
            "mechanics": "Isolated",
            "force": "Push",
            "preparation": "Lie on the bench holding dumbbells.",
            "execution": "Open and close the arms in an arc.",
        },
        {
            "exercise_id": "EX003",
            "exercise_name": "Lat Pulldown",
            "difficulty_score": 2,
            "level": "Beginner",
            "equipment": "Cable",
            "muscle_group": "Back",
            "muscles_worked": "Latissimus Dorsi",
            "mechanics": "Compound",
            "force": "Pull",
            "preparation": "Sit at the machine.",
            "execution": "Pull the bar to the chest.",
        },
    ])


# Test 1: dataset loading and TF-IDF creation

def test_loading_and_tfidf() -> bool:
    """Check that a small dataset loads and the TF-IDF matrix is built from it."""
    print("Test 1: Dataset loading and TF-IDF creation")

    recommender = build_recommender(make_small_dataset(), "model2_test_exercises.csv")

    # All three exercises should survive loading.
    if len(recommender.data) != 3:
        return show_result(False, f"Expected 3 exercises but got {len(recommender.data)}.")

    # The model builds one text document per exercise before fitting TF-IDF.
    if "model_features" not in recommender.data.columns:
        return show_result(False, "Expected a model_features column to be created.")

    # One TF-IDF row per exercise. If these numbers disagree, the scores would be
    # matched to the wrong exercises.
    if recommender.feature_matrix.shape[0] != 3:
        return show_result(
            False,
            f"Expected 3 TF-IDF rows but got {recommender.feature_matrix.shape[0]}.",
        )

    # The vectorizer should have learned some vocabulary from the feature text.
    if recommender.feature_matrix.shape[1] == 0:
        return show_result(False, "Expected the TF-IDF matrix to have some terms.")

    return show_result(True)


# Test 2: missing required column

def test_missing_column() -> bool:
    """Check that the model reports a clear error when a column is missing."""
    print("Test 2: Missing required column")

    # Remove the force column, which the model needs to build its feature text.
    broken_dataset = make_small_dataset().drop(columns=["force"])
    path = Path(tempfile.gettempdir()) / "model2_test_missing_column.csv"
    broken_dataset.to_csv(path, index=False)

    # The model validates the dataset when it is created, so the error should
    # appear here rather than later during ranking.
    try:
        ExerciseRecommender(path)
        return show_result(False, "Expected a ValueError but the dataset loaded without one.")
    except ValueError as error:
        # The message should name the missing column so the problem is easy to fix.
        if "force" not in str(error):
            return show_result(False, "Expected the error message to name the missing column.")
        return show_result(True)


# Test 3: user input normalisation

def test_input_normalisation() -> bool:
    """Check that user inputs are tidied before the model uses them."""
    print("Test 3: User input normalisation")

    # Extra spaces should be removed, blank entries dropped, and a repeated
    # choice kept only once. The order the user chose is preserved.
    cleaned = ExerciseRecommender._normalise_multi_value(
        ["  Barbell  ", "Dumbbell", "", "Barbell"]
    )

    if cleaned != ["Barbell", "Dumbbell"]:
        return show_result(False, f"Expected ['Barbell', 'Dumbbell'] but got {cleaned}.")

    # No selection at all should be treated as no preference, not as an error.
    if ExerciseRecommender._normalise_multi_value(None) != []:
        return show_result(False, "Expected an empty list when nothing is selected.")

    # Capital letters are handled by the TF-IDF vectorizer rather than by the
    # cleaning step, because the vectorizer is set up with lowercase=True.
    # The same text in capitals should therefore give the same query vector.
    recommender = build_recommender(make_small_dataset(), "model2_test_exercises.csv")

    upper_vector = recommender.vectorizer.transform(["muscle group CHEST"]).toarray().tolist()
    lower_vector = recommender.vectorizer.transform(["muscle group chest"]).toarray().tolist()

    # This confirms the words were actually recognised, so the check above is
    # not simply comparing two empty vectors.
    if sum(lower_vector[0]) == 0:
        return show_result(False, "Expected the test words to be found in the TF-IDF vocabulary.")

    if upper_vector != lower_vector:
        return show_result(False, "Expected capital letters to be treated the same as lower case.")

    return show_result(True)


# Test 4: feature weighting through repeated text

def test_feature_weighting() -> bool:
    """Check that stronger features are repeated more often in the user query."""
    print("Test 4: Feature weighting")

    recommender = build_recommender(make_small_dataset(), "model2_test_exercises.csv")

    # Build the text query the model would send to TF-IDF for this profile.
    query = recommender._build_user_query(
        muscle_groups=["Chest"],
        level="Beginner",
        equipment=["Barbell"],
        mechanics="Compound",
        force="Push",
        specific_muscles="",
    )

    # Model 2 weights features by repeating them in the text. Muscle group is
    # the strongest signal at 5 repeats, then level and equipment at 3, then
    # mechanics and force at 2.
    if query.count("muscle group Chest") != 5:
        return show_result(
            False,
            f"Expected muscle group 5 times but got {query.count('muscle group Chest')}.",
        )

    if query.count("level Beginner") != 3:
        return show_result(
            False,
            f"Expected level 3 times but got {query.count('level Beginner')}.",
        )

    if query.count("equipment Barbell") != 3:
        return show_result(
            False,
            f"Expected equipment 3 times but got {query.count('equipment Barbell')}.",
        )

    if query.count("force Push") != 2:
        return show_result(
            False,
            f"Expected force 2 times but got {query.count('force Push')}.",
        )

    # The important comparison: the muscle group must appear more often than a
    # lower weighted feature, otherwise the weighting has no effect.
    if query.count("muscle group Chest") <= query.count("mechanics Compound"):
        return show_result(False, "Expected the muscle group to be repeated more than mechanics.")

    return show_result(True)


# Test 5: "Any" is not added to the TF-IDF query

def test_any_excluded_from_query() -> bool:
    """Check that 'Any' never reaches the text used for TF-IDF similarity."""
    print("Test 5: Any option is excluded from TF-IDF query")

    recommender = build_recommender(make_small_dataset(), "model2_test_exercises.csv")

    # "Any" means the user has no preference,
    # so it should not be included in the TF-IDF query.
    query = recommender._build_user_query(
        muscle_groups=["Chest"],
        level="Any",
        equipment=[],
        mechanics="Any",
        force="Any",
        specific_muscles="",
    )

    # If "any" were included it would be treated as a real search word and
    # could match exercises for no good reason.
    if "any" in query.lower():
        return show_result(False, f"Did not expect the word 'any' in the query but got: {query}")

    # The labels should be missing too, because the whole part is skipped.
    if "level" in query or "mechanics" in query or "force" in query:
        return show_result(
            False,
            f"Expected no level, mechanics or force text in the query but got: {query}",
        )

    # The muscle group was a real choice, so it should still be there.
    if "muscle group Chest" not in query:
        return show_result(False, "Expected the selected muscle group to stay in the query.")

    return show_result(True)


# Test 6: strict filtering

def test_strict_filtering() -> bool:
    """Check that strict filters keep only exercises matching every choice."""
    print("Test 6: Strict filtering")

    recommender = build_recommender(make_small_dataset(), "model2_test_exercises.csv")

    # Only the first exercise matches all five of these choices.
    # top_n is set higher than the dataset so the filter does the work, not the cut-off.
    results = recommender.recommend(
        muscle_groups=["Chest"],
        level="Beginner",
        equipment=["Barbell"],
        mechanics="Compound",
        force="Push",
        top_n=10,
        strict_filters=True,
    )

    if len(results) != 1:
        return show_result(False, f"Expected 1 exercise but got {len(results)}.")

    if results.iloc[0]["exercise_id"] != "EX001":
        return show_result(
            False,
            f"Expected EX001 but got {results.iloc[0]['exercise_id']}.",
        )

    # A filter combination that nothing satisfies should return no rows rather
    # than an error. The app then relaxes the filter and tells the user.
    empty_results = recommender.recommend(
        muscle_groups=["Back"],
        level="Advanced",
        equipment=["Barbell"],
        mechanics="Isolated",
        force="Push",
        top_n=10,
        strict_filters=True,
    )

    if len(empty_results) != 0:
        return show_result(False, "Expected no exercises for an impossible filter combination.")

    return show_result(True)


# Test 7: "Any" and empty equipment do not filter

def test_any_and_empty_equipment() -> bool:
    """Check that 'Any' and an empty equipment list do not remove exercises."""
    print("Test 7: Any and empty equipment filtering")

    recommender = build_recommender(make_small_dataset(), "model2_test_exercises.csv")

    # Model 2 always requires at least one muscle group, so both groups in the
    # small dataset are selected here. Every other category is left open, which
    # means nothing should be removed.
    results = recommender.recommend(
        muscle_groups=["Chest", "Back"],
        level="Any",
        equipment=[],
        mechanics="Any",
        force="Any",
        top_n=10,
        strict_filters=True,
    )

    # The IDs are sorted before comparing, because this test is about which
    # exercises survive the filter, not about the ranking order.
    if sorted(results["exercise_id"]) != ["EX001", "EX002", "EX003"]:
        return show_result(
            False,
            f"Expected all 3 exercises to be kept but got {sorted(results['exercise_id'])}.",
        )

    # Now only the level is set. The two beginner exercises use different
    # equipment, and the empty equipment list should not remove either of them.
    beginner_results = recommender.recommend(
        muscle_groups=["Chest", "Back"],
        level="Beginner",
        equipment=[],
        mechanics="Any",
        force="Any",
        top_n=10,
        strict_filters=True,
    )

    if sorted(beginner_results["exercise_id"]) != ["EX001", "EX003"]:
        return show_result(
            False,
            "Expected the two beginner exercises to be kept when equipment is empty "
            f"but got {sorted(beginner_results['exercise_id'])}.",
        )

    return show_result(True)


# Test 8: recommendation ranking

def test_recommendation_ranking() -> bool:
    """Check that a close match to the profile is ranked above a poor match."""
    print("Test 8: Recommendation ranking")

    # EX001 matches the profile below on every feature.
    # EX002 matches none of them.
    exercises = pd.DataFrame([
        {
            "exercise_id": "EX001",
            "exercise_name": "Bench Press",
            "difficulty_score": 3,
            "level": "Beginner",
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
            "exercise_name": "Cable Row",
            "difficulty_score": 4,
            "level": "Advanced",
            "equipment": "Cable",
            "muscle_group": "Back",
            "muscles_worked": "Latissimus Dorsi",
            "mechanics": "Isolated",
            "force": "Pull",
            "preparation": "Sit at the machine.",
            "execution": "Pull the handle to the waist.",
        },
    ])

    recommender = build_recommender(exercises, "model2_test_ranking.csv")

    # Strict filtering is switched off on purpose. The poor match would
    # otherwise be removed before scoring, and this test is about the ranking.
    results = recommender.recommend(
        muscle_groups=["Chest"],
        level="Beginner",
        equipment=["Barbell"],
        mechanics="Compound",
        force="Push",
        top_n=2,
        strict_filters=False,
    )

    if len(results) != 2:
        return show_result(False, f"Expected both exercises to be scored but got {len(results)}.")

    # The strong match should be first in the list.
    if results.iloc[0]["exercise_id"] != "EX001":
        return show_result(
            False,
            "Expected the strongly matching exercise to rank above the poorly matching exercise.",
        )

    # The matching exercise should have a higher cosine similarity score,
    # which is what puts it first.
    top_score = results.iloc[0]["similarity_score"]
    second_score = results.iloc[1]["similarity_score"]

    if top_score <= second_score:
        return show_result(
            False,
            f"Expected the top score {top_score} to be higher than {second_score}.",
        )

    return show_result(True)


# Test 9: the same user profile gives the same recommendations

def test_same_profile_same_results() -> bool:
    """Check that the same profile always produces the same recommendations."""
    print("Test 9: Same profile gives same recommendations")

    recommender = build_recommender(make_small_dataset(), "model2_test_exercises.csv")

    # Build the same user profile twice.
    # The recommender should return the same results each time.
    first_run = recommender.recommend(
        muscle_groups=["Chest", "Back"],
        level="Beginner",
        equipment=["Barbell"],
        mechanics="Compound",
        force="Push",
        top_n=3,
        strict_filters=False,
    )

    second_run = recommender.recommend(
        muscle_groups=["Chest", "Back"],
        level="Beginner",
        equipment=["Barbell"],
        mechanics="Compound",
        force="Push",
        top_n=3,
        strict_filters=False,
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


# Test 10: Top-N and deterministic tie order

def test_top_n_and_tie_order() -> bool:
    """Check that Top-N returns the right number of rows in a consistent order."""
    print("Test 10: Top-N and deterministic tie ordering")

    # These three exercises are identical apart from the ID, so their feature
    # text is identical and their similarity scores are tied. They are listed
    # out of order on purpose, which shows the tie is settled by exercise_id
    # and not by the order of the rows in the file.
    exercises = pd.DataFrame([
        {
            "exercise_id": "EX003",
            "exercise_name": "Bench Press",
            "difficulty_score": 3,
            "level": "Beginner",
            "equipment": "Barbell",
            "muscle_group": "Chest",
            "muscles_worked": "Pectoralis Major",
            "mechanics": "Compound",
            "force": "Push",
            "preparation": "Lie on the bench.",
            "execution": "Press the bar upwards.",
        },
        {
            "exercise_id": "EX001",
            "exercise_name": "Bench Press",
            "difficulty_score": 3,
            "level": "Beginner",
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
            "exercise_name": "Bench Press",
            "difficulty_score": 3,
            "level": "Beginner",
            "equipment": "Barbell",
            "muscle_group": "Chest",
            "muscles_worked": "Pectoralis Major",
            "mechanics": "Compound",
            "force": "Push",
            "preparation": "Lie on the bench.",
            "execution": "Press the bar upwards.",
        },
    ])

    recommender = build_recommender(exercises, "model2_test_ties.csv")

    results = recommender.recommend(
        muscle_groups=["Chest"],
        level="Beginner",
        equipment=["Barbell"],
        mechanics="Compound",
        force="Push",
        top_n=2,
        strict_filters=True,
    )

    # Asking for 2 recommendations should return exactly 2 rows.
    if len(results) != 2:
        return show_result(False, f"Expected 2 recommendations but got {len(results)}.")

    # The three scores should genuinely be tied, which is the situation this
    # test is designed to check.
    scores = [round(score, 10) for score in results["similarity_score"]]
    if scores[0] != scores[1]:
        return show_result(False, f"Expected the two scores to be tied but got {scores}.")

    # Model 2 does not add a rank column, so the returned order is the ranking.
    # With tied scores the two lowest exercise IDs should come back, in order.
    if list(results["exercise_id"]) != ["EX001", "EX002"]:
        return show_result(
            False,
            "Expected tied exercises to be returned in exercise_id order but got "
            f"{list(results['exercise_id'])}.",
        )

    return show_result(True)


# Main

def main() -> None:
    """Run every test in turn and print a short summary at the end."""
    print("MODEL 2 TESTING")
    print("===============")
    print()

    # The tests are listed here so they can be run one after another.
    tests = [
        test_loading_and_tfidf,
        test_missing_column,
        test_input_normalisation,
        test_feature_weighting,
        test_any_excluded_from_query,
        test_strict_filtering,
        test_any_and_empty_equipment,
        test_recommendation_ranking,
        test_same_profile_same_results,
        test_top_n_and_tie_order,
    ]

    # Each test returns True or False, and the results are collected in a list
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
