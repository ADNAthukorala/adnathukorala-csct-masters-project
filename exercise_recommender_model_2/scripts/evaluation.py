"""
Offline evaluation of Model 2, the TF-IDF exercise recommender.

Each of the 50 user profiles is passed to the recommender, and the Top 8
exercises it returns are compared with the 10 exercises that the ground truth
lists as relevant for that user. Two things are measured:

  1. Ranking quality  - Precision@8, Recall@8, F1@8 and NDCG@8
  2. Feature matching - whether the recommendations respect the preferences
                        the user stated

The strict category filter is switched off here, which is also the default
setting in the app. Nothing is removed in advance, so the ranking depends only
on the similarity score. This keeps the test fair, because a hard filter would
do part of the model's work for it.

The method is identical to the Model 1 evaluation: same 50 profiles, same
ground truth, same cut-off, same metrics and same output files. Only the way
the recommender is called differs, because Model 2 exposes a class rather than
a single function. Keeping everything else fixed means the two sets of numbers
can be placed side by side.

Run from the project folder:

    python scripts/evaluation.py

Three CSV files are written into results/ and a summary is printed.
"""

from __future__ import annotations

# math is imported for log2, which is used in the NDCG discount.
import math
import sys
from pathlib import Path

import pandas as pd


# File paths

# The project folder is added to the import path so that the src package can
# be found when this script is run on its own.
PROJECT_DIRECTORY = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIRECTORY))

# The recommender is imported rather than rewritten here, so the evaluation
# measures the model the application actually uses.
from src.recommender import ExerciseRecommender          # noqa: E402

DATA_DIRECTORY = PROJECT_DIRECTORY / "data" / "processed"
EXERCISES_PATH = DATA_DIRECTORY / "cleaned_exercise_dataset.csv"
PROFILES_PATH = DATA_DIRECTORY / "generated_user_profiles.csv"
GROUND_TRUTH_PATH = DATA_DIRECTORY / "ground_truth_evaluation_dataset.csv"

RESULTS_DIRECTORY = PROJECT_DIRECTORY / "results"


# Settings

# Every metric considers only the first 8 recommendations.
K = 8

# The strict filter is left off, matching the app's default and the conditions
# used for Model 1.
STRICT_FILTERS = False

# The five features the model scores on, written as they appear in the dataset.
FEATURE_NAMES = ["muscle_group", "level", "equipment", "mechanics", "force"]


# PART 0 - Small text helpers

# Model 2 keeps all of its logic inside one module and has no text_utils
# package, so the two helpers the evaluation needs are defined here. They
# behave the same way as the versions used in the Model 1 evaluation.

def clean_text(value) -> str:
    """
    Put a value into a plain, comparable form.

    Text is lowercased and the surrounding spaces are removed, so "Body Weight"
    and "body weight " are treated as the same value. Missing entries become an
    empty string rather than raising an error.
    """
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""

    return " ".join(str(value).split()).lower()


def split_values(value) -> list[str]:
    """
    Turn one comma-separated field into a list of separate values.

    The equipment column of a profile holds several items in a single string,
    for example "Barbell, Cable, Dumbbell". Empty pieces are dropped and the
    original spelling is kept, because the recommender compares these strings
    against the dataset.
    """
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return []

    return [piece.strip() for piece in str(value).split(",") if piece.strip()]


# PART 1 - Loading the data

def load_all_data():
    """Load the recommender and the two CSV files needed by the evaluation."""
    # Model 2 loads and validates the dataset inside its own constructor, and
    # fits the TF-IDF vectoriser at the same time. Building it once here means
    # the evaluation sees the exercises in exactly the same state as the app,
    # and the vectoriser is not refitted for every user.
    recommender = ExerciseRecommender(EXERCISES_PATH)

    # One row per user, 50 in total.
    profiles = pd.read_csv(PROFILES_PATH)

    # 10 ranked relevant exercises per user, so 500 rows. The utf-8-sig
    # encoding removes the hidden marker Excel writes at the start of a file,
    # which would otherwise hide the "user_id" column.
    ground_truth = pd.read_csv(GROUND_TRUTH_PATH, encoding="utf-8-sig")

    return recommender, profiles, ground_truth


# PART 2 - Ranking metrics

# The functions below share the same two inputs:
#   recommended_ids = the Top-8 exercise IDs returned by the model, in order
#   relevant_ids    = the exercise IDs the ground truth marks as relevant
def precision_at_k(recommended_ids, relevant_ids, k):
    """
    Precision@K = relevant items in the Top-K divided by K.

    It answers the question: out of the 8 exercises shown, how many were
    actually suitable? A value of 0.50 means 4 of the 8 were in the ground
    truth.
    """
    top_k = recommended_ids[:k]
    hits = sum(1 for item in top_k if item in relevant_ids)

    # Avoid division by zero.
    if k == 0:
        return 0.0

    return hits / k


def recall_at_k(recommended_ids, relevant_ids, k):
    """
    Recall@K = relevant items in the Top-K divided by the total number of
    relevant items.

    It answers the question: out of the 10 exercises the user should have
    been offered, how many did the model find?

    Note that there are 10 relevant items but only 8 places, so the highest
    score possible here is 8/10 = 0.800 and not 1.000.
    """
    top_k = recommended_ids[:k]
    hits = sum(1 for item in top_k if item in relevant_ids)

    # The measure is undefined when a user has no relevant items.
    if len(relevant_ids) == 0:
        return 0.0

    return hits / len(relevant_ids)


def f1_at_k(precision, recall):
    """
    F1@K = the harmonic mean of precision and recall.

    It combines the two measures and gives a low score whenever one of them
    is much weaker than the other.
    """
    # Two zero values would cause a division by zero.
    if precision + recall == 0:
        return 0.0

    return (2 * precision * recall) / (precision + recall)


def ndcg_at_k(recommended_ids, relevance_scores, k):
    """
    NDCG@K - Normalised Discounted Cumulative Gain.

    Precision and recall only ask whether a suitable exercise was found.
    NDCG also considers where it was placed, so a good exercise in position 1
    scores higher than the same exercise in position 8.

    The calculation has three stages:
      1. Gain  - relevance_scores gives each ID a relevance value
                 (ground-truth rank 1 = 10 points, rank 10 = 1, absent = 0).
      2. DCG   - each gain is divided by log2(position + 1), so items further
                 down the list count for less.
      3. Norm  - divide by the DCG of the perfect ordering (IDCG). This puts
                 the result on a 0 to 1 scale, where 1.000 is the ideal
                 ranking.
    """
    # Stage 2: applied to the list the model actually returned.
    dcg = 0.0
    for position, exercise_id in enumerate(recommended_ids[:k], start=1):
        # Exercises missing from the ground truth add nothing to the score.
        gain = relevance_scores.get(exercise_id, 0.0)
        dcg += gain / math.log2(position + 1)

    # Stage 3: The ideal list holds the same gains in descending order.
    ideal_gains = sorted(relevance_scores.values(), reverse=True)

    idcg = 0.0
    for position, gain in enumerate(ideal_gains[:k], start=1):
        idcg += gain / math.log2(position + 1)

    # NDCG is undefined when nothing is relevant.
    if idcg == 0:
        return 0.0

    return dcg / idcg


# PART 3 - Feature match check

def check_feature_matches(exercise_row, profile_row):
    """
    Compare one exercise with one profile and return five true or false
    answers.

    This asks a different question from precision. Precision asks whether an
    exercise was on the approved list, while this asks whether it respects
    what the user requested. An exercise can pass one test and fail the
    other, so both are worth reporting.

    clean_text is applied throughout so that capital letters or extra spaces
    never cause a false mismatch.
    """
    # The preferences stated by the user, in normalised form.
    wanted_muscle_group = clean_text(profile_row["preferred_target_muscle_group"])
    wanted_level = clean_text(profile_row["experience_level"])
    wanted_mechanics = clean_text(profile_row["preferred_mechanics"])
    wanted_force = clean_text(profile_row["preferred_force_type"])

    # Equipment is treated differently because a user may own several items,
    # so the test is whether the exercise's equipment appears in that list.
    owned_equipment = {
        clean_text(item) for item in split_values(profile_row["available_equipments"])
    }

    return {
        "muscle_group": clean_text(exercise_row["muscle_group"]) == wanted_muscle_group,
        "level": clean_text(exercise_row["level"]) == wanted_level,
        "equipment": clean_text(exercise_row["equipment"]) in owned_equipment,
        "mechanics": clean_text(exercise_row["mechanics"]) == wanted_mechanics,
        "force": clean_text(exercise_row["force"]) == wanted_force,
    }


# PART 4 - Evaluating one user

def evaluate_one_user(recommender, profile_row, user_ground_truth):
    """
    Run the recommender for a single user and score the result.

    Returns that user's scores, which form one row of the per-user report,
    together with one detail row for each recommended exercise.
    """
    user_id = profile_row["user_id"]

    # 4a. The equipment field holds several items in one string, so it is
    # split into the list that recommend expects.
    available_equipment = split_values(profile_row["available_equipments"])

    # 4b. Ask the model for its Top 8. Nothing is removed beforehand. The
    # muscle group is wrapped in a list because Model 2 accepts several, and
    # specific_muscles is left empty because the profiles do not record any
    # free-text muscle request.
    recommendations = recommender.recommend(
        muscle_groups=[profile_row["preferred_target_muscle_group"]],
        level=profile_row["experience_level"],
        equipment=available_equipment,
        mechanics=profile_row["preferred_mechanics"],
        force=profile_row["preferred_force_type"],
        specific_muscles="",
        top_n=K,
        strict_filters=STRICT_FILTERS,
    )

    # The order is kept because NDCG depends on it.
    recommended_ids = recommendations["exercise_id"].tolist()

    # 4c. This user's ground truth: the IDs that count as relevant
    relevant_ids = set(user_ground_truth["exercise_id"])

    # The graded version used by NDCG. Rank 1 is the best, so it is
    # turned into the largest gain. With 10 items: rank 1 -> 10, rank 10 -> 1.
    number_of_relevant = len(user_ground_truth)
    relevance_scores = {
        row["exercise_id"]: (number_of_relevant + 1) - row["rank"]
        for _, row in user_ground_truth.iterrows()
    }

    # 4d. Ranking metrics for this user.
    precision = precision_at_k(recommended_ids, relevant_ids, K)
    recall = recall_at_k(recommended_ids, relevant_ids, K)
    f1 = f1_at_k(precision, recall)
    ndcg = ndcg_at_k(recommended_ids, relevance_scores, K)

    # 4e. Feature matching: test all five features on every recommendation
    # and keep a running count of the matches.
    feature_hit_counts = {name: 0 for name in FEATURE_NAMES}
    detail_rows = []

    for position, (_, exercise_row) in enumerate(recommendations.iterrows(), start=1):
        matches = check_feature_matches(exercise_row, profile_row)

        for name in FEATURE_NAMES:
            if matches[name]:
                feature_hit_counts[name] += 1

        # Stored so that individual recommendations can be inspected later.
        detail_rows.append({
            "user_id": user_id,
            "rank": position,
            "exercise_id": exercise_row["exercise_id"],
            "exercise_name": exercise_row["exercise_name"],
            "similarity_score": round(exercise_row["similarity_score"], 4),
            # The percentage shown in the app, which is what the model sorts on.
            "recommendation_score": exercise_row["recommendation_score"],
            # Did this exercise appear in the ground truth at all?
            "in_ground_truth": exercise_row["exercise_id"] in relevant_ids,
            "muscle_group_match": matches["muscle_group"],
            "level_match": matches["level"],
            "equipment_match": matches["equipment"],
            "mechanics_match": matches["mechanics"],
            "force_match": matches["force"],
            # How many of the five features this single exercise satisfied.
            "features_matched_out_of_5": sum(matches.values()),
        })

    # The counts become rates: of the 8 shown, what share matched on each feature?
    number_recommended = len(recommendations)
    user_scores = {
        "user_id": user_id,
        "precision_at_8": round(precision, 4),
        "recall_at_8": round(recall, 4),
        "f1_at_8": round(f1, 4),
        "ndcg_at_8": round(ndcg, 4),
    }

    for name in FEATURE_NAMES:
        user_scores[f"{name}_match_rate"] = round(
            feature_hit_counts[name] / number_recommended, 4
        )

    return user_scores, detail_rows


# PART 5 - Running everything and reporting

def main():
    """Run the complete evaluation and print the results."""
    print("=" * 70)
    print("MODEL 2 EVALUATION - TF-IDF Exercise Recommender")
    print("Strict category filter: OFF (all exercises stay eligible)")
    print("=" * 70)

    # 5a. Load the model and the two datasets.
    recommender, profiles, ground_truth = load_all_data()

    print(f"\nExercises loaded      : {len(recommender.data)}")
    print(f"User profiles loaded  : {len(profiles)}")
    print(f"Ground-truth rows     : {len(ground_truth)}")
    print(f"Cut-off K             : {K}")

    # 5b. Score every user in turn.
    all_user_scores = []
    all_detail_rows = []

    for _, profile_row in profiles.iterrows():
        user_id = profile_row["user_id"]

        # The ground-truth rows belonging to this user only.
        user_ground_truth = ground_truth[ground_truth["user_id"] == user_id]

        # A user without ground truth cannot be scored, so they are skipped.
        if len(user_ground_truth) == 0:
            print(f"  Skipping {user_id}: no ground-truth rows found.")
            continue

        user_scores, detail_rows = evaluate_one_user(
            recommender, profile_row, user_ground_truth
        )

        all_user_scores.append(user_scores)
        all_detail_rows.extend(detail_rows)

    per_user_results = pd.DataFrame(all_user_scores)
    detailed_results = pd.DataFrame(all_detail_rows)

    # 5c. Average across users. Every user counts equally, so this is a plain
    # mean of the per user scores rather than a total over all recommendations.
    print("\n" + "=" * 70)
    print(f"RANKING METRICS - averaged over {len(per_user_results)} users")
    print("=" * 70)

    mean_precision = per_user_results["precision_at_8"].mean()
    mean_recall = per_user_results["recall_at_8"].mean()
    mean_f1 = per_user_results["f1_at_8"].mean()
    mean_ndcg = per_user_results["ndcg_at_8"].mean()

    # Each score is a proportion, so multiplying by 100 gives the percentage.
    # Both forms are shown in decimals and percentages for easier comparison with other models.
    print(f"  Precision@8 : {mean_precision:.4f}   ({mean_precision * 100:.2f}%)")
    print(f"  Recall@8    : {mean_recall:.4f}   ({mean_recall * 100:.2f}%)")
    print(" " * 16 + "maximum possible is 0.8000 (80.00%), because")
    print(" " * 16 + "there are 10 relevant exercises but only 8 slots")
    print(f"  F1@8        : {mean_f1:.4f}   ({mean_f1 * 100:.2f}%)")
    print(f"  NDCG@8      : {mean_ndcg:.4f}   ({mean_ndcg * 100:.2f}%)")

    # 5d. Feature match rates.
    print("\n" + "=" * 70)
    print("FEATURE MATCH RATES - share of the 8 recommendations that matched")
    print("=" * 70)

    # Readable labels, used only for printing.
    display_names = {
        "muscle_group": "Muscle group",
        "level": "Level",
        "equipment": "Equipment",
        "mechanics": "Mechanics",
        "force": "Force",
    }

    feature_averages = {}
    for name in FEATURE_NAMES:
        average = per_user_results[f"{name}_match_rate"].mean()
        feature_averages[name] = average
        print(f"  {display_names[name]:<14}: {average:.4f}   ({average * 100:.2f}%)")

    # A single headline figure for feature agreement.
    overall_feature_match = sum(feature_averages.values()) / len(FEATURE_NAMES)
    print(f"  {'-' * 40}")
    print(
        f"  {'Overall':<14}: {overall_feature_match:.4f}   "
        f"({overall_feature_match * 100:.2f}%)"
    )

    # 5e. Save the output files.
    RESULTS_DIRECTORY.mkdir(exist_ok=True)

    per_user_path = RESULTS_DIRECTORY / "model2_evaluation_per_user.csv"
    detailed_path = RESULTS_DIRECTORY / "model2_evaluation_detailed.csv"
    summary_path = RESULTS_DIRECTORY / "model2_evaluation_summary.csv"

    # The per user file also gets a percentage version of every score column.
    # Everything except user_id is a proportion, so each is simply x100.
    per_user_output = per_user_results.copy()
    for column in per_user_results.columns:
        if column != "user_id":
            per_user_output[f"{column}_percent"] = (
                per_user_results[column] * 100
            ).round(2)

    per_user_output.to_csv(per_user_path, index=False)
    detailed_results.to_csv(detailed_path, index=False)

    # A summary with one row per metric.
    summary_rows = [
        ("Precision@8", mean_precision),
        ("Recall@8", mean_recall),
        ("F1@8", mean_f1),
        ("NDCG@8", mean_ndcg),
        ("Muscle group match", feature_averages["muscle_group"]),
        ("Level match", feature_averages["level"]),
        ("Equipment match", feature_averages["equipment"]),
        ("Mechanics match", feature_averages["mechanics"]),
        ("Force match", feature_averages["force"]),
        ("Overall feature match", overall_feature_match),
    ]

    # Each metric is written twice, as a decimal and as a percentage.
    summary_table = pd.DataFrame([
        {
            "metric": name,
            "score": round(value, 4),
            "score_percent": round(value * 100, 2),
        }
        for name, value in summary_rows
    ])
    summary_table.to_csv(summary_path, index=False)

    print("\n" + "=" * 70)
    print("FILES SAVED")
    print("=" * 70)
    print(f"  {summary_path}")
    print(f"  {per_user_path}")
    print(f"  {detailed_path}")
    print()

if __name__ == "__main__":
    main()
