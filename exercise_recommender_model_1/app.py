"""Streamlit web interface for the exercise recommender (Model 1).

This file is only the front end: it collects the user's choices, hands them to
the recommender in `src/recommender.py` and lays the results out on screen.
None of the recommendation logic lives here.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from src.data_loader import load_exercise_data
from src.filters import ANY_OPTION, apply_filters
from src.recommender import FEATURE_WEIGHTS, recommend_exercises
from src.text_utils import clean_text, get_unique_values


# Browser tab settings. A wide layout leaves room for the recommendation cards.
st.set_page_config(
    page_title="Exercise Recommendation System",
    page_icon="🏋️",
    layout="wide",
)


# The dataset path is built from the location of this file, so the app still finds
# the data no matter which folder it is started from.
PROJECT_DIRECTORY = Path(__file__).resolve().parent
DATASET_PATH = (
    PROJECT_DIRECTORY / "data" / "processed" / "cleaned_exercise_dataset.csv"
)


@st.cache_data(show_spinner=False)
def load_data_for_app(path: Path) -> pd.DataFrame:
    """Load exercise data once and cache it for faster Streamlit interaction."""
    return load_exercise_data(path)


def format_category(value) -> str:
    """Present a cleaned category value in a readable form for the interface."""
    text = str(value).strip()
    return text.title() if text else "Not available"


def build_match_summary(
    exercise: pd.Series,
    target_muscle_group: str,
    experience_level: str,
    available_equipment: list[str],
    exercise_mechanics: str,
    force_type: str,
) -> str:
    """Write a short sentence explaining why an exercise was suggested.

    This is purely for the user's benefit. It reads a result the recommender has
    already produced and cannot change the score or the position in the list.
    """
    matched: list[str] = []

    # Compare each preference against the exercise and note the ones that line up.
    if clean_text(exercise["muscle_group"]) == clean_text(target_muscle_group):
        matched.append(f"targets {format_category(exercise['muscle_group'])}")

    if experience_level != ANY_OPTION and clean_text(exercise["level"]) == clean_text(
        experience_level
    ):
        matched.append(f"matches {format_category(exercise['level'])} level")

    equipment_values = [clean_text(item) for item in available_equipment]
    if equipment_values and clean_text(exercise["equipment"]) in equipment_values:
        matched.append(f"uses {format_category(exercise['equipment'])}")

    if exercise_mechanics != ANY_OPTION and clean_text(
        exercise["mechanics"]
    ) == clean_text(exercise_mechanics):
        matched.append(f"is {format_category(exercise['mechanics']).lower()}")

    if force_type != ANY_OPTION and clean_text(exercise["force"]) == clean_text(
        force_type
    ):
        matched.append(f"uses a {format_category(exercise['force']).lower()} action")

    # An exercise can still rank well without matching any single preference exactly,
    # because the score reflects the profile as a whole.
    if not matched:
        return "Has the closest overall feature profile."

    return ", ".join(matched).capitalize() + "."


st.title("🏋️ Exercise Recommendation System")
st.caption(
    "Choose your training preferences to receive ranked exercise recommendations "
    "from the cleaned dataset."
)


# Load the dataset and stop with a useful message if loading fails.
try:
    exercises = load_data_for_app(DATASET_PATH)
except (FileNotFoundError, ValueError, pd.errors.ParserError) as error:
    st.error(f"The recommendation system could not be loaded: {error}")
    st.stop()


# Read all available category values directly from the cleaned dataset.
muscle_group_options = get_unique_values(exercises, "muscle_group")
level_options = ["Beginner", "Intermediate", "Advanced"]
equipment_options = get_unique_values(exercises, "equipment")
mechanics_options = get_unique_values(exercises, "mechanics")
force_options = get_unique_values(exercises, "force")


# Display small dataset indicators so the user understands the available coverage.
metric_one, metric_two, metric_three = st.columns(3)
metric_one.metric("Exercises", f"{len(exercises):,}")
metric_two.metric("Muscle groups", len(muscle_group_options))
metric_three.metric("Equipment types", len(equipment_options))


# Keep all user controls inside one form to avoid unnecessary reruns.
with st.form("recommendation_form"):
    st.subheader("Your preferences")

    first_column, second_column = st.columns(2)
    with first_column:
        # Allow exactly one target muscle group to keep the request focused.
        target_muscle_group = st.selectbox(
            "Target muscle group",
            options=muscle_group_options,
            index=(
                muscle_group_options.index("Chest")
                if "Chest" in muscle_group_options
                else 0
            ),
            help="Select the main muscle group you want to train.",
        )
        experience_level = st.selectbox(
            "Experience level",
            options=[ANY_OPTION] + level_options,
            index=0,
        )
        available_equipment = st.multiselect(
            "Available equipment",
            options=equipment_options,
            help="Leave empty to allow all equipment types.",
        )

    # The second row contains the optional movement controls.
    with second_column:
        exercise_mechanics = st.selectbox(
            "Exercise mechanics",
            options=[ANY_OPTION] + mechanics_options,
            index=0,
        )
        force_type = st.selectbox(
            "Force type",
            options=[ANY_OPTION] + force_options,
            index=0,
        )

    # Let the user control both result count and filter behaviour.
    result_column, filter_column = st.columns(2)
    with result_column:
        top_k = st.slider(
            "Number of recommendations",
            min_value=3,
            max_value=15,
            value=5,
            step=1,
        )
    with filter_column:
        # Off by default, so the user sees the closest overall matches first and
        # can then decide to tighten things up.
        strict_filters = st.checkbox(
            "Require exact matches for selected categories",
            value=False,
            help=(
                "When selected, muscle group, level, equipment, mechanics and force "
                "are used as hard filters. The app relaxes them automatically if no "
                "exercise matches."
            ),
        )

    # A form submit button triggers recommendation generation.
    submitted = st.form_submit_button(
        "Recommend exercises",
        type="primary",
        use_container_width=True,
    )


# Generate and display results only after the form has been submitted.
if submitted:
    # Apply the simple category filter before the recommender runs.
    candidate_exercises = exercises
    exact_matching_applied = False

    if strict_filters:
        filtered_exercises = apply_filters(
            exercises=exercises,
            target_muscle_group=target_muscle_group,
            experience_level=experience_level,
            available_equipment=available_equipment,
            exercise_mechanics=exercise_mechanics,
            force_type=force_type,
        )

        # Automatically provide useful alternatives when a strict combination is empty.
        if filtered_exercises.empty:
            st.info(
                "No exercise matched every selected category, so the search was "
                "widened to the whole dataset. The exercises below are the closest "
                "overall matches and may target another muscle group."
            )
        else:
            candidate_exercises = filtered_exercises
            exact_matching_applied = True

    # "Any" is sent to the model as empty text, which the encoder already reads as
    # "no preference", so that feature simply stops influencing the score.
    recommendations = recommend_exercises(
        exercises=candidate_exercises,
        target_muscle_group=target_muscle_group,
        experience_level=(
            "" if experience_level == ANY_OPTION else experience_level
        ),
        available_equipment=available_equipment,
        exercise_mechanics=(
            "" if exercise_mechanics == ANY_OPTION else exercise_mechanics
        ),
        force_type="" if force_type == ANY_OPTION else force_type,
        top_k=top_k,
    )

    # Show a heading before the ranked recommendation cards.
    st.divider()
    st.subheader("Recommended exercises")

    if recommendations.empty:
        st.error(
            "No exercises were available to recommend. Please change the selected "
            "preferences and try again."
        )
    else:
        # The cosine-similarity score is shown exactly as the model produced it.
        # Its highest possible value depends on how many preferences were chosen,
        # so exercises that satisfy every selected category can share one score.
        if exact_matching_applied and recommendations["similarity_score"].nunique() == 1:
            st.caption(
                "Every exercise below satisfies all the preferences you selected, so "
                "the model gives them the same similarity score."
            )

        # Display each result in an expander to keep long instructions readable.
        for _, exercise in recommendations.iterrows():
            rank = int(exercise["rank"])
            similarity_percentage = float(exercise["similarity_score"]) * 100

            card_title = (
                f"{rank}. {exercise['exercise_name']} — "
                f"{similarity_percentage:.1f}% similarity"
            )
            with st.expander(card_title, expanded=(rank <= 3)):
                # Three columns of details keep the card compact and scannable.
                detail_one, detail_two, detail_three = st.columns(3)
                detail_one.markdown(
                    f"**Muscle group:** {format_category(exercise['muscle_group'])}"
                )
                detail_one.markdown(
                    f"**Muscles worked:** {exercise['muscles_worked']}"
                )
                detail_two.markdown(
                    f"**Level:** {format_category(exercise['level'])}"
                )
                detail_two.markdown(
                    f"**Difficulty score:** {exercise['difficulty_score']}"
                )
                detail_three.markdown(
                    f"**Equipment:** {format_category(exercise['equipment'])}"
                )
                detail_three.markdown(
                    f"**Movement:** {format_category(exercise['mechanics'])} / "
                    f"{format_category(exercise['force'])}"
                )

                match_summary = build_match_summary(
                    exercise=exercise,
                    target_muscle_group=target_muscle_group,
                    experience_level=experience_level,
                    available_equipment=available_equipment,
                    exercise_mechanics=exercise_mechanics,
                    force_type=force_type,
                )
                st.markdown(f"**Why recommended:** {match_summary}")

                st.markdown("**Preparation**")
                st.write(
                    exercise.get("preparation") or "No preparation details available."
                )
                st.markdown("**Execution**")
                st.write(
                    exercise.get("execution") or "No execution details available."
                )

        # Prepare the result table that can be downloaded for later use.
        result_csv = recommendations.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download recommendations as CSV",
            data=result_csv,
            file_name="exercise_recommendations.csv",
            mime="text/csv",
        )
else:
    st.info("Choose your preferences above and select 'Recommend exercises'.")


# Explain the model in the app so results are transparent to project reviewers.
with st.expander("How the recommendation model works"):
    st.markdown(
        """
        1. The selected preferences are collected into one user profile using exactly
           five features: target muscle group, experience level, available equipment,
           exercise mechanics, and force type.
        2. The user profile and every exercise are converted into numbers with
           multi-hot encoding in the same feature space.
        3. Feature weights are applied before the comparison is made.
        4. Cosine similarity measures how closely each exercise matches the profile
           and the exercises are ranked from most to least similar.
        5. The optional category filter only decides which exercises are eligible.
           It never changes how the similarity score is calculated.
        6. The value shown on each card is the cosine-similarity score itself. It is
           not normalised, so its highest possible value depends on how many
           preferences were selected, and it is not a medical or physiological
           effectiveness score.
        """
    )

    # Read straight from the recommender, so this stays correct if the weights change.
    st.markdown(
        f"""
        **Feature weights used:**
        - Target muscle group: {FEATURE_WEIGHTS['muscle_group']}
        - Experience level: {FEATURE_WEIGHTS['level']}
        - Available equipment: {FEATURE_WEIGHTS['equipment']}
        - Exercise mechanics: {FEATURE_WEIGHTS['mechanics']}
        - Force type: {FEATURE_WEIGHTS['force']}
        """
    )

    st.caption(
        "Because the model compares only these five categories, exercises that "
        "satisfy every selected category share the same similarity score. Switching "
        "off exact-match filtering compares your preferences against the whole "
        "dataset, which spreads the scores out."
    )


# Include an appropriate safety note for a fitness recommendation prototype.
st.caption(
    "This prototype provides dataset-based suggestions, not medical advice. Users "
    "should consider health conditions, correct technique and professional guidance."
)
