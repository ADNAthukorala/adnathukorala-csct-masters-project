"""Streamlit user interface for the basic exercise recommendation model."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from src.recommender import ExerciseRecommender


# Configure the browser tab and use a wide layout for recommendation cards.
st.set_page_config(
    page_title="Exercise Recommendation System",
    page_icon="🏋️",
    layout="wide",
)


# Build the dataset path from the project root so the app works from any directory.
PROJECT_DIRECTORY = Path(__file__).resolve().parent
DATASET_PATH = (
    PROJECT_DIRECTORY / "data" / "processed" / "cleaned_exercise_dataset.csv"
)


@st.cache_resource(show_spinner=False)
def load_recommender(dataset_path: Path) -> ExerciseRecommender:
    """Load and fit the model once, then reuse it across Streamlit reruns."""

    # Streamlit reruns the script after interactions, so resource caching avoids
    # repeatedly fitting the same TF-IDF model.
    return ExerciseRecommender(dataset_path)


# Show a clear title and describe the purpose of the prototype.
st.title("🏋️ Exercise Recommendation System")
st.caption(
    "Choose your training preferences to receive ranked, explainable exercise "
    "recommendations from the cleaned dataset."
)


# Load the recommendation model and stop with a useful message if loading fails.
try:
    recommender = load_recommender(DATASET_PATH)
except (FileNotFoundError, ValueError, pd.errors.ParserError) as error:
    st.error(f"The recommendation system could not be loaded: {error}")
    st.stop()


# Read all available category values directly from the cleaned dataset.
options = recommender.available_options()


# Display small dataset indicators so the user understands the available coverage.
metric_one, metric_two, metric_three = st.columns(3)
metric_one.metric("Exercises", f"{len(recommender.data):,}")
metric_two.metric("Muscle groups", len(options["muscle_groups"]))
metric_three.metric("Equipment types", len(options["equipment"]))


# Keep all user controls inside one form to avoid unnecessary model reruns.
with st.form("recommendation_form"):
    st.subheader("Your preferences")

    # The first row contains the most important recommendation inputs.
    first_column, second_column = st.columns(2)
    with first_column:
        # Allow exactly one target muscle group to keep the request focused.
        selected_muscle_group = st.selectbox(
            "Target muscle group",
            options=options["muscle_groups"],
            help="Select the main muscle group you want to train.",
        )

        # The recommendation model accepts a sequence, so wrap the selection in a list.
        selected_muscle_groups = [selected_muscle_group]
        selected_level = st.selectbox(
            "Experience level",
            options=["Any"] + options["levels"],
            index=0,
        )
        selected_equipment = st.multiselect(
            "Available equipment",
            options=options["equipment"],
            help="Leave empty to allow all equipment types.",
        )

    # The second row contains optional movement and result controls.
    with second_column:
        selected_mechanics = st.selectbox(
            "Exercise mechanics",
            options=["Any"] + options["mechanics"],
            index=0,
        )
        selected_force = st.selectbox(
            "Force type",
            options=["Any"] + options["forces"],
            index=0,
        )
        specific_muscles = st.text_input(
            "Specific muscles or keywords",
            placeholder="For example: biceps, hamstrings, latissimus dorsi",
            help="Optional free text is matched using the TF-IDF model.",
        )

    # Let the user control both result count and filter behaviour.
    result_column, filter_column = st.columns(2)
    with result_column:
        number_of_results = st.slider(
            "Number of recommendations",
            min_value=3,
            max_value=15,
            value=5,
            step=1,
        )
    with filter_column:
        strict_filters = st.checkbox(
            "Require exact matches for selected categories",
            value=False,
            help=(
                "When selected, level, equipment, mechanics, and force are used "
                "as hard filters. The app relaxes them automatically if no row matches."
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
    # Run the model with all preferences chosen in the form.
    results = recommender.recommend(
        muscle_groups=selected_muscle_groups,
        level=selected_level,
        equipment=selected_equipment,
        mechanics=selected_mechanics,
        force=selected_force,
        specific_muscles=specific_muscles,
        top_n=number_of_results,
        strict_filters=strict_filters,
    )

    # Automatically provide useful alternatives when a strict combination is empty.
    if results.empty and strict_filters:
        st.info(
            "No exercise matched every strict filter. Showing the closest overall "
            "matches instead."
        )
        results = recommender.recommend(
            muscle_groups=selected_muscle_groups,
            level=selected_level,
            equipment=selected_equipment,
            mechanics=selected_mechanics,
            force=selected_force,
            specific_muscles=specific_muscles,
            top_n=number_of_results,
            strict_filters=False,
        )

    # Show a heading before the ranked recommendation cards.
    st.divider()
    st.subheader("Recommended exercises")

    # Display each result in an expander to keep long instructions readable.
    for rank, (_, exercise) in enumerate(results.iterrows(), start=1):
        card_title = (
            f"{rank}. {exercise['exercise_name']} — "
            f"{exercise['recommendation_score']:.1f}% match"
        )
        with st.expander(card_title, expanded=(rank <= 3)):
            detail_one, detail_two, detail_three = st.columns(3)
            detail_one.markdown(f"**Muscle group:** {exercise['muscle_group']}")
            detail_one.markdown(f"**Muscles worked:** {exercise['muscles_worked']}")
            detail_two.markdown(f"**Level:** {exercise['level']}")
            detail_two.markdown(
                f"**Difficulty score:** {exercise['difficulty_score']}"
            )
            detail_three.markdown(f"**Equipment:** {exercise['equipment']}")
            detail_three.markdown(
                f"**Movement:** {exercise['mechanics']} / {exercise['force']}"
            )

            st.markdown(f"**Why recommended:** {exercise['recommendation_reason']}")
            st.markdown("**Preparation**")
            st.write(exercise["preparation"])
            st.markdown("**Execution**")
            st.write(exercise["execution"])

    # Prepare a smaller result table that can be downloaded for later use.
    export_columns = [
        "exercise_id",
        "exercise_name",
        "recommendation_score",
        "level",
        "equipment",
        "muscle_group",
        "muscles_worked",
        "mechanics",
        "force",
        "preparation",
        "execution",
    ]
    result_csv = results[export_columns].to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download recommendations as CSV",
        data=result_csv,
        file_name="exercise_recommendations.csv",
        mime="text/csv",
    )


# Explain the model in the app so results are transparent to project reviewers.
with st.expander("How the recommendation model works"):
    st.markdown(
        """
        1. Each exercise is converted into weighted text using its muscle group,
           muscles worked, level, equipment, mechanics, force, and name.
        2. TF-IDF converts the exercise text and your preferences into numeric vectors.
        3. Cosine similarity measures how closely each exercise matches your request.
        4. That similarity is the score. No extra points are added for exact matches,
           so the ranking depends only on the TF-IDF comparison.
        5. The final score ranks exercises; it is a matching score, not a medical or
           physiological effectiveness score.
        """
    )


# Include an appropriate safety note for a fitness recommendation prototype.
st.caption(
    "This prototype provides dataset-based suggestions, not medical advice. Users "
    "should consider health conditions, correct technique, and professional guidance."
)
