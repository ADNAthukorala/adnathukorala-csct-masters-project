"""Content based recommendation model.

This module contains the exercise recommendation system. It loads the cleaned
exercise dataset and compares the user's selected preferences with the available
exercises. The system uses TF-IDF to convert exercise features and user choices
into numerical values, then applies cosine similarity to measure how closely
they match. Ranking is based on cosine similarity alone: no additional
exact-match bonus points are applied, so the influence of each preference comes
only from the feature weighting used to build the TF-IDF documents. The
exercises are then ranked by their similarity scores and the best matches are
returned as personalised recommendations.
"""


from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# Define the columns that must exist before the recommendation model can run.
REQUIRED_COLUMNS = {
    "exercise_id",
    "exercise_name",
    "difficulty_score",
    "level",
    "equipment",
    "muscle_group",
    "muscles_worked",
    "mechanics",
    "force",
    "preparation",
    "execution",
}


class ExerciseRecommender:
    """Build and query a transparent content-based exercise recommender."""

    def __init__(self, dataset_path: str | Path) -> None:
        """Load the dataset, validate it, and fit the TF-IDF feature model."""

        # cCnverts the supplied dataset location into a Path object for safer file handling.
        self.dataset_path = Path(dataset_path)

        # Load the cleaned CSV dataset into a pandas DataFrame.
        self.data = pd.read_csv(self.dataset_path)

        # Validate the dataset before any model processing is performed.
        self._validate_dataset()

        # Reset the index so matrix row numbers always match DataFrame row numbers.
        self.data = self.data.reset_index(drop=True)

        # Build one weighted feature document for every exercise.
        self.data["model_features"] = self.data.apply(
            self._build_exercise_feature_text,
            axis=1,
        )

        # Configure TF-IDF to learn both single terms and useful two-word phrases.
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            strip_accents="unicode",
            sublinear_tf=True,
        )

        # Fit the vectorizer once and store every exercise as a sparse matrix row.
        self.feature_matrix = self.vectorizer.fit_transform(
            self.data["model_features"]
        )

    def _validate_dataset(self) -> None:
        """Raise a clear error when the input data is incomplete or malformed."""

        # Identify any required columns that are absent from the uploaded dataset.
        missing_columns = REQUIRED_COLUMNS.difference(self.data.columns)
        if missing_columns:
            missing_text = ", ".join(sorted(missing_columns))
            raise ValueError(f"The dataset is missing required columns: {missing_text}")

        # An empty dataset cannot be used to fit a recommendation model.
        if self.data.empty:
            raise ValueError("The dataset contains no exercise rows.")

        # Exercise IDs should uniquely identify rows shown in the application.
        if self.data["exercise_id"].duplicated().any():
            raise ValueError("The dataset contains duplicate exercise IDs.")

        # Required recommendation fields should not contain missing values.
        required_frame = self.data[list(REQUIRED_COLUMNS)]
        if required_frame.isna().any().any():
            raise ValueError("The dataset contains missing values in required columns.")

        # Difficulty must be numeric because it is used for ranking and display.
        if not pd.api.types.is_numeric_dtype(self.data["difficulty_score"]):
            raise ValueError("difficulty_score must be a numeric column.")

    @staticmethod
    def _repeat_feature(label: str, value: object, repetitions: int) -> str:
        """Repeat a labelled value so important preferences receive more weight."""

        # Convert the value to clean text and keep the label beside the value.
        labelled_value = f"{label} {str(value).strip()}"

        # Repetition is a simple and visible method of applying feature weighting.
        return " ".join([labelled_value] * repetitions)

    def _build_exercise_feature_text(self, row: pd.Series) -> str:
        """Create the weighted text representation used for one exercise."""

        # Muscle group and exact muscles are the strongest recommendation signals.
        muscle_group = self._repeat_feature(
            "muscle group",
            row["muscle_group"],
            repetitions=5,
        )
        muscles_worked = self._repeat_feature(
            "muscles worked",
            row["muscles_worked"],
            repetitions=4,
        )

        # Level and equipment are important practical user constraints.
        level = self._repeat_feature("level", row["level"], repetitions=3)
        equipment = self._repeat_feature(
            "equipment",
            row["equipment"],
            repetitions=3,
        )

        # Mechanics, force, and the exercise name provide additional distinctions.
        mechanics = self._repeat_feature(
            "mechanics",
            row["mechanics"],
            repetitions=2,
        )
        force = self._repeat_feature("force", row["force"], repetitions=2)
        exercise_name = self._repeat_feature(
            "exercise name",
            row["exercise_name"],
            repetitions=1,
        )

        # Preparation and execution are intentionally excluded because long
        # instructional text would overpower the user's categorical preferences.
        return " ".join(
            [
                muscle_group,
                muscles_worked,
                level,
                equipment,
                mechanics,
                force,
                exercise_name,
            ]
        )

    @staticmethod
    def _normalise_multi_value(values: Iterable[str] | None) -> list[str]:
        """Return a clean, unique list while preserving the user's order."""

        # Treat a missing input as an empty list, which means no preference.
        if values is None:
            return []

        # dict.fromkeys removes duplicates without changing the original order.
        return list(dict.fromkeys(str(value).strip() for value in values if value))

    def _build_user_query(
        self,
        muscle_groups: Sequence[str],
        level: str | None,
        equipment: Sequence[str],
        mechanics: str | None,
        force: str | None,
        specific_muscles: str,
    ) -> str:
        """Create a weighted feature document from the user's selected preferences."""

        # Store all active query components before joining them into one document.
        query_parts: list[str] = []

        # Add every selected muscle group with the same weighting used in training.
        for muscle_group in muscle_groups:
            query_parts.append(
                self._repeat_feature("muscle group", muscle_group, repetitions=5)
            )

        # Add an exact training level only when the user has selected one.
        if level and level != "Any":
            query_parts.append(self._repeat_feature("level", level, repetitions=3))

        # Add all equipment options selected by the user.
        for equipment_item in equipment:
            query_parts.append(
                self._repeat_feature("equipment", equipment_item, repetitions=3)
            )

        # Add optional exercise mechanics and force preferences.
        if mechanics and mechanics != "Any":
            query_parts.append(
                self._repeat_feature("mechanics", mechanics, repetitions=2)
            )
        if force and force != "Any":
            query_parts.append(self._repeat_feature("force", force, repetitions=2))

        # Add free text to match specific muscle names such as biceps or hamstrings.
        if specific_muscles.strip():
            query_parts.append(
                self._repeat_feature(
                    "muscles worked",
                    specific_muscles.strip(),
                    repetitions=4,
                )
            )

        # A joined string can be passed directly to the fitted TF-IDF vectorizer.
        return " ".join(query_parts)

    def _strict_candidate_mask(
        self,
        muscle_groups: Sequence[str],
        level: str | None,
        equipment: Sequence[str],
        mechanics: str | None,
        force: str | None,
    ) -> pd.Series:
        """Return rows that exactly satisfy all selected structured preferences."""

        # Start with every row available and progressively apply active filters.
        mask = pd.Series(True, index=self.data.index)

        # At least one muscle group is required by the application.
        if muscle_groups:
            mask &= self.data["muscle_group"].isin(muscle_groups)

        # "Any" means that a field should not reduce the candidate pool.
        if level and level != "Any":
            mask &= self.data["level"].eq(level)
        if equipment:
            mask &= self.data["equipment"].isin(equipment)
        if mechanics and mechanics != "Any":
            mask &= self.data["mechanics"].eq(mechanics)
        if force and force != "Any":
            mask &= self.data["force"].eq(force)

        return mask

    @staticmethod
    def _build_reason(
        row: pd.Series,
        muscle_groups: Sequence[str],
        level: str | None,
        equipment: Sequence[str],
        mechanics: str | None,
        force: str | None,
    ) -> str:
        """Create a short human readable explanation for one recommendation."""

        # Collect only preferences that the exercise actually matches.
        reasons: list[str] = []
        if row["muscle_group"] in muscle_groups:
            reasons.append(f"targets {row['muscle_group']}")
        if level and level != "Any" and row["level"] == level:
            reasons.append(f"matches {level} level")
        if equipment and row["equipment"] in equipment:
            reasons.append(f"uses {row['equipment']}")
        if mechanics and mechanics != "Any" and row["mechanics"] == mechanics:
            reasons.append(f"is {mechanics.lower()}")
        if force and force != "Any" and row["force"] == force:
            reasons.append(f"uses a {force.lower()} action")

        # Provide a fallback explanation when relaxed matching is used.
        if not reasons:
            reasons.append("has the closest overall feature profile")

        return ", ".join(reasons).capitalize() + "."

    def recommend(
        self,
        muscle_groups: Sequence[str],
        level: str | None = "Any",
        equipment: Iterable[str] | None = None,
        mechanics: str | None = "Any",
        force: str | None = "Any",
        specific_muscles: str = "",
        top_n: int = 5,
        strict_filters: bool = True,
    ) -> pd.DataFrame:
        """Return ranked exercise recommendations for the supplied preferences."""

        # Clean user inputs so model behaviour is predictable.
        clean_muscle_groups = self._normalise_multi_value(muscle_groups)
        clean_equipment = self._normalise_multi_value(equipment)

        # Muscle group is the minimum information needed for a useful recommendation.
        if not clean_muscle_groups:
            raise ValueError("Select at least one muscle group.")

        # Restrict the requested result count to a sensible positive integer.
        top_n = max(1, int(top_n))

        # Build the user's weighted text query and convert it to the model space.
        user_query = self._build_user_query(
            muscle_groups=clean_muscle_groups,
            level=level,
            equipment=clean_equipment,
            mechanics=mechanics,
            force=force,
            specific_muscles=specific_muscles,
        )
        user_vector = self.vectorizer.transform([user_query])

        # Strict mode filters candidates before similarity ranking is calculated.
        if strict_filters:
            candidate_mask = self._strict_candidate_mask(
                muscle_groups=clean_muscle_groups,
                level=level,
                equipment=clean_equipment,
                mechanics=mechanics,
                force=force,
            )
        else:
            # Relaxed mode keeps all exercises and lets similarity determine rank.
            candidate_mask = pd.Series(True, index=self.data.index)

        # Keep the original integer positions for sparse matrix row selection.
        candidate_indices = self.data.index[candidate_mask].to_numpy()

        # Return an empty result with stable columns when strict filters match nothing.
        if len(candidate_indices) == 0:
            result_columns = list(self.data.columns) + [
                "similarity_score",
                "recommendation_score",
                "recommendation_reason",
            ]
            return pd.DataFrame(columns=result_columns)

        # Calculate cosine similarity between the user and candidate feature vectors.
        similarities = cosine_similarity(
            user_vector,
            self.feature_matrix[candidate_indices],
        ).flatten()

        # Copy candidate rows so scores can be added without changing source data.
        candidates = self.data.loc[candidate_indices].copy()

        # Cosine similarity is already bounded between 0 and 1, so the score is
        # simply expressed as a percentage. No exact-match bonus points are added,
        # which keeps the ranking a direct reflection of the TF-IDF comparison.
        percentage_score = np.clip(similarities * 100.0, 0.0, 100.0)

        # Store component and final scores for evaluation and transparent display.
        candidates["similarity_score"] = similarities
        candidates["recommendation_score"] = percentage_score.round(1)

        # Explain the main exact matches behind every recommendation.
        candidates["recommendation_reason"] = candidates.apply(
            lambda row: self._build_reason(
                row=row,
                muscle_groups=clean_muscle_groups,
                level=level,
                equipment=clean_equipment,
                mechanics=mechanics,
                force=force,
            ),
            axis=1,
        )

        # Rank by the model's own score, then settle ties with exercise_id.
        #
        # The raw similarity is used rather than recommendation_score, because
        # that column is rounded to one decimal place for display and would
        # merge exercises that are genuinely different.
        #
        # Rounding to ten decimal places removes floating-point noise, which is
        # about 1e-16, without touching any real difference in score.
        # exercise_id is unique, so the final order is fully determined and
        # never depends on the order the rows sit in the CSV file.
        candidates = candidates.assign(
            rank_score=candidates["similarity_score"].round(10)
        )
        candidates = candidates.sort_values(
            by=["rank_score", "exercise_id"],
            ascending=[False, True],
        )
        candidates = candidates.drop(columns=["rank_score"])

        # Remove the internal model text before returning results to the application.
        return candidates.head(top_n).drop(columns=["model_features"])

    def available_options(self) -> dict[str, list[str]]:
        """Return sorted values used to build Streamlit selection controls."""

        # Create one central source for all user interface option lists.
        return {
            "muscle_groups": sorted(self.data["muscle_group"].unique().tolist()),
            "levels": sorted(
                self.data["level"].unique().tolist(),
                key=lambda value: ["Beginner", "Intermediate", "Advanced"].index(value),
            ),
            "equipment": sorted(self.data["equipment"].unique().tolist()),
            "mechanics": sorted(self.data["mechanics"].unique().tolist()),
            "forces": sorted(self.data["force"].unique().tolist()),
        }
