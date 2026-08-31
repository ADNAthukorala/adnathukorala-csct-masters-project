"""Simple category filters for the exercise recommender.

This module adds the same straightforward filtering approach used in the second
model. The categories chosen by the user are treated as exact-match
requirements, so only exercises that satisfy every selected category stay in the
candidate pool. When a combination returns nothing, the Streamlit app relaxes
the filter automatically and tells the user.

Important: this filter only reduces the pool of candidate exercises before the
recommender runs. It does not change how the recommendation model encodes
features, calculates cosine similarity, weights features, or ranks results.
Dropping rows does not alter the cosine-similarity score of the rows that
remain, so the surviving exercises keep exactly the scores and the relative
order that the unchanged Model 1 algorithm gives them.
"""

from __future__ import annotations

import pandas as pd

from .text_utils import clean_text, split_values


# Chosen in a dropdown when the user has no preference for that category.
ANY_OPTION = "Any"


def _row_matches_selection(cell_value, selected_values: list[str]) -> bool:
    """Return True when a dataset cell matches at least one selected value."""
    # Dataset cells can hold one value or a comma-separated list of values.
    cell_values = set(split_values(cell_value))

    # Both sides are normalised the same way so capitalisation never matters.
    return any(clean_text(value) in cell_values for value in selected_values if value)


def build_filter_mask(
    exercises: pd.DataFrame,
    target_muscle_group: str,
    experience_level: str,
    available_equipment: list[str],
    exercise_mechanics: str,
    force_type: str,
) -> pd.Series:
    """Return a boolean mask of exercises that match all selected categories."""
    # Start by keeping every exercise, then narrow the pool one category at a time.
    mask = pd.Series(True, index=exercises.index)

    # A single target muscle group is always selected in the interface.
    if target_muscle_group and target_muscle_group != ANY_OPTION:
        mask &= exercises["muscle_group"].apply(
            lambda value: _row_matches_selection(value, [target_muscle_group])
        )

    # "Any" means the category should not reduce the candidate pool.
    if experience_level and experience_level != ANY_OPTION:
        mask &= exercises["level"].apply(
            lambda value: _row_matches_selection(value, [experience_level])
        )

    # An empty equipment selection means all equipment types are allowed.
    if available_equipment:
        mask &= exercises["equipment"].apply(
            lambda value: _row_matches_selection(value, available_equipment)
        )

    if exercise_mechanics and exercise_mechanics != ANY_OPTION:
        mask &= exercises["mechanics"].apply(
            lambda value: _row_matches_selection(value, [exercise_mechanics])
        )

    if force_type and force_type != ANY_OPTION:
        mask &= exercises["force"].apply(
            lambda value: _row_matches_selection(value, [force_type])
        )

    return mask


def apply_filters(
    exercises: pd.DataFrame,
    target_muscle_group: str,
    experience_level: str,
    available_equipment: list[str],
    exercise_mechanics: str,
    force_type: str,
) -> pd.DataFrame:
    """Return only the exercises that satisfy every selected category."""
    mask = build_filter_mask(
        exercises=exercises,
        target_muscle_group=target_muscle_group,
        experience_level=experience_level,
        available_equipment=available_equipment,
        exercise_mechanics=exercise_mechanics,
        force_type=force_type,
    )

    # A copy is returned so the original loaded dataset is never modified.
    return exercises[mask].copy()
