from __future__ import annotations

from pathlib import Path

import pandas as pd

from .text_utils import clean_column_names, clean_text


# Columns that must exist in the cleaned dataset before the app can run.
# These names follow the cleaned_exercise_dataset.csv schema.
REQUIRED_COLUMNS = [
    "exercise_id",
    "exercise_name",
    "difficulty_score",
    "level",
    "equipment",
    "muscle_group",
    "muscles_worked",
    "mechanics",
    "force",
]


# Only the five model features are normalised here, because these are the
# columns the recommender encodes and the category filter compares.
TEXT_COLUMNS = [
    "level",
    "equipment",
    "muscle_group",
    "mechanics",
    "force",
]


# Step-by-step instructions shown on the recommendation cards.
INSTRUCTION_COLUMNS = [
    "preparation",
    "execution",
]


def load_exercise_data(path: str | Path) -> pd.DataFrame:
    """Load, validate and clean the exercise dataset."""
    path = Path(path)

    # Fail early with a message instead of a confusing pandas error.
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}. Place cleaned_exercise_dataset.csv inside the "
            "data/processed folder."
        )

    df = pd.read_csv(path)
    df = clean_column_names(df)

    # Stop here if the file is missing anything the recommender depends on.
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns in exercise dataset: {missing_columns}")

    # Standardise the feature columns so user choices match dataset values reliably.
    for column in TEXT_COLUMNS:
        if column in df.columns:
            df[column] = df[column].apply(clean_text)

    # Blank instructions become empty text so the app never prints "nan".
    for column in INSTRUCTION_COLUMNS:
        if column in df.columns:
            df[column] = df[column].fillna("").astype(str).str.strip()

    return df
