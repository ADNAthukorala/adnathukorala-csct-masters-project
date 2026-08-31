from __future__ import annotations

import pandas as pd


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of the dataframe with clean Python-friendly column names."""
    df = df.copy()

    # Lowercase everything and swap spaces and dashes for underscores,
    # so "Muscle Group" becomes "muscle_group".
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
        .str.replace("-", "_", regex=False)
    )
    return df


def clean_text(value) -> str:
    """Clean text values so matching is consistent."""
    if pd.isna(value):
        return ""

    value = str(value).strip().lower()

    # The dataset separates multiple values in a few different ways.
    # Turning them all into commas means only one separator has to be handled later.
    value = value.replace("&", ",")
    value = value.replace("/", ",")
    value = value.replace("|", ",")
    value = value.replace(";", ",")

    # Collapse any repeated spaces into a single space.
    value = " ".join(value.split())
    return value


def split_values(value) -> list[str]:
    """Split comma-separated values into a clean list."""
    value = clean_text(value)
    if value == "":
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def get_unique_values(df: pd.DataFrame, column: str) -> list[str]:
    """Return sorted unique values from a column, including comma-separated values."""
    values: set[str] = set()

    # A cell may hold several values, so each one is split before being collected.
    # A set is used because the same value appears in many rows.
    for item in df[column].dropna():
        values.update(split_values(item))
    return sorted([value.title() for value in values])


def get_muscles_for_muscle_group(
    df: pd.DataFrame,
    selected_muscle_group: str,
) -> list[str]:
    """
    Return only the muscles that belong to the selected muscle group.

    Example: if selected_muscle_group is Back, this returns only back muscles.
    """
    selected_group_clean = clean_text(selected_muscle_group)

    filtered_df = df[
        df["muscle_group"].apply(lambda value: clean_text(value) == selected_group_clean)
    ]

    values: set[str] = set()
    for item in filtered_df["muscles_worked"].dropna():
        values.update(split_values(item))

    return sorted([value.title() for value in values])
