import pandas as pd
import re
from pathlib import Path


# Main project folder path.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Path to the original dataset.
input_file_path = PROJECT_ROOT / "data" / "raw" / "gym_exercise_dataset.csv"

# Save the cleaned dataset.
output_file_path = PROJECT_ROOT / "data" / "processed" / "cleaned_exercise_dataset.csv"


# Check the file is available before continuing. If the file is not found, raise an error with a clear message.
if not input_file_path.exists():
    raise FileNotFoundError(
        f"The dataset was not found here: {input_file_path}"
    )


# 1. Load the original dataset.
df = pd.read_csv(input_file_path)

# Save the original number of row. Later we can check how many rows were removed during cleaning.
original_row_count = len(df)


# 2. Select only the columns needed for this project.
selected_columns = [
    "Exercise Name",
    "Equipment",
    "Main_muscle",
    "Target_Muscles",
    "Secondary Muscles",
    "Difficulty (1-5)",
    "Mechanics",
    "Force",
    "Preparation",
    "Execution"
]

# Check that all required columns are available before continuing.
missing_columns = [
    column for column in selected_columns if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"These required columns are missing: {missing_columns}"
    )

# Create a separate copy so the original dataframe is not changed.
exercise_df = df[selected_columns].copy()


# 3. Rename the columns so they are easier to use in Python.
exercise_df = exercise_df.rename(columns={
    "Exercise Name": "exercise_name",
    "Equipment": "equipment",
    "Main_muscle": "muscle_group",
    "Target_Muscles": "target_muscles",
    "Secondary Muscles": "secondary_muscles",
    "Difficulty (1-5)": "difficulty_score",
    "Mechanics": "mechanics",
    "Force": "force",
    "Preparation": "preparation",
    "Execution": "execution"
})


# 4. Convert the difficulty score into a number.
# Any value that cannot be converted will become a missing value.
exercise_df["difficulty_score"] = pd.to_numeric(
    exercise_df["difficulty_score"],
    errors="coerce"
)


# 5. Convert the difficulty score into a simple user level.
def map_level(difficulty_score):
    # Scores 1 and 2 are beginner exercises.
    if difficulty_score in [1, 2]:
        return "Beginner"

    # Score 3 is an intermediate exercise.
    if difficulty_score == 3:
        return "Intermediate"

    # Scores 4 and 5 are advanced exercises.
    if difficulty_score in [4, 5]:
        return "Advanced"

    # Any other value is treated as missing.
    return pd.NA


# Create the level column but keep difficulty_score as well.
exercise_df["level"] = exercise_df["difficulty_score"].apply(map_level)


# 6. Clean normal text values.
def clean_text(value):
    # Keep a missing value as missing.
    if pd.isna(value):
        return pd.NA

    # Convert the value to text before cleaning it.
    value = str(value)

    # Remove hidden zero-width spaces that may be inside the dataset.
    value = value.replace("\u200b", "")

    # Replace repeated spaces, tabs and line breaks with one space.
    value = re.sub(r"\s+", " ", value)

    # Remove extra spaces and commas from the beginning and end.
    value = value.strip(" ,")

    # Treat blank text and the word None as missing values.
    if value == "" or value.lower() == "none":
        return pd.NA

    return value


# 7. Apply the text cleaning to all text columns.
text_columns = [
    "exercise_name",
    "equipment",
    "muscle_group",
    "target_muscles",
    "secondary_muscles",
    "mechanics",
    "force",
    "preparation",
    "execution"
]

for column in text_columns:
    exercise_df[column] = exercise_df[column].apply(clean_text)


# 8. Standardise selected equipment names.
# These mappings keep the same equipment names consistent in the final file.
equipment_mapping = {
    "Assisted (machine)": "Assisted",
    "Assisted (partner)": "Assisted",
    "Assisted Chest Dip": "Assisted",

    "Band Resistive": "Resistance Band",
    "Band-assisted": "Resistance Band",

    "Cable (pull side)": "Cable",
    "Cable Standing Fly": "Cable",

    "Sled (plate loaded)": "Sled",
    "Sled (selectorized)": "Sled",

    "Isometric": "Body Weight",
    "Plyometric": "Body Weight",

    "Suspended": "Suspension Straps",
    "Suspension": "Suspension Straps",

    "Lever": "Lever (selectorized)",

    "Weighted Chest Dip": "Weighted",

    "Lever (selectorized) Chest Dip": "Lever (selectorized)"
}

exercise_df["equipment"] = exercise_df["equipment"].replace(
    equipment_mapping
)


# 9. Standardise the Force column.
push_pull_rows_changed = (
    exercise_df["force"] == "Push & Pull"
).sum()

exercise_df["force"] = exercise_df["force"].replace({
    "Push & Pull": "Push"
})


# 10. Split a muscle column into a clean list of muscle names.
def split_muscles(value):
    # A missing secondary-muscle value should give an empty list,
    # not cause the whole exercise to be deleted.
    if pd.isna(value):
        return []

    cleaned_muscles = []
    seen_muscles = set()

    # The muscles in this dataset are separated by commas.
    for muscle in str(value).split(","):
        muscle = clean_text(muscle)

        # Ignore blank parts created by extra commas.
        if pd.isna(muscle):
            continue

        # Use lowercase only for checking duplicates.
        # The original capitalisation is kept in the final result.
        muscle_key = muscle.casefold()

        if muscle_key not in seen_muscles:
            cleaned_muscles.append(muscle)
            seen_muscles.add(muscle_key)

    return cleaned_muscles


# 11. Remove muscles repeated between target and secondary muscles,
# and then combine both columns into one muscles column.
def combine_muscle_columns(target_value, secondary_value):
    # Get clean and unique target muscles.
    target_list = split_muscles(target_value)

    # Get clean and unique secondary muscles.
    secondary_list = split_muscles(secondary_value)

    # Make a set of target muscles so repeated secondary muscles can be removed.
    target_keys = {muscle.casefold() for muscle in target_list}

    # Keep only secondary muscles that are not already target muscles.
    secondary_without_repeats = [
        muscle
        for muscle in secondary_list
        if muscle.casefold() not in target_keys
    ]

    # Put target muscles first, followed by the remaining secondary muscles.
    combined_muscles = target_list + secondary_without_repeats

    # Return a missing value only if both original columns were empty.
    if not combined_muscles:
        return pd.NA

    # Save the combined muscles as one comma-separated text value.
    return ", ".join(combined_muscles)


exercise_df["muscles_worked"] = exercise_df.apply(
    lambda row: combine_muscle_columns(
        row["target_muscles"],
        row["secondary_muscles"]
    ),
    axis=1
)


# 12. Remove rows only when an important field is missing.
# secondary_muscles is not included because it is allowed to be blank.
# The new muscles column will still contain the target muscles.
required_columns = [
    "exercise_name",
    "equipment",
    "muscle_group",
    "muscles_worked",
    "difficulty_score",
    "level",
    "mechanics",
    "force",
    "preparation",
    "execution"
]

rows_before_missing_check = len(exercise_df)
exercise_df = exercise_df.dropna(subset=required_columns)
rows_removed_for_missing_values = rows_before_missing_check - len(exercise_df)


# 13. Remove the two old muscle columns because they are now combined.
exercise_df = exercise_df.drop(columns=[
    "target_muscles",
    "secondary_muscles"
])


# 14. Remove exact duplicate exercise records.
# The difficulty score and the combined muscles column are included in this check.
rows_before_duplicate_check = len(exercise_df)

exercise_df = exercise_df.drop_duplicates(
    subset=[
        "exercise_name",
        "difficulty_score",
        "level",
        "equipment",
        "muscle_group",
        "muscles_worked",
        "mechanics",
        "force",
        "preparation",
        "execution"
    ],
    keep="first"
)

rows_removed_as_duplicates = rows_before_duplicate_check - len(exercise_df)


# 15. Reset the row index before creating the exercise IDs.
exercise_df = exercise_df.reset_index(drop=True)


# 16. Create a new exercise ID after all cleaning is complete.
exercise_df.insert(
    0,
    "exercise_id",
    [f"EX{str(number).zfill(3)}" for number in range(1, len(exercise_df) + 1)]
)


# 17. Arrange the columns in the order I want in the final dataset.
final_columns = [
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
    "execution"
]

exercise_df = exercise_df[final_columns]


# 18. Save the final cleaned dataset.
exercise_df.to_csv(output_file_path, index=False)


# 19. Display a simple cleaning summary so I can check the result.
print("Cleaned exercise dataset created successfully!")
print("Input file:", input_file_path)
print("Output file:", output_file_path)
print("Original row count:", original_row_count)
print("Rows removed for missing required values:", rows_removed_for_missing_values)
print("Rows removed as duplicates:", rows_removed_as_duplicates)
print("Push & Pull values changed to Push:", push_pull_rows_changed)
print("Final dataset shape:", exercise_df.shape)
print("\nFirst five rows:")
print(exercise_df.head())


# 20. Display the final equipment values.
print("\nFinal equipment values:")
print(exercise_df["equipment"].value_counts())

# 21. Display the final muscle group values.
print("\nFinal muscle group values:")
print(exercise_df["muscle_group"].value_counts())

# 22. Display the final level values.
print("\nFinal level values:")
print(exercise_df["level"].value_counts())

# 23. Check that the final dataset has no missing required values.
print("\nMissing values in the final dataset:")
print(exercise_df.isna().sum())
