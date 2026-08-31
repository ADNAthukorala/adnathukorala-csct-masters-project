import random
import itertools
from pathlib import Path

import pandas as pd

# Settings

RANDOM_SEED = 42    # Fixed seed so the same 50 users are generated every time.
MIN_RELEVANT = 10   # Every selected user must have at least 10 relevant exercises.
PREFERRED_MAX = 30  # Prefer profiles with no more than 30 relevant exercises. This makes the later manual top-10 selection easier.
N_USERS = 50        # Total number of synthetic users to generate.

# Get the project root folder.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Input cleaned exercise dataset.
CLEANED_DATASET = PROJECT_ROOT / "data" / "cleaned_exercise_dataset.csv" 

# Folder where the generated CSV files will be saved.
OUT_DIR = PROJECT_ROOT / "data" / "generated_user_profiles"                        

# Create the output folder if it does not already exist.
OUT_DIR.mkdir(parents=True, exist_ok=True)

# STEP 1 - LOAD THE CLEANED EXERCISE DATASET
df = pd.read_csv(CLEANED_DATASET)


# STEP 2 - Define the equipment profiles (training environments)

# Reason: A user normally trains in a specific environment where a typical set
# of equipment is available, rather than having randomly selected equipment. 
# Random equipment selection could create unusual combinations, such as
# Smith machine + Resistance Band only. Since a Smith machine is usually
# found in a gym environment, it would normally be available alongside
# other common equipment such as Dumbbells. 

# So each user is assigned to one of 6 realistic training environments. Every one
# is built only from equipment values that actually appear in the cleaned dataset.

# FOUR RULES USED TO ASSIGN EQUIPMENT TO PROFILES

# R1. "Body Weight" appears in EVERY profile.
#     Bodyweight exercises do not require specialised equipment, so they
#     can be performed in any training environment.

# R2. "Self-assisted" also appears in EVERY profile.
#     In this dataset, Self-assisted represents exercises where the user
#     reduces or controls the load using their own body or simple support.
#     It does not represent a specialised resistance machine.
#     This is different from "Assisted", which represents machine-assisted
#     exercises such as assisted pull-ups or dips.

# R3. Equipment such as Cable, Lever machines, Smith, Sled and Assisted
#     machines is typically associated with gym or facility-based training.
#     Therefore, these equipment types are not included in the home profiles.

# R4. "Weighted" represents bodyweight exercises performed with additional
#     resistance, such as a weight belt, vest, plate or other external load.
#     It is therefore included only in environments where additional
#     resistance is reasonably available, such as Garage, Functional and
#     Full Commercial gym profiles.

EQUIPMENT_PROFILES = {
    # Basic home setup with portable equipment.
    "Home - Minimal Setup": [
        "Body Weight",
        "Resistance Band",
        "Suspension Straps",
        "Self-assisted",
    ],
    # Home setup with dumbbells and portable equipment.
    "Home - Dumbbell Setup": [
        "Body Weight",
        "Dumbbell",
        "Resistance Band",
        "Suspension Straps",
        "Self-assisted",
    ],
    # Free-weight setup with barbells and dumbbells.
    "Garage - Free Weights": [
        "Body Weight",
        "Barbell",
        "Dumbbell",
        "Weighted",
        "Resistance Band",
        "Self-assisted",
    ],
    # Functional training setup with free weights, cables and sled.
    "Functional - Athletic Performance": [
        "Body Weight",
        "Dumbbell",
        "Barbell",
        "Resistance Band",
        "Suspension Straps",
        "Cable",
        "Sled",
        "Weighted",
        "Self-assisted",
    ],
    # Gym setup mainly using machines and cables.
    "Machine & Cable Gym": [
        "Body Weight",
        "Cable",
        "Lever (selectorized)",
        "Lever (plate loaded)",
        "Assisted",
        "Smith",
        "Self-assisted",
    ],
    # Full gym with every equipment type in the cleaned dataset.
    "Full Commercial Gym": [
        "Assisted",
        "Barbell",
        "Body Weight",
        "Cable",
        "Dumbbell",
        "Lever (plate loaded)",
        "Lever (selectorized)",
        "Resistance Band",
        "Self-assisted",
        "Sled",
        "Smith",
        "Suspension Straps",
        "Weighted",
    ],
}

# Short description of each equipment profile.
# This is later saved into equipment_profiles.csv.
EQUIPMENT_PROFILE_NOTES = {
    "Home - Minimal Setup":
        "Minimal home-training setup using portable equipment such as resistance bands, "
        "suspension straps and simple support for self-assisted exercises.",
    "Home - Dumbbell Setup":
        "Home training setup with dumbbells and portable equipment such as resistance bands and suspension straps.",
    "Garage - Free Weights":
        "Strength-focused garage setup with barbells, dumbbells, "
        "resistance bands and support for weighted and self-assisted exercises.",
    "Functional - Athletic Performance":
        "Functional training setup with free weights, cables, sleds and suspension equipment.",
    "Machine & Cable Gym":
        "Machine-focused gym setup with cables, selectorized and plate-loaded machines, a Smith machine and assisted equipment.",
    "Full Commercial Gym":
        "Full commercial gym with access to all equipment types in the cleaned dataset.",
}

# Check that every equipment value used above exists in the cleaned dataset.
_valid = set(df["equipment"].unique())

for _env, _items in EQUIPMENT_PROFILES.items():

     # Check that no unknown equipment has been added.
    assert set(_items) <= _valid, f"{_env} contains equipment not in the cleaned dataset"

    # Body Weight should be available in every environment.
    assert "Body Weight" in _items, f"{_env} must include Body Weight (rule R1)"

    # Self-assisted exercises are treated as available in every environment.
    assert "Self-assisted" in _items, f"{_env} must include Self-assisted (rule R2)"

    # Make sure the same equipment is not listed twice.
    assert len(_items) == len(set(_items)), f"{_env} lists the same equipment twice"


# STEP 3 - DEFINE ALLOWED EXPERIENCE LEVELS

# A user can perform exercises at their own level or below.

# Beginner     -> Beginner only
# Intermediate -> Beginner + Intermediate
# Advanced     -> Beginner + Intermediate + Advanced

ALLOWED_LEVELS = {
    "Beginner":     ["Beginner"],
    "Intermediate": ["Beginner", "Intermediate"],
    "Advanced":     ["Beginner", "Intermediate", "Advanced"],
}

# Used only to order relevant exercises for easier manual review later.
TARGET_DIFFICULTY = {"Beginner": 2, "Intermediate": 3, "Advanced": 4}



# STEP 4 - THE RELEVANCE RULE (THIS IS THE DEFINITION OF "RELEVANT EXERCISE")

# An exercise is relevant to a user when ALL FIVE of these hold:
#   1. exercise muscle_group == user's preferred_target_muscle_group
#   2. exercise level        is allowed for the user's experience_level
#   3. exercise equipment    is one of the user's available_equipments
#   4. exercise mechanics    == user's preferred_mechanics
#   5. exercise force        == user's preferred_force_type

def relevant_exercises(muscle_group, experience, equipment_list, mechanics, force):
    return df[
        (df["muscle_group"] == muscle_group)
        & (df["level"].isin(ALLOWED_LEVELS[experience]))
        & (df["equipment"].isin(equipment_list))
        & (df["mechanics"] == mechanics)
        & (df["force"] == force)
    ]



# STEP 5 - GENERATE EVERY POSSIBLE PROFILE AND TEST IT AGAINST THE DATASET

# 9 muscle groups x 3 levels x 6 equipment profiles x 2 mechanics x 2 forces
# = 648 candidate profiles.

# Each candidate profile is checked against the cleaned exercise dataset.
# Only profiles with at least 10 relevant exercises are kept.

# This removes combinations that are not sufficiently supported by the dataset,
# such as Back + Push, Chest + Pull, or Neck/Calves/Forearm + Compound when
# these combinations do not provide enough matching exercises.

candidates = []
for mg, exp, profile_name, mech, force in itertools.product(

    # All muscle groups in the dataset.
    sorted(df["muscle_group"].unique()),

    # Beginner, Intermediate and Advanced.
    ALLOWED_LEVELS.keys(),

    # Six equipment environments.
    EQUIPMENT_PROFILES.keys(),

    # Compound and Isolated.
    sorted(df["mechanics"].unique()),

    # Push and Pull.
    sorted(df["force"].unique()),
):
    # Count how many exercises match this possible user profile.
    n = len(relevant_exercises(mg, exp, EQUIPMENT_PROFILES[profile_name], mech, force))

    # Store the profile and its number of relevant exercises.
    candidates.append(
        dict(muscle_group=mg, experience_level=exp, equipment_profile=profile_name,
             mechanics=mech, force=force, n_relevant=n)
    )

# Convert all possible profiles into a DataFrame.
candidates = pd.DataFrame(candidates)

# Keep only profiles with at least 10 relevant exercises.
feasible = candidates[candidates["n_relevant"] >= MIN_RELEVANT].copy()

# Show how many profiles were tested and how many were usable.
print(f"Candidate profiles tested : {len(candidates)}")
print(f"Feasible (>= {MIN_RELEVANT} matches) : {len(feasible)}")


# STEP 6 - DECIDE HOW MANY USERS TO CREATE FOR EACH MUSCLE GROUP

# All 9 muscle groups are included.
# Larger catalogue groups receive more users, while smaller groups such as
# Forearm and Neck still remain represented in the evaluation.
ALLOCATION = {
    "Hips": 9, "Back": 8, "Chest": 7, "Thighs": 7, "Upper Arms": 6,
    "Shoulder": 5, "Calves": 4, "Forearm": 2, "Neck": 2,
}

# Check that the allocation adds up to exactly 50 users.
assert sum(ALLOCATION.values()) == N_USERS


# STEP 7 - SELECT 50 DIFFERENT USER PROFILES

# Within each muscle group, the script repeatedly selects the
# feasible profile that best improves overall balance, preferring:
# (a) an experience level that has been used least so far,
# (b) an equipment profile that has been used least so far,
# (c) a relevant pool size between 10 and 30, which is large enough
#     to satisfy the requirement but small enough for manual annotation.

# No profile is selected more than once, so all 50 users are distinct.

rng = random.Random(RANDOM_SEED)

# Count how many times each experience level has been selected.
exp_used = {k: 0 for k in ALLOWED_LEVELS}

# Count how many times each equipment profile has been selected.
eq_used = {k: 0 for k in EQUIPMENT_PROFILES}

# Final selected profiles will be stored here.
selected = []

# Select the required number of users for each muscle group.
for mg, quota in ALLOCATION.items():

    # Get all feasible profiles for this muscle group.
    pool = feasible[feasible["muscle_group"] == mg].to_dict("records")

    # Shuffle first so ties are reproducible but not always based on row order.
    rng.shuffle(pool)

    taken = 0

    while taken < quota and pool:

        # Sort profiles so under-used experience levels and equipment
        # environments are preferred.
        pool.sort(key=lambda r: (

            # Prefer experience levels used less often.
            exp_used[r["experience_level"]],                       # balance levels

            # Prefer equipment profiles used less often.
            eq_used[r["equipment_profile"]],                       # balance equipment

            # Prefer relevant pools between 10 and 30 exercises.
            0 if MIN_RELEVANT <= r["n_relevant"] <= PREFERRED_MAX else 1,

            # Prefer a pool close to 16 exercises.
            abs(r["n_relevant"] - 16),  
        ))   

        # Take the best profile.                            
        pick = pool.pop(0)

        # Update selection counts.
        exp_used[pick["experience_level"]] += 1
        eq_used[pick["equipment_profile"]] += 1

        # Save the selected profile.
        selected.append(pick)

        taken += 1

# Convert selected profiles into a DataFrame.
profiles = pd.DataFrame(selected).reset_index(drop=True)

# Add user IDs: U001 to U050.
profiles.insert(0, "user_id", [f"U{i:03d}" for i in range(1, len(profiles) + 1)])


# STEP 8 - CREATE THE OUTPUT TABLES

# Two main datasets are created:

# generated_user_profiles.csv: one row for each synthetic user

# user_relevant_exercises.csv: all relevant exercises found for each user

profile_rows, relevance_rows = [], []

for _, u in profiles.iterrows():

    # Get the equipment available to this user.
    eq_list = EQUIPMENT_PROFILES[u["equipment_profile"]]

    # Create one row for generated_user_profiles.csv.
    profile_rows.append({
        "user_id": u["user_id"],
        "preferred_target_muscle_group": u["muscle_group"],
        "experience_level": u["experience_level"],
        "available_equipments": ", ".join(eq_list),
        "preferred_mechanics": u["mechanics"],
        "preferred_force_type": u["force"],
    })

    # Find every relevant exercise for this user.
    matched = relevant_exercises(
                u["muscle_group"],
                u["experience_level"],
                eq_list,
                u["mechanics"],
                u["force"],
    ).copy()

    # Order the exercises to make later manual review easier.
    # This is only a suggested ordering.
    # It does NOT automatically choose the final top 10 ground-truth exercises.
    target = TARGET_DIFFICULTY[u["experience_level"]]

    # Exact experience-level matches are shown first.
    matched["_lvl"] = (matched["level"] != u["experience_level"]).astype(int)

    # Then prefer exercises with difficulty closer to the target difficulty.
    matched["_gap"] = (matched["difficulty_score"] - target).abs()

    # Finally sort alphabetically by exercise name.
    matched = matched.sort_values(["_lvl", "_gap", "exercise_name"])

    # Add every matched exercise to user_relevant_exercises.csv.
    for rank, (_, ex) in enumerate(matched.iterrows(), start=1):
        relevance_rows.append({
            "user_id": u["user_id"],
            "suggested_rank": rank,
            "exercise_id": ex["exercise_id"],
            "exercise_name": ex["exercise_name"],
            "muscle_group": ex["muscle_group"],
            "level": ex["level"],
            "difficulty_score": ex["difficulty_score"],
            "equipment": ex["equipment"],
            "mechanics": ex["mechanics"],
            "force": ex["force"],
            "muscles_worked": ex["muscles_worked"],
        })

# Create the final DataFrames.
generated_user_profiles = pd.DataFrame(profile_rows)
user_relevant = pd.DataFrame(relevance_rows)


# STEP 9 - CHECK THE GENERATED DATA BEFORE SAVING

# These checks make sure:
#   - exactly 50 users were created
#   - every user ID is unique
#   - every user has at least 10 relevant exercises
#   - all 50 user profiles are different

# Count relevant exercises for each user.
counts = user_relevant.groupby("user_id").size()

# Exactly 50 profiles must exist.
assert len(generated_user_profiles) == N_USERS, "must be exactly 50 users"

# User IDs must not repeat.
assert generated_user_profiles["user_id"].is_unique, "user_id must be unique"

# Every user must appear in the relevant-exercise dataset.
assert len(counts) == N_USERS, "every user must appear in the relevance table"

# Every user must have at least 10 relevant exercises.
assert counts.min() >= MIN_RELEVANT, "every user needs >= 10 relevant exercises"

# The 50 user profiles must all be different.
assert len(generated_user_profiles.drop_duplicates(
    subset=[c for c in generated_user_profiles.columns if c != "user_id"])) == N_USERS, \
    "all 50 profiles must be distinct"


# STEP 10 - PRINT A SUMMARY

print("\nAll checks passed.")
print(f"Relevant exercises per user -> min {counts.min()}, "
      f"median {int(counts.median())}, max {counts.max()}")
print("\nExperience level distribution:")
print(generated_user_profiles["experience_level"].value_counts().to_string())
print("\nEquipment profile distribution:")
print(profiles["equipment_profile"].value_counts().to_string())
print("\nMuscle group distribution:")
print(generated_user_profiles["preferred_target_muscle_group"].value_counts().to_string())
print("\nMechanics / Force distribution:")
print(generated_user_profiles["preferred_mechanics"].value_counts().to_string())
print(generated_user_profiles["preferred_force_type"].value_counts().to_string())


# STEP 11 - CREATE AN EQUIPMENT PROFILE REFERENCE TABLE

# This file records:
#   - equipment in each profile
#   - number of exercises reachable with that equipment
#   - number of feasible profiles
#   - number of final users assigned to it
env_ref = pd.DataFrame([
    {"equipment_profile": name,
     "n_equipment_types": len(items),
     "equipment_list": ", ".join(items),
     "exercises_reachable": int(df["equipment"].isin(items).sum()),
     "feasible_profiles": int((feasible["equipment_profile"] == name).sum()),
     "users_assigned": int((profiles["equipment_profile"] == name).sum()),
     "profile_description": EQUIPMENT_PROFILE_NOTES[name]}
    for name, items in EQUIPMENT_PROFILES.items()
])


# STEP 12 - SAVE ALL OUTPUT FILES

generated_user_profiles.to_csv(OUT_DIR / "generated_user_profiles.csv", index=False)
user_relevant.to_csv(OUT_DIR / "user_relevant_exercises.csv", index=False)
feasible.sort_values(["muscle_group", "experience_level"]).to_csv(OUT_DIR / "feasible_profile_combinations.csv", index=False)
env_ref.to_csv(OUT_DIR / "equipment_profiles.csv", index=False)
print("\nSaved: generated_user_profiles.csv, user_relevant_exercises.csv, "
      "feasible_profile_combinations.csv, equipment_profiles.csv")
