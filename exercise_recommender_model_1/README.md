# Exercise Recommendation System (Model 1)

A content-based exercise recommender with a Streamlit web application. The user
selects their training preferences and the model returns a ranked list of
matching exercises from a cleaned dataset of 615 gym exercises.

The model does not judge whether an exercise is good or bad. It only measures how
closely each exercise matches the preferences the user has entered.

## How it works

The model compares the user with every exercise using five features: muscle
group, experience level, equipment, mechanics and force.

1. The selected preferences are collected into a single user profile.
2. The profile and all the exercises are turned into 0/1 columns using multi-hot
   encoding, so both are described in the same way.
3. Each feature is multiplied by a weight: muscle group 5.0, level 3.0,
   equipment 3.0, mechanics 2.0, force 2.0.
4. Cosine similarity gives each exercise a score for how close it is to the
   profile.
5. The exercises are ranked from the highest score to the lowest and the top
   results are shown. Exercises with the same score are ordered by exercise ID,
   so the ranking is always the same when the model is run again.

There is also an optional exact match setting. When it is switched on, only
exercises matching every selected category are considered. If that combination
returns nothing, the app searches the whole dataset again and tells the user.
The setting only decides which exercises are eligible. It never changes how the
scores are calculated.

## Inputs and outputs

**Inputs** (chosen in the app):

- Target muscle group — required, one of 9
- Experience level — Beginner, Intermediate, Advanced, or Any
- Available equipment — any of 13 types, or none to allow all
- Exercise mechanics — Compound, Isolated, or Any
- Force type — Push, Pull, or Any
- Number of recommendations — 3 to 15
- Require exact matches — on or off

**Outputs:**

- A ranked list of exercises, each with a similarity percentage
- Details for each exercise: muscle group, muscles worked, level, difficulty
  score, equipment, mechanics and force
- A short reason explaining why the exercise was recommended
- Preparation and execution instructions
- A CSV download of the results

The similarity score is not normalised, so the highest possible score depends on
how many preferences were selected. Exercises that match every selected category
can therefore share the same score.

## Requirements

Python 3.10 or newer, plus:

- pandas
- numpy
- scikit-learn
- streamlit

## Installation

Open a terminal in the project folder and create a virtual environment:

```bash
python -m venv .venv
```

Activate it on macOS or Linux:

```bash
source .venv/bin/activate
```

Or on Windows:

```bash
.venv\Scripts\activate
```

Install the packages:

```bash
python -m pip install -r requirements.txt
```

## How to run

Start the Streamlit application from the project folder:

```bash
streamlit run app.py
```

Streamlit shows a local address, normally `http://localhost:8501`, which can be
opened in a browser.

The cleaned dataset is already included, so nothing else needs preparing. These
scripts can also be run from the project folder if needed:

```bash
python scripts/data_cleaning.py               # rebuilds the cleaned dataset from the raw data
python scripts/evaluation.py                  # evaluates the model and writes CSV files to results/
python scripts/evaluation_equal_weights.py    # same evaluation using equal feature weights
python scripts/testing.py                     # runs the tests for the model
```
