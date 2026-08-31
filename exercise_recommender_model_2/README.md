# Exercise Recommendation System (Model 2)

A content-based exercise recommender with a Streamlit web application. The user
selects their training preferences and the model returns a ranked list of
matching exercises from a cleaned dataset of 615 gym exercises.

The model does not judge whether an exercise is good or bad. It only measures how
closely each exercise matches the preferences the user has entered.

## How it works

1. Each exercise is turned into a short piece of text made from its muscle group,
   muscles worked, level, equipment, mechanics, force and name. More important
   fields are repeated more times so they carry more weight: muscle group ×5,
   muscles worked ×4, level and equipment ×3, mechanics and force ×2, name ×1.
2. TF-IDF converts that text into numbers.
3. The user's preferences are written as the same kind of text and converted
   using the same TF-IDF model.
4. Cosine similarity measures how close the user's preferences are to each
   exercise. Nothing is added to this score, so the ranking comes only from the
   text comparison.
5. The exercises are sorted from the highest score to the lowest and the top
   results are shown. Exercises with the same score are ordered by exercise ID,
   so the ranking is always the same when the model is run again.

Preparation and execution instructions are left out of the matching step, because
this long text would drown out the short preference fields. They are still shown
with each recommendation.

There is also an optional exact match setting. When it is switched on, exercises
that do not match every selected category are removed before ranking. If nothing
is left, the app falls back to normal ranking and tells the user.

## Inputs and outputs

**Inputs** (chosen in the app):

- Target muscle group — required, one of 9
- Experience level — Beginner, Intermediate, Advanced, or Any
- Available equipment — any of 13 types, or none to allow all
- Exercise mechanics — Compound, Isolated, or Any
- Force type — Push, Pull, or Any
- Specific muscles or keywords — optional free text
- Number of recommendations — 3 to 15
- Require exact matches — on or off

**Outputs:**

- A ranked list of exercises, each with a match percentage
- Details for each exercise: muscle group, muscles worked, level, difficulty
  score, equipment, mechanics and force
- A short reason explaining why the exercise was recommended
- Preparation and execution instructions
- A CSV download of the results

The percentage is a text similarity score. It is not a probability and not a
measure of how effective an exercise is.

## Requirements

Python 3.10 or newer, plus:

- pandas
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
