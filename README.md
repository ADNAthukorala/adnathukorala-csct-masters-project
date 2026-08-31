# Content-Based Exercise Recommender System for Personalised Strength Training

This repository contains the final codebase for an MSc project that designs and evaluates a content-based exercise recommendation system for strength training.

The system recommends exercises by comparing a user's training preferences with information stored for each exercise. Two different content-based recommendation approaches are implemented and evaluated using the same cleaned exercise dataset and ground-truth evaluation data.

## Project overview

The user can provide preferences such as:

- target muscle group
- experience level
- available equipment
- exercise mechanics
- force type

The system then ranks the exercises that best match those preferences and displays the top recommendations through a Streamlit web application.

The cleaned dataset used by both models contains **615 exercises**. Each exercise includes information such as exercise name, difficulty, level, equipment, muscle group, muscles worked, mechanics, force, preparation, and execution instructions.

## Recommendation models

### Model 1 - Multi-hot encoding and cosine similarity

Model 1 represents the main categorical features using multi-hot encoding. The user profile is encoded in the same feature space as the exercises, and cosine similarity is used to calculate how closely each exercise matches the user's preferences.

The model uses five recommendation features with the following final weights:

| Feature | Weight |
|---|---:|
| Muscle group | 5.0 |
| Experience level | 3.0 |
| Equipment | 3.0 |
| Mechanics | 2.0 |
| Force | 2.0 |

The exercises are ranked from the highest cosine-similarity score to the lowest. Exercise ID is used as a deterministic tie-breaker when scores are equal.

### Model 2 - TF-IDF and cosine similarity

Model 2 represents exercise information as weighted text. TF-IDF converts this text into numerical vectors, and cosine similarity is then used to compare the user's preferences with each exercise.

The feature importance is introduced by repeating important fields in the text representation:

| Feature | Repetition |
|---|---:|
| Muscle group | 5 |
| Muscles worked | 4 |
| Experience level | 3 |
| Equipment | 3 |
| Mechanics | 2 |
| Force | 2 |
| Exercise name | 1 |

Preparation and execution instructions are displayed to the user but are not included in the TF-IDF matching process because their longer text could dominate the shorter preference features.

Both models also provide an optional exact-match setting in the Streamlit interface. This can restrict the candidate exercises to those matching all selected structured preferences. The main offline evaluation is performed without strict filtering so that the ranking quality comes from the recommendation model itself.

## Repository structure

```text
adnathukorala-csct-masters-project/
│
├── creating_ground_truth_evaluation_dataset/
│   ├── data/
│   ├── docs/
│   ├── final_ground_truth_evaluation_dataset/
│   └── scripts/
│
├── evaluation/
│   ├── model1_evaluation_summary.csv
│   ├── model1_evaluation_equal_weights_summary.csv
│   ├── model2_evaluation_summary.csv
│   └── model2_evaluation_equal_weights_summary.csv
│
├── exercise_recommender_model_1/
│   ├── app.py
│   ├── data/
│   │   ├── raw/
│   │   └── processed/
│   ├── results/
│   ├── scripts/
│   ├── src/
│   ├── README.md
│   └── requirements.txt
│
├── exercise_recommender_model_2/
│   ├── app.py
│   ├── data/
│   │   ├── raw/
│   │   └── processed/
│   ├── results/
│   ├── scripts/
│   ├── src/
│   ├── README.md
│   └── requirements.txt
│
└── tests cases/
```

## Main folders

### `creating_ground_truth_evaluation_dataset/`

Contains the files used to create the synthetic evaluation profiles and the final ground-truth evaluation dataset.

The profile-generation script creates realistic user profiles using combinations that are supported by the exercise dataset. The final evaluation data contains:

- **50 synthetic user profiles**
- **10 ranked relevant exercises per user**
- **500 ground-truth rows in total**
- a review field explaining the relevance of each selected exercise

### `exercise_recommender_model_1/`

Contains the complete implementation of Model 1, including:

- Streamlit application
- raw and cleaned datasets
- feature encoding and weighting code
- recommendation logic
- filtering functions
- data-cleaning script
- testing script
- evaluation scripts
- saved evaluation results

### `exercise_recommender_model_2/`

Contains the complete implementation of Model 2, including:

- Streamlit application
- raw and cleaned datasets
- TF-IDF recommendation logic
- data-cleaning script
- testing script
- evaluation scripts
- saved evaluation results

### `evaluation/`

Contains the final summary CSV files used to compare the two models. Results are included for both the final weighted models and equal-weight versions.

## Evaluation method

Both models are evaluated using the same 50 user profiles and ground-truth dataset.

For each user, the model generates the **Top 8 recommendations**. These are compared with the **10 relevant exercises** stored in the ground truth.

The following ranking metrics are calculated:

- **Precision@8** - how many of the recommended exercises are relevant
- **Recall@8** - how many of the relevant exercises are successfully retrieved
- **F1@8** - combined measure of precision and recall
- **NDCG@8** - measures both relevance and ranking position

The evaluation also checks how often the recommendations match the user's selected:

- muscle group
- experience level
- available equipment
- mechanics
- force type

Separate evaluation scripts are included for the final feature weights and for equal weights so the effect of feature weighting can be compared.

## Requirements

Python **3.10 or newer** is recommended.

The main Python packages are:

- pandas
- numpy
- scikit-learn
- streamlit

Each model folder contains its own `requirements.txt` file.

## How to run Model 1

Open a terminal in the repository and move into the Model 1 folder:

```bash
cd exercise_recommender_model_1
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on macOS or Linux:

```bash
source .venv/bin/activate
```

On Windows:

```bash
.venv\Scripts\activate
```

Install the required packages:

```bash
python -m pip install -r requirements.txt
```

Run the Streamlit application:

```bash
streamlit run app.py
```

## How to run Model 2

From the repository root:

```bash
cd exercise_recommender_model_2
```

Create and activate a virtual environment if required, then install the dependencies:

```bash
python -m pip install -r requirements.txt
```

Run the Streamlit application:

```bash
streamlit run app.py
```

## Testing

Each model includes a testing script that checks the main parts of the recommendation process.

Run the tests from inside the relevant model folder:

```bash
python scripts/testing.py
```

The current codebase passes:

- **Model 1: 10/10 tests**
- **Model 2: 10/10 tests**

The tests cover areas such as dataset loading, feature representation, weighting, filtering, ranking, repeated results for the same profile, and deterministic tie handling.

## Running the evaluation

From inside either model folder, run:

```bash
python scripts/evaluation.py
```

This evaluates the final weighted version of the model and saves detailed, per-user, and summary CSV files in the model's `results/` folder.

To evaluate the equal-weight comparison version, run:

```bash
python scripts/evaluation_equal_weights.py
```

## Data cleaning

The cleaned exercise dataset is already included in each model, so the application can be run without repeating the cleaning process.

If the cleaned dataset needs to be rebuilt from the raw dataset, run from the relevant model folder:

```bash
python scripts/data_cleaning.py
```

## Streamlit output

The application displays ranked exercise recommendations together with useful exercise information, including:

- recommendation or similarity score
- exercise name
- muscle group and muscles worked
- experience level and difficulty
- required equipment
- mechanics and force type
- reason for the recommendation
- preparation instructions
- execution instructions

The recommendation results can also be downloaded as a CSV file from the application.

## Note

This project is an academic prototype designed to investigate content-based exercise recommendation and model evaluation. The recommendation score represents similarity between the user's preferences and exercise features. It should not be interpreted as a medical judgement or as a measure of exercise effectiveness or safety for an individual.
