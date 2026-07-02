# California Housing Price Prediction

A simple machine learning project to predict California house prices using Python and scikit-learn.

## Project Structure

```text
California-Housing-Price/
├── data/
│   ├── README.md
│   └── housing_sample.csv
├── models/
│   └── README.md
├── notebooks/
│   └── original_notebook.ipynb
├── outputs/
│   ├── actual_vs_predicted_linear_regression.png
│   ├── actual_vs_predicted_random_forest_regression.png
│   ├── housing_predictions_linear_regression.csv
│   ├── housing_predictions_random_forest.csv
│   └── model_metrics.csv
├── src/
│   └── train.py
├── README.md
└── requirements.txt
```

## Models Used

- Linear Regression
- Random Forest Regression

## Dataset

The original notebook uses the California housing dataset from the Hands-On Machine Learning GitHub repository.

The training script will try to download the full dataset automatically and save it as:

```text
data/housing.csv
```

A small sample file is included only so the project folder is complete.

## How to Run

Install the required libraries:

```bash
pip install -r requirements.txt
```

Run the training script:

```bash
python src/train.py
```

## Outputs

The project saves:

- model metrics in `outputs/model_metrics.csv`
- prediction CSV files in `outputs/`
- actual vs predicted plots in `outputs/`
- trained models in `models/`

## Current Results From Uploaded Prediction Files

| model                    |    rmse |     mae |   r2_score |
|:-------------------------|--------:|--------:|-----------:|
| Linear Regression        | 70059.2 | 50670.5 |     0.6254 |
| Random Forest Regression | 48781.9 | 31465.2 |     0.8184 |

