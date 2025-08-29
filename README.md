# Movie Revenue Prediction

Machine learning model predicting box office revenue using budget and popularity metrics.

## Project Overview

This project analyzes movie industry data to predict box office revenue using linear regression with logarithmic transformation. The model identifies key financial indicators that drive movie success.

## Key Results

- **Model:** Linear Regression with log transformation
- **R² Score:** 0.65 (65% variance explained)
- **RMSE:** 1.18 (log scale)
- **Features:** Budget (log transformed), Popularity score
- Successfully improved predictions through feature engineering

## Technologies

- Python (pandas, scikit-learn, numpy, matplotlib)
- Linear Regression with log transformation
- Feature engineering and preprocessing
- Modular code architecture

## Project Structure
```
MovieRevenuePrediction/
├── data/ # Dataset storage
│ └── tmdb_movie_dataset.csv
├── src/ # Source code
│ ├── data_processing.py # Data cleaning and preprocessing
│ └── model.py # Model training and evaluation
├── notebooks/ # Jupyter notebooks
│ └── movie_revenue_analysis.ipynb
├── results/ # Model outputs
│ ├── plots/ # Visualization outputs
│ └── metrics.txt # Model performance metrics
├── requirements.txt # Project dependencies
└── README.md # Project documentation
```

## Features

- Automated data preprocessing pipeline
- Log transformation for handling skewed financial data
- Train/test split validation
- Comprehensive model evaluation metrics
- Revenue prediction for new movies

## Example Prediction

```python
# Predict revenue for a $50M budget movie with popularity score of 3
predicted_revenue = revenue_prediction(reg, 50000000, 3)
# Output: $116,991,401

## Usage

```bash
# Install dependencies
pip install -r requirements.txt

# Run the notebook
jupyter notebook movie_revenue_analysis.ipynb

# Or use the modules directly
from src.model import train_model, revenue_prediction
```
