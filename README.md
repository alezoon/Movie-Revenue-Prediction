# Movie Revenue Prediction
Machine learning project that predicts movie revenues based on budget and popularity

## Project Overview
This project was created to learn and understand the machine learning workflow.
The model predicts movie revenues using budget and popularity data from kaggle TMDB (Movie Database)

## Key Results
- **Model**: Linear Regression
- **R2 Score**: 0.65 (65% revenue variance)
- **Features**: Budget (log transformed), popularity score
- **Prediction Accuracy**: RMSE of 1.18


## Project structure
MovieRevenuePrediction/
├── data/
│ └── tmdb_movie_dataset.csv
├── src/
│ ├── data_processing.py # Data cleaning and preprocessing
│ └── model.py # Model training and evaluation
├── notebooks/
│ └── movie_revenue_analysis.ipynb
├── results/
│ ├── plots/ # Visualizations
│ └── metrics.txt # Model performance metrics
├── requirements.txt
└── README.md


## Getting Started

### Prerequisites
```bash
pip install -r requirements.txt