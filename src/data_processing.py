import pandas as pd
import numpy as np

def load_and_clean_data(filepath):
    """
    Loads dataset and drops unnessecary columns
    Pre-condition:
        :param filepath: A string of the datas path
    Post-condition:
        None
    Return:
        Movie dataset with dropped columns
    """
    dataset = pd.read_csv(filepath)

    movie_data = dataset.drop(columns=[
    'id', 'imdb_id', 'original_title', 'cast', 'director',
    'tagline', 'overview', 'keywords', 'genres',
    'production_companies', 'popularity_level'
    ])

    return movie_data



def apply_log_transforms(movie_data):
    """
    Applies log transfomration to budget and revenue
    Pre-condition:
        :param movie_data: Valid Data
    Post-condition:
        Unchanged original movie_data
    Return:
        Transformed movie_data
    """
    movie_data = movie_data.copy()
    movie_data['log_budget'] = np.log1p(movie_data['budget'])
    movie_data['log_revenue'] = np.log1p(movie_data['revenue'])

    return movie_data



def feature_prer(movie_data):
    """
    Prepare X, y for modeling
    Pre-condition:
        :param movie_data: Transformed data
    Post-condition:
        None
    Return:
        X, y datas
    """
    X= movie_data[['log_budget', 'popularity']].values
    y = movie_data['log_revenue'].values

    return X,y