import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def train_model(X,y, test_size=0.2, random_state=42):
    """
    Train sklearn linear regression model
    Pre-condition:
        :param X: Input data
        :param y: expected output data
    Post-condition:
        None
    Return:
        reg: Trained model
        X_train, X_test, y_train, y_test: Train test split data
    """

    X_train, X_test, y_train, y_test =  train_test_split(X,y, test_size=test_size, random_state=random_state)

    reg = LinearRegression()
    reg.fit(X_train,y_train)

    return reg, X_train, X_test, y_train, y_test



def model_evaluation(reg, X_test, y_test):
    """
    Evaluate model and return metrics
    Pre-condition:
        :param reg: Trained model
        :param X_test: Input test data
        :param y_test: Output test data
    Post-condition
        None
    Return:
        Metrics
    """

    y_pred = reg.predict(X_test)

    metrics = {
        'mse': mean_squared_error(y_test, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
        'mae': mean_absolute_error(y_test, y_pred),
        'r2': r2_score(y_test, y_pred),
        'predictions': y_pred
    }
    return metrics



def predict_revenue(reg, budget_dollars, popularity):
    """
    Predicts the revenue for a new movie
    Pre-condition:
        :param reg: Trained model
        :param budget_dollars: Int or Float, movie budget
        :param popularity: Int or Float, popularity of movie (0 < x < 10)
    Post-condition:
        None
    Return:
        predicted_revenue
    """

    log_budget = np.log1p(budget_dollars)
    predicted_log_revenue = reg.predict([[log_budget, popularity]])[0]
    predicted_revenue = np.expm1(predicted_log_revenue)
    
    return predict_revenue