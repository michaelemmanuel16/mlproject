import os
import sys
from dataclasses import dataclass

from sklearn.ensemble import (
  AdaBoostRegressor,
  GradientBoostingRegressor,
  RandomForestRegressor
)
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models 

@dataclass
class ModelTrainerConfig:
  """Configuration class for model trainer."""
  trained_model_file_path=os.path.join('artifacts', 'model.pkl')
  
class ModelTrainer:
  """Class responsible for training and evaluating regression models."""
  def __init__(self):
    self.model_trainer_config=ModelTrainerConfig()
    
  def initiate_model_trainer(self, train_array, test_array):
    """
    Initiates the model training and evaluation process.

    Args:
        train_array (ndarray): Training data array with input features and target variable.
        test_array (ndarray): Test data array with input features and target variable.

    Returns:
        float: R-squared score of the best model.

    Raises:
        CustomException: If an exception occurs during model training or evaluation.
    """
    try:
      logging.info('Split training and test input data')
      X_train, y_train, X_test, y_test = (
        train_array[:,:-1], 
        train_array[:, -1],
        test_array[:,:-1],
        test_array[:,-1]
      )
      models = {
        "Linear Regression": LinearRegression(),
        "Lasso": Lasso(),
        "Ridge": Ridge(),
        "K-Neighbors Regressor": KNeighborsRegressor(),
        "Decision Tree": DecisionTreeRegressor(),
        "Random Forest Regressor": RandomForestRegressor(),
        "XGBRegressor": XGBRegressor(), 
        "AdaBoost Regressor": AdaBoostRegressor()
        }
      
      params = {
        "Linear Regression": {
          'fit_intercept':[True, False],
        },
        "K-Neighbors Regressor": {
          'n_neighbors':[5, 7, 9, 11],
          # 'weights':['uniform', 'distance'],
          'algorithm':['ball_tree', 'kd_tree', 'brute']
        },
        "Decision Tree": {
          'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
          # 'splitter':['best', 'random'],
          # 'max_features':['sqrt', 'log2']
          
        },
        "Random Forest Regressor": {
          # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
          # 'max_features':['sqrt', 'log2'],
          'n_estimators':[8, 16, 32, 64, 128, 256]
        },
        "XGBRegressor": {
          'learning_rate':[.1, .01, .05, .001],
          'n_estimators':[8, 16, 32, 64, 128, 256],
        }, 
        "AdaBoost Regressor": {
          'learning_rate':[.1, .01, .05, .001],
          'n_estimators':[8, 16, 32, 64, 128, 256],
        }
      }
      
      model_report:dict=evaluate_models(
        X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, models=models, params=params
        )
      
      # Get the best model score
      best_model_score = max(sorted(model_report.values()))
      
      # Get the best model name
      best_model_name = list(model_report.keys())[
        list(model_report.values()).index(best_model_score)
      ]
      
      best_model = models[best_model_name]
      
      if best_model_score<0.6:
        raise CustomException('No best model found')
      logging.info('Best model found both training and testing dataset')
      
      save_object(
        file_path=self.model_trainer_config.trained_model_file_path,
        obj=best_model
      )
      
      predicted=best_model.predict(X_test)
      
      r2_square = r2_score(y_test, predicted)
      return r2_square
      
    except Exception as e:
      return CustomException(e, sys)
    
