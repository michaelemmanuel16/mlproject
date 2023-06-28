import os 
import sys
import numpy as np
import pandas as pd
import dill
from sklearn.metrics import r2_score

from src.exception import CustomException


def save_object(file_path, obj):
    """
    Function to create filepath and save object
    Parameters
    ----------
    file_path :
        
    obj :
        

    Returns pickle (.pkl) file to specified file path
    -------

    """
    try:
      dir_path = os.path.dirname(file_path)
      
      os.makedirs(dir_path, exist_ok=True)
      
      with open(file_path, 'wb') as file_obj:
        dill.dump(obj, file_obj)
      
    except Exception as e:
      raise CustomException(e, sys)
  
  
  
def evaluate_models(X_train, y_train, X_test, y_test, models):
    """
    Function to evaluate model performance
    Parameters
    ----------
    X_train : training dataset features matrix
        
    y_train : training dataset target vector
        
    X_test : test dataset features matrix
        
    y_test : test dataset target vector
        
    models : dictionary of models
        

    Returns dictionary of models and r2_score
    -------

    """
    try:
      report={}
      for i in range(len(list(models))):
        model = list(models.values())[i]
        
        model.fit(X_train, y_train) # Train model
        
        y_train_pred = model.predict(X_train)
        
        y_test_pred = model.predict(X_test)
        
        # Evaluate Train dataset
        train_model_score = r2_score(y_train, y_train_pred)
        # Evaluate Test dataset
        test_model_score = r2_score(y_test, y_test_pred)
          
        report[list(models.keys())[i]] = test_model_score
        
        return report
    except Exception as e:
      raise CustomException(e, sys)