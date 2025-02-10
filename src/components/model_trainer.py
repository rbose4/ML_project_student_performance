import os
import sys
from dataclasses import dataclass
from pathlib import Path

# from catboost import CatBoostRegressor
# from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (AdaBoostRegressor,
                               RandomForestRegressor, 
                               GradientBoostingRegressor)
from sklearn.metrics import r2_score

from src.exception import CustomException
from src.logger import logging
from src.utils import save_objects, evaluate_model

ARTIFACTS_DIR = 'artifacts'
MODEL_FILE = 'model.pkl'

@dataclass
class ModelTrainerConfig:
    trained_model_file_path:Path = Path(ARTIFACTS_DIR)/MODEL_FILE

class Model_Trainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
    
    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Split training and test input data")
            X_train,y_train,X_test,y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1])
            models = {
                "Linear Regression":LinearRegression(),
                "Random Forest":RandomForestRegressor(),
                "Decision Tree":DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                #"XGBoost Regressor":XGBRegressor(),
                #"CatBoost Regressor": CatBoostRegressor(verbose=False),
                "Adaboost Regressor":AdaBoostRegressor()
            }

            params={
                "Decision Tree": {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'splitter':['best','random'],
                    # 'max_features':['sqrt','log2'],
                },
                "Random Forest":{
                    # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                 
                    # 'max_features':['sqrt','log2',None],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Gradient Boosting":{
                    # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    # 'criterion':['squared_error', 'friedman_mse'],
                    # 'max_features':['auto','sqrt','log2'],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Linear Regression":{},
                # "XGBoost Regressor":{
                #     'learning_rate':[.1,.01,.05,.001],
                #     'n_estimators': [8,16,32,64,128,256]
                # },
                # "CatBoosting Regressor":{
                #     'depth': [6,8,10],
                #     'learning_rate': [0.01, 0.05, 0.1],
                #     'iterations': [30, 50, 100]
                # },
                "Adaboost Regressor":{
                    'learning_rate':[.1,.01,0.5,.001],
                    # 'loss':['linear','square','exponential'],
                    'n_estimators': [8,16,32,64,128,256]
                }
                
            }

            model_report:dict = evaluate_model(X_train,y_train, X_test, y_test, models, params)

            # To get the best model name from dict
            best_model_name = max(model_report, key=model_report.get)
            # To get the best model score from dict
            best_model_score = model_report[best_model_name]
            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("Best model cannot be found")
            logging.info(f"Best found model on both training and testing dataset")

            save_objects(file_path=self.model_trainer_config.trained_model_file_path,
                         obj = best_model)
            
            predicted = best_model.predict(X_test)
            r_sqaured = r2_score(y_test, predicted)
            return r_sqaured, best_model_name

        except Exception as e:
            raise CustomException(e,sys)
