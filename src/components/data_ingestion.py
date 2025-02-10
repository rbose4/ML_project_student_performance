import os
import sys
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from pathlib import Path

from src.exception import CustomException
from src.logger import logging
from src.components.data_transformation import DataTransformation, DataTransformationConfig
from src.components.model_trainer import ModelTrainerConfig, Model_Trainer


ARTIFACTS_DIR = 'artifacts'
TRAIN_FILE = 'train.csv'
TEST_FILE = 'test.csv'
RAW_FILE = 'data.csv'

@dataclass
class DataIngestionConfig:
    ''' Configuration for data ingestion '''
    train_data_path: Path = Path(ARTIFACTS_DIR)/TRAIN_FILE
    test_data_path: Path = Path(ARTIFACTS_DIR)/TEST_FILE
    raw_data_path: Path = Path(ARTIFACTS_DIR)/RAW_FILE


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
    
    def initiate_data_ingestion(self):
        logging.info("Entered data ingestion method or component")
        try:
            df = pd.read_csv('notebook/data/stud.csv')
            logging.info('Read the dataset as dataframe')

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            logging.info("Train test split initiated")
            train_set, test_set = train_test_split(df,test_size=0.2, random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Ingestion of the data is completed")

            return(self.ingestion_config.train_data_path,
                   self.ingestion_config.test_data_path)
        except Exception as e:
            CustomException(e, sys)




if __name__ =="__main__":
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()
    print("Training data path", train_data)

    data_transformation = DataTransformation()
    train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_path=train_data, test_path=test_data)
    model_trainer = Model_Trainer()
    score, model_name = model_trainer.initiate_model_trainer(train_arr, test_arr)
    print(f"Best model: {model_name}, R Sqaured value: {score} ")