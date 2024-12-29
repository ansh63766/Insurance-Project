import os
import sys
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object

import pandas as pd # type: ignore
import numpy as np # type: ignore

from src.components.data_preprocessing import DataPreprocessingConfig, DataPreprocessor, MissingValueHandler
from src.components.model_trainer import ModelTrainerConfig, ModelTrainer

@dataclass
class DataIngestionConfig:
    raw_data_path: str = os.path.join('artifacts', 'raw_explored_data.csv')
    loaded_data_path: str = os.path.join('artifacts', 'loaded_data.csv')

class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def initiate_data_ingestion(self):
        logging.info("Starting data ingestion process")
        try:
            df = pd.read_csv(self.config.raw_data_path)
            logging.info(f"Loaded dataset from {self.config.raw_data_path}")
            
            df.to_csv(self.config.loaded_data_path, index=False)
            logging.info(f"Saved loaded data to {self.config.loaded_data_path}")
            
            return self.config.loaded_data_path

        except Exception as e:
            logging.error("Error occurred during data ingestion")
            raise CustomException(e, sys)

if __name__ == "__main__":
    config = DataIngestionConfig()
    data_ingestion = DataIngestion(config)
    loaded_data_path = data_ingestion.initiate_data_ingestion()

    config_preprocessing = DataPreprocessingConfig()
    preprocessor = DataPreprocessor(config_preprocessing)
    train_data_path, test_data_path = preprocessor.preprocess_data(file_path = loaded_data_path, target_column="Premium Amount", numerical_method="mean", categorical_method="mode", threshold_for_dropping_cols=0.5) 

    X_train, y_train = load_object(train_data_path)
    X_test, y_test = load_object(test_data_path)

    model_trainer_config = ModelTrainerConfig()
    model_trainer = ModelTrainer(model_trainer_config)
    best_model = model_trainer.train_and_evaluate_model(X_train, y_train, X_test, y_test)