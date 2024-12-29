import os
import sys
from dataclasses import dataclass

import pandas as pd # type: ignore

from src.exception import CustomException
from src.logger import logging
from src.utils import load_object

from src.components.data_preprocessing import DataPreprocessingConfig, DataPreprocessor, MissingValueHandler

@dataclass
class SubmissionConfig:
    raw_data_path: str = os.path.join('artifacts', 'test_data_for_submission.csv')
    submission_path: str = os.path.join('artifacts', 'submission.csv')
    trained_model_path: str = os.path.join('artifacts', 'trained_model.pkl')

class Submission:
    def __init__(self, config: SubmissionConfig):
        self.config = config

    def load_test_data_file(self):
        logging.info("Loading test data for submission")
        try:
            df = pd.read_csv(self.config.raw_data_path)
            logging.info(f"Loaded dataset from {self.config.raw_data_path}")

            return df

        except Exception as e:
            logging.error("Error occurred during submission")
            raise CustomException(e, sys)

    def load_model(self):
        logging.info("Loading trained model")
        try:
            model = load_object(self.config.trained_model_path)
            logging.info(f"Loaded trained model from {self.config.trained_model_path}")

            return model

        except Exception as e:
            logging.error("Error occurred during submission")
            raise CustomException(e, sys)    
    
    def predict(self, model, test_data):
        logging.info("Predicting on test data")
        try:
            predictions = model.predict(test_data)
            logging.info("Predictions completed")

            return predictions

        except Exception as e:
            logging.error("Error occurred during submission")
            raise CustomException(e, sys)
    
    def save_submission(self, submission_df):
        logging.info("Saving submission file")
        try:
            submission_df.to_csv(self.config.submission_path, index=False)
            logging.info(f"Saved submission file to {self.config.submission_path}")

        except Exception as e:
            logging.error("Error occurred during submission")
            raise CustomException(e, sys)

    def main_func(self):
        """Main function to execute the entire submission pipeline."""
        try:
            # Load the test data
            test_data = self.load_test_data_file()

            # Preprocess the test data for submission
            config_preprocessing = DataPreprocessingConfig()
            preprocessor = DataPreprocessor(config_preprocessing)

            test_data, id_column = preprocessor.preprocessing_test_data_for_submission(
                file_path=self.config.raw_data_path,
                numerical_method="mean",
                categorical_method="mode"
            )

            # Load the trained model
            model = self.load_model()

            # Make predictions on the processed test data
            predictions = self.predict(model, test_data)

            # Create a DataFrame for submission with the id column and the predictions
            submission_df = pd.DataFrame({"id": id_column, "Premium Amount": predictions})

            # Save the submission DataFrame as a CSV
            self.save_submission(submission_df)

        except Exception as e:
            logging.error("Error occurred during the main submission process")
            raise CustomException(e, sys)

if __name__ == "__main__":
    try:
        # Create an instance of the Submission class with the desired config
        submission_config = SubmissionConfig()
        submission = Submission(submission_config)

        submission.main_func()

    except Exception as e:
        logging.error("Error occurred while running the submission process")
        raise CustomException(e, sys)
