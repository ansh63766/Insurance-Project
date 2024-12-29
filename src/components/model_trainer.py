import os
import sys
from dataclasses import dataclass
from sklearn.ensemble import RandomForestRegressor  # type: ignore
from sklearn.svm import SVR  # type: ignore
from sklearn.linear_model import LinearRegression  # type: ignore
from sklearn.tree import DecisionTreeRegressor  # type: ignore
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score  # type: ignore
from sklearn.model_selection import GridSearchCV  # type: ignore
from sklearn.metrics import root_mean_squared_log_error # type: ignore

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, load_object

@dataclass
class ModelTrainerConfig:
    trained_model_path: str = os.path.join('artifacts', 'trained_model.pkl')
    model_performance_path: str = os.path.join('artifacts', 'model_performance.txt')

class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def train_and_evaluate_model(self, X_train, y_train, X_test, y_test):
        """Train multiple models with different hyperparameters, evaluate them, and select the best one."""
        try:
            logging.info("Training multiple models")
            
            models = {
                'LinearRegression': LinearRegression(),
                'RandomForest': RandomForestRegressor(random_state=42),
                # 'SVR': SVR(),
                # 'DecisionTree': DecisionTreeRegressor(random_state=42),
            }

            # Hyperparameter grids for different models
            param_grids = {
                'LinearRegression': {},
                'RandomForest': {
                    'n_estimators': [100, 150, 200],
                    'max_depth': [2, 10, 15],
                    'min_samples_split': [2, 5, 10]
                },
                # 'SVR': {
                #     'C': [0.1, 1],
                #     'kernel': ['linear', 'rbf']
                # },
                # 'RandomForest': {
                #     'n_estimators': [100, 150, 200, 250, 300],
                #     'max_depth': [2, 4, 5, 8, 10, 15],
                #     'min_samples_split': [2, 4, 5, 8, 10]
                # },
                # 'DecisionTree': {
                #     'max_depth': [2, 4, 5, 8, 10, 15],
                #     'min_samples_split': [2, 4, 5, 8, 10]
                # }
            }

            from sklearn.metrics import make_scorer # type: ignore
            rmsle_scorer = make_scorer(root_mean_squared_log_error, greater_is_better=False)

            best_model = None
            best_score = float('-inf')
            best_model_name = ""

            # Performance log for each model
            performance_log = []

            for model_name, model in models.items():
                logging.info(f"Training {model_name}")
                # Use RMSLE as the scoring metric in GridSearchCV
                grid_search = GridSearchCV(
                    model, 
                    param_grids[model_name], 
                    cv=10, 
                    scoring=rmsle_scorer, 
                    n_jobs=10, 
                    verbose=3
                )
                grid_search.fit(X_train, y_train)
                
                best_model_for_this = grid_search.best_estimator_
                best_score_for_this = -grid_search.best_score_

                logging.info(f"Best {model_name} model parameters: {grid_search.best_params_}")
                logging.info(f"Best {model_name} model RMSLE score: {best_score_for_this}")

                # Check if this model has the best score
                if best_score_for_this > best_score:
                    best_score = best_score_for_this
                    best_model = best_model_for_this
                    best_model_name = model_name
                    logging.info(f"New best model found: {model_name}")

                # Evaluate each model on the test data
                y_pred = best_model_for_this.predict(X_test)
                mae = mean_absolute_error(y_test, y_pred)
                mse = mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                rmsle = root_mean_squared_log_error(y_test, y_pred)

                performance_log.append(f"Model: {model_name}\n")
                performance_log.append(f"Best Parameters: {grid_search.best_params_}\n")
                performance_log.append(f"MAE: {mae}\n")
                performance_log.append(f"MSE: {mse}\n")
                performance_log.append(f"R2 Score: {r2}\n")
                performance_log.append(f"RMSLE: {rmsle}\n")
                performance_log.append("="*80 + "\n")

            # Save the best model
            save_object(self.config.trained_model_path, best_model)
            logging.info(f"Best model saved at {self.config.trained_model_path}")

            # Save model performance log
            self.save_model_performance(performance_log)

            return best_model

        except Exception as e:
            logging.error("Error occurred during model training and evaluation")
            raise CustomException(e, sys)

    def save_model_performance(self, performance_log):
        """Save performance of all models and mention the best model in a readable text file."""
        try:
            with open(self.config.model_performance_path, 'w') as f:
                # Write all model performance details
                f.writelines(performance_log)
                
                # Write the best model information at the end
                f.write("="*80 + "\n")
                f.write(f"Best Model: {performance_log[-7]}\n")  # The last model's name
                f.write(f"Best Model MAE: {performance_log[-6]}\n")  # Last MAE
                f.write(f"Best Model RMSLE: {performance_log[-5]}\n")  # Last RMSLE
                f.write("="*80 + "\n")

            logging.info(f"Model performance saved at {self.config.model_performance_path}")
        except Exception as e:
            logging.error("Error occurred while saving model performance")
            raise CustomException(e, sys)

    def load_trained_model(self):
        """Load the trained model from file."""
        try:
            logging.info(f"Loading trained model from {self.config.trained_model_path}")
            model = load_object(self.config.trained_model_path)
            return model
        except Exception as e:
            logging.error("Error occurred while loading the trained model")
            raise CustomException(e, sys)
