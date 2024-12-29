import os
import sys

import numpy as np # type: ignore
import pandas as pd # type: ignore

from dataclasses import dataclass
from sklearn.model_selection import train_test_split # type: ignore
from sklearn.preprocessing import StandardScaler, LabelEncoder # type: ignore
from fancyimpute import IterativeImputer # type: ignore
from sklearn.ensemble import RandomForestClassifier # type: ignore
from sklearn.impute import KNNImputer # type: ignore
from collections import defaultdict # type: ignore

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataPreprocessingConfig:
    train_data_path: str = os.path.join("artifacts", "train_data.pkl")
    test_data_path: str = os.path.join("artifacts", "test_data.pkl")

class MissingValueHandler:
    def handle_numerical(self, df: pd.DataFrame, method: str = "mean") -> pd.DataFrame:
        """Handles missing values in numerical columns based on the specified method."""
        try:
            logging.info(f"Handling missing values for numerical data using {method}")
            if method == "mean":
                return self.fill_mean(df)
            elif method == "median":
                return self.fill_median(df)
            elif method == "mode":
                return self.fill_mode(df)
            elif method == "constant":
                return self.fill_constant(df, value=0)  # Default constant value
            elif method == "ffill":
                return self.fill_forward(df)
            elif method == "bfill":
                return self.fill_backward(df)
            elif method == "interpolate":
                return self.fill_interpolation(df)
            elif method == "knn":
                return self.fill_knn(df)
            else:
                raise ValueError(f"Unsupported method {method} for numerical data")
        except Exception as e:
            raise CustomException(e, sys)
        
    def handle_categorical(self, df: pd.DataFrame, method: str = "mode") -> pd.DataFrame:
        """Handles missing values in categorical columns based on the specified method."""
        try:
            logging.info(f"Handling missing values for categorical data using {method}")
            if method == "mode":
                return self.fill_mode(df, categorical=True)
            elif method == "constant":
                return self.fill_constant(df, value="Missing", categorical=True)
            elif method == "ffill":
                return self.fill_forward(df, categorical=True)
            elif method == "bfill":
                return self.fill_backward(df, categorical=True)
            elif method == "knn":
                return self.fill_knn_categorical(df)
            elif method == "mice":
                return self.fill_mice_categorical(df)
            elif method == "random_forest":
                return self.fill_rf_categorical(df)
            elif method == "shd":
                return self.fill_shd_categorical(df)
            else:
                raise ValueError(f"Unsupported method {method} for categorical data")
        except Exception as e:
            raise CustomException(e, sys)

    def fill_mean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fills missing values with the mean of each column."""
        for column in df.select_dtypes(include=['number']).columns:
            df[column] = df[column].fillna(df[column].mean())
        return df

    def fill_median(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fills missing values with the median of each column."""
        for column in df.select_dtypes(include=['number']).columns:
            df[column] = df[column].fillna(df[column].median())
        return df

    def fill_mode(self, df: pd.DataFrame, categorical: bool = False) -> pd.DataFrame:
        """Fills missing values with the mode of each column."""
        columns = df.select_dtypes(include=['object']).columns if categorical else df.select_dtypes(include=['number']).columns
        for column in columns:
            df[column] = df[column].fillna(df[column].mode()[0])
        return df

    def fill_constant(self, df: pd.DataFrame, value, categorical: bool = False) -> pd.DataFrame:
        """Fills missing values with a constant value."""
        columns = df.select_dtypes(include=['object']).columns if categorical else df.select_dtypes(include=['number']).columns
        for column in columns:
            df[column] = df[column].fillna(value)
        return df

    def fill_forward(self, df: pd.DataFrame, categorical: bool = False) -> pd.DataFrame:
        """Fills missing values with the previous value in the column."""
        columns = df.select_dtypes(include=['object']).columns if categorical else df.select_dtypes(include=['number']).columns
        for column in columns:
            df[column] = df[column].fillna(method='ffill')
        return df

    def fill_backward(self, df: pd.DataFrame, categorical: bool = False) -> pd.DataFrame:
        """Fills missing values with the next value in the column."""
        columns = df.select_dtypes(include=['object']).columns if categorical else df.select_dtypes(include=['number']).columns
        for column in columns:
            df[column] = df[column].fillna(method='bfill')
        return df

    def fill_interpolation(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fills missing values using linear interpolation."""
        for column in df.select_dtypes(include=['number']).columns:
            df[column] = df[column].interpolate(method='linear')
        return df

    def fill_knn(self, df: pd.DataFrame) -> pd.DataFrame:
        """KNN Imputation for numerical data."""
        try:
            imputer = KNNImputer(n_neighbors=5)
            numeric_columns = df.select_dtypes(include=['number']).columns
            df[numeric_columns] = imputer.fit_transform(df[numeric_columns])
            return df
        except Exception as e:
            raise CustomException(e, sys)

    def fill_knn_categorical(self, df: pd.DataFrame) -> pd.DataFrame:
        """KNN Imputation for categorical data."""
        try:
            imputer = KNNImputer(n_neighbors=5)
            categorical_columns = df.select_dtypes(include=['object']).columns
            for column in categorical_columns:
                encoded = pd.get_dummies(df[column], prefix=column)
                imputed = imputer.fit_transform(encoded)
                df[column] = pd.DataFrame(imputed).idxmax(axis=1)
            return df
        except Exception as e:
            raise CustomException(e, sys)

    def fill_mice_categorical(self, df: pd.DataFrame) -> pd.DataFrame:
        """Multiple Imputation by Chained Equations for categorical data."""
        try:
            imputer = IterativeImputer()
            categorical_columns = df.select_dtypes(include=['object']).columns
            for column in categorical_columns:
                encoded = pd.get_dummies(df[column], prefix=column)
                imputed = imputer.fit_transform(encoded)
                df[column] = pd.DataFrame(imputed).idxmax(axis=1)
            return df
        except Exception as e:
            raise CustomException(e, sys)

    def fill_rf_categorical(self, df: pd.DataFrame) -> pd.DataFrame:
        """Random Forest Imputation for categorical data."""
        try:
            categorical_columns = df.select_dtypes(include=['object']).columns
            for column in categorical_columns:
                df_not_null = df[df[column].notnull()]
                df_null = df[df[column].isnull()]
                rf_model = RandomForestClassifier()
                X = df_not_null.drop(columns=[column])
                y = df_not_null[column]
                rf_model.fit(X, y)
                df.loc[df_null.index, column] = rf_model.predict(df_null.drop(columns=[column]))
            return df
        except Exception as e:
            raise CustomException(e, sys)

    def fill_shd_categorical(self, df: pd.DataFrame) -> pd.DataFrame:
        """Sequential Hot-Deck Imputation for categorical data."""
        try:
            categorical_columns = df.select_dtypes(include=['object']).columns
            for column in categorical_columns:
                mode_map = defaultdict(lambda: "Missing")
                for value in df[column].unique():
                    if pd.notna(value):
                        mode_map[value] = value
                df[column] = df[column].fillna(df[column].map(mode_map))
            return df
        except Exception as e:
            raise CustomException(e, sys)

class DataPreprocessor:
    def __init__(self, config: DataPreprocessingConfig):
        self.config = config
        self.missing_value_handler = MissingValueHandler()

    def load_data(self, file_path: str) -> pd.DataFrame:
        """Loads data from a CSV file."""
        try:
            logging.info(f"Loading data from {file_path}")
            return pd.read_csv(file_path)
        except Exception as e:
            raise CustomException(e, sys)
    
    def drop_high_missing_columns(self, df: pd.DataFrame, threshold: float = 0.25) -> pd.DataFrame:
        """Drops columns with missing values above a certain threshold."""
        try:
            logging.info(f"Dropping columns with missing values above {threshold}")
            missing_values = df.isnull().mean()
            columns_to_drop = missing_values[missing_values > threshold].index
            df = df.drop(columns=columns_to_drop)
            logging.info("Columns dropped successfully")
            return df
        except Exception as e:
            raise CustomException(e, sys)

    def handle_missing_values(self, df: pd.DataFrame, numerical_method: str = "mean", categorical_method: str = "mode") -> pd.DataFrame:
        """Handles missing values by delegating to MissingValueHandler."""
        try:
            logging.info("Handling missing values using numerical method as {numerical_method} and categorical method as {categorical_method}")
            df = self.missing_value_handler.handle_numerical(df, method=numerical_method)
            df = self.missing_value_handler.handle_categorical(df, method=categorical_method)
            logging.info("Missing values handled successfully")
            return df
        except Exception as e:
            raise CustomException(e, sys)

    def encode_categorical_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encodes categorical features using Label Encoding."""
        try:
            logging.info("Label Encoding the categorical data")
            label_encoders = {}
            for column in df.select_dtypes(include=['object']).columns:
                le = LabelEncoder()
                df[column] = le.fit_transform(df[column])
                label_encoders[column] = le
            logging.info("Label Encoding completed")
            return df
        except Exception as e:
            raise CustomException(e, sys)

    def scale_data(self, X: np.ndarray) -> np.ndarray:
        """Scales data using Standard Scaler."""
        try:
            logging.info("Scaling data")
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            logging.info("Data scaled successfully")
            return X_scaled
        except Exception as e:
            raise CustomException(e, sys)

    def split_data(self, df: pd.DataFrame, target_column: str):
        """Splits the data into train and test sets."""
        try:
            logging.info("Splitting data into train and test sets")
            X = df.drop(columns=[target_column])
            y = df[target_column]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            logging.info("Data split successfully")
            return X_train, X_test, y_train, y_test
        except Exception as e:
            raise CustomException(e, sys)

    def preprocess_data(self, file_path: str, target_column: str, numerical_method: str = "mean", categorical_method: str = "mode", threshold_for_dropping_cols: float = 0.25):
        """Main method to preprocess data."""
        try:
            logging.info("Starting data preprocessing")
            df = self.load_data(file_path)
            df = self.drop_high_missing_columns(df, threshold=threshold_for_dropping_cols)
            df = self.handle_missing_values(df, numerical_method=numerical_method, categorical_method=categorical_method)
            df = self.encode_categorical_data(df)

            X_train, X_test, y_train, y_test = self.split_data(df, target_column)

            X_train_scaled = self.scale_data(X_train)
            X_test_scaled = self.scale_data(X_test)

            # Save processed data
            save_object(self.config.train_data_path, (X_train_scaled, y_train))
            save_object(self.config.test_data_path, (X_test_scaled, y_test))

            logging.info("Data preprocessing completed successfully")
            
            return self.config.train_data_path, self.config.test_data_path
        except Exception as e:
            raise CustomException(e, sys)

    def preprocessing_test_data_for_submission(self, file_path: str, numerical_method: str = "mean", categorical_method: str = "mode"):
        """Preprocesses test data for submission."""
        try:
            logging.info("Starting data preprocessing for test data for submission")
            df = self.load_data(file_path)

            if 'id' in df.columns:
                id_column = df['id']
                df = df.drop(columns=['id'])
                logging.info("ID column separated from the test data")
            else:
                id_column = None
                logging.info("ID column not found in the test data")

            df = df.drop('Policy Start Date', axis=1)

            df = self.handle_missing_values(df, numerical_method=numerical_method, categorical_method=categorical_method)
            df = self.encode_categorical_data(df)
            df = self.scale_data(df)
            logging.info("Data preprocessing for test data for submission completed successfully")
            
            return df, id_column
        except Exception as e:
            raise CustomException(e, sys)