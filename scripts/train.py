import gc
import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Tuple, Union

import joblib
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_selection import RFE
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

from correlation import calculate_correlation_with_target


class ClassifierTrainer:
    """
    A class for training and evaluating classifiers.
    """

    def __init__(
        self,
        classifiers: List[Tuple[str, Any, Dict[str, Union[int, float]]]],
        data_dir: str,
    ):
        """
        Initialize the ClassifierTrainer.

        Args:
            classifiers (List[Tuple[str, Any, Dict[str, Union[int, float]]]): List of classifiers with names and parameters.
            data_dir (str): Directory path where the training data is located.
        """
        self.classifiers = classifiers
        self.data_dir = data_dir

    def load_training_data(self, train_data_parquet_file: str) -> pd.DataFrame:
        """
        Load training data from a Parquet file.

        Args:
            train_data_parquet_file (str): Name of the Parquet file containing training data.

        Returns:
            pd.DataFrame: Loaded training data as a DataFrame.
        """
        train_data = pd.read_parquet(
            os.path.join(self.data_dir, train_data_parquet_file)
        )
        return train_data

    def split_data(
        self, train_data: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Split the training data into train and validation sets.

        Args:
            train_data (pd.DataFrame): The training data as a DataFrame.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]: Split data (X_train, X_val, y_train, y_val).
        """
        X_train, X_val, y_train, y_val = train_test_split(
            train_data.iloc[:, :-1],
            train_data["target"],
            test_size=0.2,
            random_state=42,
        )
        return X_train, X_val, y_train, y_val

    def train_classifier(
        self,
        name: str,
        classifier: Any,
        params: Dict[str, Union[int, float]],
        X_train_encoded: pd.DataFrame,
        y_train: pd.Series,
    ) -> GridSearchCV:
        """
        Train a classifier using hyperparameter tuning.

        Args:
            name (str): Name of the classifier.
            classifier (Any): The classifier to train.
            params (Dict[str, Union[int, float]]): Hyperparameters for grid search.
            X_train_encoded (pd.DataFrame): Encoded training data.
            y_train (pd.Series): Training labels.

        Returns:
            GridSearchCV: Trained classifier with the best hyperparameters.
        """
        rfe = RFE(estimator=classifier, n_features_to_select=10)
        imputer = SimpleImputer(strategy="median")
        pipeline = Pipeline(
            [
                ("imputer", imputer),
                ("feature_selection", rfe),
                ("classifier", classifier),
            ]
        )

        grid = GridSearchCV(pipeline, param_grid=params, cv=3, n_jobs=-1, verbose=3)
        grid.fit(X_train_encoded, y_train)

        return grid

    def evaluate_classifier(
        self, classifier: GridSearchCV, X_val_encoded: pd.DataFrame, y_val: pd.Series
    ) -> float:
        """
        Evaluate the classifier using Gini coefficient.

        Args:
            classifier (GridSearchCV): Trained classifier.
            X_val_encoded (pd.DataFrame): Encoded validation data.
            y_val (pd.Series): Validation labels.

        Returns:
            float: Gini coefficient for the classifier's performance on the validation set.
        """
        y_prob = classifier.predict_proba(X_val_encoded)
        gini = 2 * roc_auc_score(y_val, y_prob[:, 1]) - 1
        return gini

    def run(self):
        train_data = self.load_training_data("train_data_downsampled.parquet")
        X_train, X_val, y_train, y_val = self.split_data(train_data)

        correlation_result = calculate_correlation_with_target(X_train, y_train)
        top_ten_correlations = list(correlation_result[:15].index)

        dict_vectorizer = DictVectorizer(sparse=False)
        X_train_dict = X_train[top_ten_correlations].to_dict(orient="records")
        X_val_dict = X_val[top_ten_correlations].to_dict(orient="records")
        X_train_encoded = dict_vectorizer.fit_transform(X_train_dict)
        X_val_encoded = dict_vectorizer.transform(X_val_dict)

        log_file = "training_log.log"
        log_level = logging.INFO
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        logging.basicConfig(
            filename=log_file,
            level=log_level,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )

        results = []

        for i, (name, classifier, params) in enumerate(self.classifiers, start=1):
            logging.info(f"Step {i}: Training {name} classifier")

            grid = self.train_classifier(
                name, classifier, params, X_train_encoded, y_train
            )
            logging.info(f"Hyperparameter tuning for {name} completed.")

            selected_feature_indices = grid.best_estimator_.named_steps[
                "feature_selection"
            ].get_support(indices=True)
            logging.info(
                f"Selected Indices of Features for {name}: {selected_feature_indices}"
            )

            feature_names = dict_vectorizer.get_feature_names_out()
            selected_features = [feature_names[i] for i in selected_feature_indices]
            logging.info(f"Selected Features for {name}: {selected_features}")

            logging.info(f"Best Estimator for {name}: {grid.best_estimator_}")

            best_model = grid.best_estimator_
            model_filename = f"{name}_{timestamp}.bin"
            joblib.dump(best_model, model_filename)
            logging.info(f"Best model for {name} saved as {model_filename}.")

            gc.collect()

            logging.info(
                f"Step {i + 2}: Evaluating the best model on the validation set using Gini coefficient"
            )

            gini = self.evaluate_classifier(grid, X_val_encoded, y_val)
            logging.info(f"Gini coefficient for {name}: {gini}")

            results.append(
                {"name": name, "gini": gini, "best_params": grid.best_params_}
            )

        json_file_name = f"grid_search_results_{timestamp}.json"
        with open(json_file_name, "w") as file:
            json.dump(results, file, indent=4)
            logging.info(f"Results saved to {json_file_name}.")

        log_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        log_handler = logging.StreamHandler()
        log_handler.setFormatter(log_formatter)
        root_logger = logging.getLogger()
        root_logger.addHandler(log_handler)
        logging.info("Log messages also printed to the console.")


if __name__ == "__main__":
    data_dir = os.path.join("C:\\", "Users", "KonuTech", "zoomcamp-capstone-01", "data")
    classifiers = [
        (
            "LogisticRegression",
            LogisticRegression(n_jobs=-1),
            {
                "classifier__C": [0.1, 1.0],
                "classifier__penalty": ["l1", "l2"],
            },
        ),
        (
            "XGBoost",
            xgb.XGBClassifier(n_jobs=-1),
            {
                "classifier__n_estimators": [50, 100],
                "classifier__max_depth": [3, 5],
                "classifier__min_child_weight": [1, 2],
            },
        ),
    ]

    trainer = ClassifierTrainer(classifiers, data_dir)
    trainer.run()
