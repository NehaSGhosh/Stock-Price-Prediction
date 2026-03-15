import os
import json
from typing import Any

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

from config.configuration import ConfigurationManager
from stock_price_predictor.logger import logging
from stock_price_predictor.utils.common import (
    ensure_dir,
    parse_gcs_uri,
    save_object,
    upload_bytes_to_gcs,
)

_config_manager = ConfigurationManager()
_model_cfg = _config_manager.get_model_trainer_config()

DEFAULT_MODEL_PATH = _config_manager.get_model_artifact_path()
DEFAULT_METRICS_PATH = _config_manager.get_metrics_path()


class ModelTrainer:
    def __init__(
        self,
        target_column: str,
        random_state: int,
        train_ratio: float,
        n_estimators: int,
        max_depth: int | None,
        min_samples_split: int,
        min_samples_leaf: int,
        max_features: str | int | float | None,
        bootstrap: bool,
    ):
        self.target_column = target_column
        self.random_state = random_state
        self.train_ratio = train_ratio
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.bootstrap = bootstrap

    @staticmethod
    def _validate_min_rows(row_count: int) -> None:
        if row_count < 2:
            raise ValueError("At least 2 rows are required to split into train/test sets.")

    def _prepare_features(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, list[str]]:
        if self.target_column not in df.columns:
            raise ValueError(f"Missing target column: {self.target_column}")

        working_df = df.copy()
        if "ticker" in working_df.columns:
            working_df = pd.get_dummies(working_df, columns=["ticker"], prefix="ticker", dtype=int)

        excluded_cols = {
            "date",
            "headline",
            "headlines_list",
            "next_close",
            "next_day_price_change_pct",
            self.target_column,
        }
        feature_cols = [col for col in working_df.columns if col not in excluded_cols]
        x = working_df[feature_cols].copy()
        x = x.replace([float("inf"), float("-inf")], pd.NA)
        for col in x.columns:
            if pd.api.types.is_numeric_dtype(x[col]):
                x[col] = x[col].fillna(x[col].median())
            else:
                x[col] = x[col].fillna(0)
        y = working_df[self.target_column].astype(int)
        return x, y, feature_cols

    @staticmethod
    def _build_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        }

    @staticmethod
    def _log_metrics(prefix: str, metrics: dict[str, float]) -> None:
        logging.info("%s accuracy: %.4f", prefix, metrics["accuracy"])
        logging.info("%s precision: %.4f", prefix, metrics["precision"])
        logging.info("%s recall: %.4f", prefix, metrics["recall"])
        logging.info("%s f1: %.4f", prefix, metrics["f1"])

    def split_train_test_dataframe(self, feature_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        working_df = feature_df.reset_index(drop=True)
        self._validate_min_rows(len(working_df))
        stratify_target = None
        if self.target_column in working_df.columns:
            y = working_df[self.target_column]
            if y.nunique(dropna=True) > 1:
                stratify_target = y
        train_df, test_df = train_test_split(
            working_df,
            train_size=self.train_ratio,
            random_state=self.random_state,
            shuffle=True,
            stratify=stratify_target,
        )
        return train_df.reset_index(drop=True), test_df.reset_index(drop=True)

    def train_classifier_from_dataframe(
        self,
        feature_df: pd.DataFrame,
        model_output_path: str,
        metrics_output_path: str = DEFAULT_METRICS_PATH,
    ) -> dict[str, Any]:
        train_df, test_df = self.split_train_test_dataframe(feature_df)
        return self.train_classifier_from_splits(
            train_df=train_df,
            test_df=test_df,
            model_output_path=model_output_path,
            metrics_output_path=metrics_output_path,
        )

    def train_classifier_from_splits(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        model_output_path: str,
        metrics_output_path: str = DEFAULT_METRICS_PATH,
    ) -> dict[str, Any]:
        if train_df.empty or test_df.empty:
            raise ValueError("Both train and test datasets must be non-empty.")
        x_train, y_train, feature_cols = self._prepare_features(train_df.reset_index(drop=True))
        x_test, y_test, _ = self._prepare_features(test_df.reset_index(drop=True))

        model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            bootstrap=self.bootstrap,
            random_state=self.random_state,
            n_jobs=-1,
            class_weight="balanced_subsample",
        )
        model.fit(x_train, y_train)
        train_predictions = model.predict(x_train)
        train_metrics = self._build_metrics(y_true=y_train, y_pred=train_predictions)
        predictions = model.predict(x_test)
        test_metrics = self._build_metrics(y_true=y_test, y_pred=predictions)
        metrics = {
            "train_metrics": train_metrics,
            "test_metrics": test_metrics,
            "train_rows": int(len(x_train)),
            "test_rows": int(len(x_test)),
            "feature_count": int(len(feature_cols)),
            "train_ratio": float(self.train_ratio),
        }

        model_payload = {"model": model, "feature_columns": feature_cols}
        save_object(model_output_path, model_payload)
        self._log_metrics("Train", train_metrics)
        self._log_metrics("Test", test_metrics)
        if metrics_output_path.startswith("gs://"):
            bucket_name, blob_name = parse_gcs_uri(metrics_output_path)
            upload_bytes_to_gcs(
                data=json.dumps(metrics, indent=2).encode("utf-8"),
                bucket_name=bucket_name,
                destination_blob=blob_name,
                content_type="application/json",
            )
        else:
            ensure_dir(os.path.dirname(metrics_output_path))
            with open(metrics_output_path, "w", encoding="utf-8") as metrics_file:
                json.dump(metrics, metrics_file, indent=2)
        logging.info("Train/test metrics saved at: %s", metrics_output_path)
        return metrics
