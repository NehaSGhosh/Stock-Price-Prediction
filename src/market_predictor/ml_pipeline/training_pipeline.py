import sys
import json
from typing import Any

from config.configuration import ConfigurationManager
from market_predictor.exception import CustomException
from market_predictor.logger import logging
from market_predictor.ml_pipeline.feature_engineering import SentimentScoring
from market_predictor.ml_pipeline.model_trainer import ModelTrainer
from market_predictor.warehousing.data_storage import GoldWarehouse


class TrainingPipeline:
    def __init__(self):
        self.config_manager = ConfigurationManager()
        self.model_cfg = self.config_manager.get_model_trainer_config()
        self.classification_target_column = self.config_manager.get_classification_target_column()

    def run_model_training_pipeline(
        self,
        refresh_from_api: bool,
        model_output_path: str,
    ) -> dict[str, str]:
        try:
            _ = refresh_from_api  # Training flow resolves Gold directly from BigQuery.
            metrics_path = self.config_manager.get_train_test_metrics_path()
            gold_df, gold_path, gold_table = GoldWarehouse.resolve_gold_for_training()
            feature_df, features_table = SentimentScoring.create_gold_with_features_for_training_pipeline(
                gold_df
            )
            trainer = ModelTrainer(
                target_column=self.classification_target_column,
                random_state=self.model_cfg.random_state,
                train_ratio=self.model_cfg.train_ratio,
                n_estimators=self.model_cfg.n_estimators,
                max_depth=self.model_cfg.max_depth,
                min_samples_split=self.model_cfg.min_samples_split,
                min_samples_leaf=self.model_cfg.min_samples_leaf,
                max_features=self.model_cfg.max_features,
                bootstrap=self.model_cfg.bootstrap,
            )
            metrics = trainer.train_classifier_from_dataframe(
                feature_df=feature_df,
                model_output_path=model_output_path,
                metrics_output_path=metrics_path,
            )
            logging.info("Training metrics (train/test): %s", metrics)
            return {
                "gold_path": gold_path,
                "gold_table": gold_table,
                "gold_with_features_table": features_table,
                "model_path": model_output_path,
                "metrics_path": metrics_path,
            }
        except Exception as error:
            raise CustomException(error, sys) from error


def train_model(request: Any):
    try:
        args = request.args if request and getattr(request, "args", None) else {}
        payload = request.get_json(silent=True) if request else {}
        if payload is None:
            payload = {}

        model_output_path = payload.get("model_output_path", args.get("model_output_path"))
        refresh_raw = payload.get("refresh_from_api", args.get("refresh_from_api", "false"))
        refresh_from_api = str(refresh_raw).strip().lower() in {"1", "true", "yes", "y"}

        from market_predictor.ml_pipeline.model_trainer import DEFAULT_MODEL_PATH

        pipeline = TrainingPipeline()
        result = pipeline.run_model_training_pipeline(
            refresh_from_api=refresh_from_api,
            model_output_path=model_output_path or DEFAULT_MODEL_PATH,
        )
        response = {
            "status": "success",
            "result": result,
        }
        return (json.dumps(response), 200, {"Content-Type": "application/json"})
    except Exception as error:
        logging.exception("Cloud training failed: %s", error)
        response = {"status": "error", "message": str(error)}
        return (json.dumps(response), 500, {"Content-Type": "application/json"})


def gcp_train_model(request: Any):
    # Backward-compatible Cloud Function alias.
    return train_model(request)
