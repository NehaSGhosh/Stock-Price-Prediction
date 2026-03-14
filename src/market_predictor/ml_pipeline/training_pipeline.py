import sys

import pandas as pd

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
            _ = refresh_from_api  # Training flow now always resolves Gold as local -> BigQuery -> error.
            metrics_path = self.config_manager.get_train_test_metrics_path()
            gold_df, gold_path, gold_table = GoldWarehouse.resolve_gold_for_training()
            _, features_path, features_table = SentimentScoring.create_gold_with_features_for_training_pipeline(
                gold_df
            )
            feature_df = pd.read_csv(features_path)
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
                "gold_with_features_path": features_path,
                "gold_with_features_table": features_table,
                "model_path": model_output_path,
                "metrics_path": metrics_path,
            }
        except Exception as error:
            raise CustomException(error, sys) from error
