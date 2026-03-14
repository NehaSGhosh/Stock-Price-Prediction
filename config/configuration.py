from market_predictor.entity.config_entity import (
    DataIngestionConfig,
    ModelTrainerConfig,
)
from market_predictor.utils.common import read_yaml


class ConfigurationManager:
    def __init__(self, config_filepath: str = "config/config.yaml"):
        self.config = read_yaml(config_filepath)

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        cfg = self.config["data_ingestion"]
        project_cfg = self.config["project"]
        payload = {
            "tickers": cfg["tickers"],
            "lookback_days": int(cfg["lookback_days"]),
            "interval": cfg["interval"],
            "bigquery_project_id": project_cfg["bigquery_project_id"],
            "raw_data_dir": cfg["raw_data_dir"],
            "market_data_file": cfg["market_data_file"],
            "news_data_file": cfg["news_data_file"],
        }
        return DataIngestionConfig(**payload)

    def get_model_trainer_config(self) -> ModelTrainerConfig:
        cfg = self.config["model_trainer"]
        random_state = self.config["project"]["random_state"]
        max_depth_raw = cfg.get("max_depth", 8)
        max_depth = None if max_depth_raw in {None, "none", "None"} else int(max_depth_raw)
        return ModelTrainerConfig(
            model_dir=cfg["model_dir"],
            model_file=cfg["model_file"],
            train_ratio=float(cfg["train_ratio"]),
            random_state=random_state,
            n_estimators=int(cfg.get("n_estimators", 300)),
            max_depth=max_depth,
            min_samples_split=int(cfg.get("min_samples_split", 10)),
            min_samples_leaf=int(cfg.get("min_samples_leaf", 4)),
            max_features=cfg.get("max_features", "sqrt"),
            bootstrap=bool(cfg.get("bootstrap", True)),
        )

    def get_classification_target_column(self) -> str:
        return self.config["model_trainer"]["classification_target_column"]

    def get_target_column(self) -> str:
        return self.config["data_ingestion"]["target_column"]

    def get_gold_path(self) -> str:
        return self.config["data_processing"]["gold_path"]

    def get_gold_with_features_path(self) -> str:
        return self.config["data_processing"]["gold_with_features_path"]

    def get_train_test_metrics_path(self) -> str:
        return self.config["model_trainer"]["train_test_metrics_path"]

    def get_gold_warehouse_config(self) -> dict[str, str]:
        project_cfg = self.config["project"]
        processing_cfg = self.config["data_processing"]
        return {
            "project_id": project_cfg["bigquery_project_id"],
            "dataset_id": project_cfg["bigquery_dataset_id"],
            "table_id": processing_cfg["bigquery_gold_table_id"],
            "gold_with_features_table_id": processing_cfg["bigquery_gold_with_features_table_id"],
        }
