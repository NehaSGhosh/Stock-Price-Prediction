from stock_price_predictor.entity.config_entity import (
    DataIngestionConfig,
    ModelTrainerConfig,
)
from stock_price_predictor.utils.common import read_yaml


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
        gcs_cfg = self.get_gcs_config()
        filename = "gold.csv"
        return f"gs://{gcs_cfg['bucket_name']}/{gcs_cfg['processed_prefix'].strip('/')}/{filename}"

    def get_gold_with_features_path(self) -> str:
        gcs_cfg = self.get_gcs_config()
        data_prefix = self.config["project"].get("gcs_data_dir", "data").strip("/")
        filename = self.config["data_processing"]["gold_with_features_file"]
        return f"gs://{gcs_cfg['bucket_name']}/{data_prefix}/{filename}"

    def get_metrics_path(self) -> str:
        project_cfg = self.config["project"]
        trainer_cfg = self.config["model_trainer"]
        bucket_name = project_cfg["gcs_bucket_name"]
        models_dir = project_cfg.get("gcs_models_dir", "models").strip("/")
        metrics_file = trainer_cfg["metrics_file"]
        return f"gs://{bucket_name}/{models_dir}/{metrics_file}"

    def get_model_artifact_path(self) -> str:
        gcs_cfg = self.get_gcs_config()
        model_file = self.config["model_trainer"]["model_file"]
        return f"gs://{gcs_cfg['bucket_name']}/{gcs_cfg['models_prefix'].strip('/')}/{model_file}"

    def get_raw_blob_paths(self) -> dict[str, str]:
        ingestion_cfg = self.config["data_ingestion"]
        gcs_cfg = self.get_gcs_config()
        raw_prefix = gcs_cfg["raw_prefix"].strip("/")
        return {
            "market": f"{raw_prefix}/{ingestion_cfg['market_data_file']}",
            "news": f"{raw_prefix}/{ingestion_cfg['news_data_file']}",
        }

    def get_processed_blob_paths(self) -> dict[str, str]:
        processing_cfg = self.config["data_processing"]
        gcs_cfg = self.get_gcs_config()
        data_prefix = self.config["project"].get("gcs_data_dir", "data").strip("/")
        return {
            "gold": f"{gcs_cfg['processed_prefix'].strip('/')}/gold.csv",
            "gold_with_features": f"{data_prefix}/{processing_cfg['gold_with_features_file']}",
        }

    def get_gcs_config(self) -> dict[str, str]:
        project_cfg = self.config["project"]
        data_prefix = project_cfg.get("gcs_data_dir", "data").strip("/")
        return {
            "bucket_name": project_cfg["gcs_bucket_name"],
            "raw_prefix": project_cfg.get("gcs_raw_prefix", data_prefix),
            "processed_prefix": project_cfg.get("gcs_processed_prefix", data_prefix),
            "models_prefix": project_cfg.get("gcs_models_prefix", project_cfg.get("gcs_models_dir", "models")),
            "logs_prefix": project_cfg.get("gcs_logs_prefix", project_cfg.get("gcs_logs_dir", "logs")),
        }

    def get_gold_warehouse_config(self) -> dict[str, str]:
        project_cfg = self.config["project"]
        processing_cfg = self.config["data_processing"]
        return {
            "project_id": project_cfg["bigquery_project_id"],
            "dataset_id": project_cfg["bigquery_dataset_id"],
            "table_id": processing_cfg["bigquery_gold_table_id"],
        }
