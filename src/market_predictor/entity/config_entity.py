from dataclasses import dataclass


@dataclass(frozen=True)
class DataIngestionConfig:
    tickers: list[str]
    lookback_days: int
    interval: str
    bigquery_project_id: str
    raw_data_dir: str
    market_data_file: str
    news_data_file: str

@dataclass(frozen=True)
class ModelTrainerConfig:
    model_dir: str
    model_file: str
    train_ratio: float
    random_state: int
    n_estimators: int
    max_depth: int | None
    min_samples_split: int
    min_samples_leaf: int
    max_features: str | int | float | None
    bootstrap: bool
