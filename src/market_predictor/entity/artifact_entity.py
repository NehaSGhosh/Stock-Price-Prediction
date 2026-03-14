from dataclasses import dataclass


@dataclass(frozen=True)
class DataIngestionArtifact:
    market_data_path: str
    news_data_path: str
