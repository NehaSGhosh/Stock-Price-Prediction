from config.configuration import ConfigurationManager
import main
from stock_price_predictor.entity.config_entity import DataIngestionConfig


def _build_cfg():
    return DataIngestionConfig(
        tickers=["MSFT", "AAPL", "GOOG", "AVGO", "UBER"],
        lookback_days=29,
        interval="1d",
        bigquery_project_id="dummy-project",
        raw_data_dir="raw",
        market_data_file="market_data.csv",
        news_data_file="news_data.csv",
    )


def test_get_predict_tickers_returns_configured_tickers(monkeypatch):
    monkeypatch.setattr(main._config_manager, "get_data_ingestion_config", lambda: _build_cfg())
    assert main.get_predict_tickers() == ["MSFT", "AAPL", "GOOG", "AVGO", "UBER"]


def test_gold_warehouse_config_uses_current_keys_only():
    cfg = ConfigurationManager().get_gold_warehouse_config()
    assert set(cfg.keys()) == {"project_id", "dataset_id", "table_id"}
