import json

from stock_price_predictor.entity.artifact_entity import DataIngestionArtifact
from stock_price_predictor.entity.config_entity import DataIngestionConfig
from stock_price_predictor.ingestion.ingestion_pipeline import ingest_data


class _FakeRequest:
    def __init__(self, args=None, payload=None):
        self.args = args or {}
        self._payload = payload or {}

    def get_json(self, silent=True):
        _ = silent
        return self._payload


def _build_cfg(lookback_days=29):
    return DataIngestionConfig(
        tickers=["MSFT", "AAPL", "GOOG", "AVGO", "UBER"],
        lookback_days=lookback_days,
        interval="1d",
        bigquery_project_id="dummy-project",
        raw_data_dir="raw",
        market_data_file="market_data.csv",
        news_data_file="news_data.csv",
    )


def test_ingest_response_includes_only_append_key_for_append_mode(monkeypatch):
    cfg = _build_cfg(lookback_days=29)

    class _StubIngestion:
        def __init__(self, _cfg):
            self._cfg = _cfg

        def append_last_n_days(self, days):
            assert days == 2
            return DataIngestionArtifact(market_data_path="gs://bucket/m.csv", news_data_path="gs://bucket/n.csv")

        def initiate_data_ingestion(self):
            raise AssertionError("lookback flow should not run for append test")

        def run_with_cache(self, refresh_from_api=False):
            _ = refresh_from_api
            raise AssertionError("cache flow should not run for append test")

    monkeypatch.setattr(
        "stock_price_predictor.ingestion.ingestion_pipeline.build_effective_ingestion_config",
        lambda tickers=None, lookback_days=None: cfg,
    )
    monkeypatch.setattr("stock_price_predictor.ingestion.ingestion_pipeline.DataIngestion", _StubIngestion)

    body, status_code, _ = ingest_data(_FakeRequest(args={"append": "2"}))
    parsed = json.loads(body)

    assert status_code == 200
    assert parsed["append"] == 2
    assert "lookback_days" not in parsed


def test_ingest_response_includes_only_lookback_key_for_lookback_mode(monkeypatch):
    cfg = _build_cfg(lookback_days=29)

    class _StubIngestion:
        def __init__(self, _cfg):
            self._cfg = _cfg

        def append_last_n_days(self, days):
            _ = days
            raise AssertionError("append flow should not run for lookback test")

        def initiate_data_ingestion(self):
            return DataIngestionArtifact(market_data_path="gs://bucket/m.csv", news_data_path="gs://bucket/n.csv")

        def run_with_cache(self, refresh_from_api=False):
            _ = refresh_from_api
            raise AssertionError("cache flow should not run for lookback test")

    monkeypatch.setattr(
        "stock_price_predictor.ingestion.ingestion_pipeline.build_effective_ingestion_config",
        lambda tickers=None, lookback_days=None: cfg,
    )
    monkeypatch.setattr("stock_price_predictor.ingestion.ingestion_pipeline.DataIngestion", _StubIngestion)

    body, status_code, _ = ingest_data(_FakeRequest(args={"lookback_days": "29"}))
    parsed = json.loads(body)

    assert status_code == 200
    assert parsed["lookback_days"] == 29
    assert "append" not in parsed
