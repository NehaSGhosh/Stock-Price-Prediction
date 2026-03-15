import json
import os
import sys
from datetime import date, datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

if __package__ in {None, ""}:
    CURRENT_FILE_DIR = os.path.dirname(os.path.abspath(__file__))
    SRC_DIR = os.path.abspath(os.path.join(CURRENT_FILE_DIR, "..", ".."))
    PROJECT_ROOT_DIR = os.path.abspath(os.path.join(CURRENT_FILE_DIR, "..", "..", ".."))
    if PROJECT_ROOT_DIR not in sys.path:
        sys.path.insert(0, PROJECT_ROOT_DIR)
    if SRC_DIR not in sys.path:
        sys.path.insert(0, SRC_DIR)

from stock_price_predictor.entity.artifact_entity import DataIngestionArtifact
from stock_price_predictor.entity.config_entity import DataIngestionConfig
from stock_price_predictor.exception import CustomException
from stock_price_predictor.ingestion.market_data_ingestion import MarketDataIngestion
from stock_price_predictor.ingestion.news_ingestion import NewsIngestion
from stock_price_predictor.logger import logging
from stock_price_predictor.utils.common import gcs_blob_exists, read_from_gcs, upload_to_gcs
from stock_price_predictor.warehousing.data_storage import GoldWarehouse


class DataIngestion:
    EASTERN_TZ = ZoneInfo("America/New_York")

    def __init__(self, config: DataIngestionConfig):
        """Initialize ingestion orchestrator with config and storage targets.

        Args:
            config: Effective ingestion configuration.
        """
        self.config = config
        self.market_ingestion = MarketDataIngestion(config)
        self.news_ingestion = NewsIngestion(config)
        from config.configuration import ConfigurationManager

        cfg = ConfigurationManager()
        self.gcs_cfg = cfg.get_gcs_config()
        self.raw_blobs = cfg.get_raw_blob_paths()

    def get_news_data_path(self) -> str:
        """Build the GCS URI for the raw news CSV.

        Returns:
            str: News CSV path in GCS.
        """
        return f"gs://{self.gcs_cfg['bucket_name']}/{self.raw_blobs['news']}"

    def _get_date_window(self) -> tuple[date, date]:
        """Compute [start_date, end_date] from configured lookback.

        Returns:
            tuple[date, date]: Inclusive date window in Eastern timezone.
        """
        if self.config.lookback_days < 1:
            raise ValueError("lookback_days must be >= 1")
        end_date = datetime.now(self.EASTERN_TZ).date()
        start_date = end_date - timedelta(days=self.config.lookback_days - 1)
        return start_date, end_date

    @staticmethod
    def _read_cached_csv(df: pd.DataFrame, label: str) -> pd.DataFrame:
        """Validate cached CSV shape before reuse.

        Args:
            df: Loaded dataframe from GCS cache.
            label: Human-readable dataset name for error messages.

        Returns:
            pd.DataFrame: Validated dataframe.
        """
        if "date" not in df.columns:
            raise ValueError(f"{label} CSV is missing required 'date' column")
        return df

    def _fetch_market_data_window(
        self,
        start_date: date | None = None,
        end_date: date | None = None,
    ) -> pd.DataFrame:
        """Fetch market data for a specific window or default lookback window.

        Args:
            start_date: Optional window start date.
            end_date: Optional window end date.

        Returns:
            pd.DataFrame: Cleaned market dataset.
        """
        if start_date is None or end_date is None:
            start_date, end_date = self._get_date_window()
        return self.market_ingestion.fetch_market_data_window(start_date, end_date)

    def _fetch_news_data(
        self,
        from_date: date | None = None,
        to_date: date | None = None,
    ) -> pd.DataFrame:
        """Fetch news data for a specific window or default lookback window.

        Args:
            from_date: Optional window start date.
            to_date: Optional window end date.

        Returns:
            pd.DataFrame: Cleaned news dataset.
        """
        if from_date is None or to_date is None:
            from_date, to_date = self._get_date_window()
        return self.news_ingestion.fetch_news_data_newsapi(from_date, to_date)

    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        """Run full ingestion and refresh downstream gold dataset.

        Returns:
            DataIngestionArtifact: GCS paths for raw market/news outputs.
        """
        try:
            logging.info(
                "Data ingestion started for %d tickers, lookback_days=%d, interval=%s",
                len(self.config.tickers),
                self.config.lookback_days,
                self.config.interval,
            )
            start_date, end_date = self._get_date_window()

            market_df = self._fetch_market_data_window(start_date=start_date, end_date=end_date)
            news_df = self._fetch_news_data(from_date=start_date, to_date=end_date)

            bucket_name = self.gcs_cfg["bucket_name"]
            market_blob = self.raw_blobs["market"]
            news_blob = self.raw_blobs["news"]
            market_data_path = f"gs://{bucket_name}/{market_blob}"
            news_data_path = self.get_news_data_path()

            upload_to_gcs(df=market_df, bucket_name=bucket_name, destination_blob=market_blob)
            upload_to_gcs(df=news_df, bucket_name=bucket_name, destination_blob=news_blob)
            _, gold_table = GoldWarehouse.refresh_gold_from_raw(self.config.lookback_days)

            logging.info(
                "Market/news/gold refreshed. market_rows=%d, news_rows=%d, gold_table=%s",
                len(market_df),
                len(news_df),
                gold_table,
            )
            return DataIngestionArtifact(
                market_data_path=market_data_path,
                news_data_path=news_data_path,
            )
        except Exception as error:
            raise CustomException(error, sys) from error

    def run_with_cache(self, refresh_from_api: bool = False) -> DataIngestionArtifact:
        """Run cache-aware ingestion with append/rebuild fallback logic.

        Args:
            refresh_from_api: If True, bypass cache and run full ingestion.

        Returns:
            DataIngestionArtifact: GCS paths for raw market/news outputs.
        """
        bucket_name = self.gcs_cfg["bucket_name"]
        market_blob = self.raw_blobs["market"]
        news_blob = self.raw_blobs["news"]
        market_data_path = f"gs://{bucket_name}/{market_blob}"
        news_data_path = self.get_news_data_path()

        if not refresh_from_api and gcs_blob_exists(bucket_name, market_blob) and gcs_blob_exists(bucket_name, news_blob):
            start_date, end_date = self._get_date_window()
            try:
                cached_market_df = self._read_cached_csv(read_from_gcs(bucket_name, market_blob), "Market")
                cached_news_df = self._read_cached_csv(read_from_gcs(bucket_name, news_blob), "News")
            except Exception as error:
                logging.warning("Cache validation failed, running full ingestion. reason=%s", error)
                return self.initiate_data_ingestion()

            market_action, market_append_from = self.market_ingestion.evaluate_market_cache(
                cached_market_df, start_date, end_date
            )
            news_action, news_append_from = self.news_ingestion.evaluate_news_cache(
                cached_news_df, start_date, end_date
            )
            logging.info(
                "Cache status window start=%s end=%s | market=%s news=%s",
                start_date,
                end_date,
                market_action,
                news_action,
            )

            if market_action == "complete" and news_action == "complete":
                logging.info(
                    "Using existing CSV files, full %d-day window already present.",
                    self.config.lookback_days,
                )
                return DataIngestionArtifact(
                    market_data_path=market_data_path,
                    news_data_path=news_data_path,
                )

            if market_action == "reload" or news_action == "reload":
                logging.info(
                    "Old-window coverage missing. Rebuilding full %d-day dataset.",
                    self.config.lookback_days,
                )
                return self.initiate_data_ingestion()

            append_candidates = [
                append_from
                for append_from in [market_append_from, news_append_from]
                if append_from is not None
            ]
            append_from = min(append_candidates) if append_candidates else None

            if append_from is None or append_from > end_date:
                logging.info("No append needed after cache evaluation.")
                return DataIngestionArtifact(
                    market_data_path=market_data_path,
                    news_data_path=news_data_path,
                )

            logging.info(
                "Recent days missing. Appending data window start=%s end=%s.",
                append_from,
                end_date,
            )
            new_market_df = self._fetch_market_data_window(start_date=append_from, end_date=end_date)
            new_news_df = self._fetch_news_data(from_date=append_from, to_date=end_date)

            merged_market_df = pd.concat([cached_market_df, new_market_df], ignore_index=True)
            merged_market_df = self.market_ingestion.clean_market_data(merged_market_df)
            merged_market_df = merged_market_df[
                (merged_market_df["date"] >= start_date) & (merged_market_df["date"] <= end_date)
            ].reset_index(drop=True)

            merged_news_df = pd.concat([cached_news_df, new_news_df], ignore_index=True)
            merged_news_df = self.news_ingestion.clean_news_data(merged_news_df)
            merged_news_df = merged_news_df[
                (merged_news_df["date"] >= start_date) & (merged_news_df["date"] <= end_date)
            ].reset_index(drop=True)

            upload_to_gcs(df=merged_market_df, bucket_name=bucket_name, destination_blob=market_blob)
            upload_to_gcs(df=merged_news_df, bucket_name=bucket_name, destination_blob=news_blob)
            logging.info(
                "Append complete. market_rows=%d news_rows=%d",
                len(merged_market_df),
                len(merged_news_df),
            )
            return DataIngestionArtifact(
                market_data_path=market_data_path,
                news_data_path=news_data_path,
            )

        return self.initiate_data_ingestion()

    def append_last_n_days(self, days: int) -> DataIngestionArtifact:
        """Append latest raw rows to existing GCS datasets and rebuild gold.

        Args:
            days: Number of latest days to append.

        Returns:
            DataIngestionArtifact: GCS paths for updated market/news outputs.
        """
        if days < 1:
            raise ValueError("append must be >= 1")

        bucket_name = self.gcs_cfg["bucket_name"]
        market_blob = self.raw_blobs["market"]
        news_blob = self.raw_blobs["news"]
        market_data_path = f"gs://{bucket_name}/{market_blob}"
        news_data_path = self.get_news_data_path()

        if not gcs_blob_exists(bucket_name, market_blob) or not gcs_blob_exists(bucket_name, news_blob):
            raise FileNotFoundError(
                "Append mode requires existing raw CSV files in GCS. "
                f"Missing market/news blob under bucket: {bucket_name}"
            )

        cached_market_df = self._read_cached_csv(read_from_gcs(bucket_name, market_blob), "Market")
        cached_news_df = self._read_cached_csv(read_from_gcs(bucket_name, news_blob), "News")

        end_date = datetime.now(self.EASTERN_TZ).date()
        start_date = end_date - timedelta(days=days - 1)
        logging.info("Appending last %d day(s). window_start=%s window_end=%s", days, start_date, end_date)

        new_market_df = self._fetch_market_data_window(start_date=start_date, end_date=end_date)
        new_news_df = self._fetch_news_data(from_date=start_date, to_date=end_date)

        merged_market_df = pd.concat([cached_market_df, new_market_df], ignore_index=True)
        merged_market_df = self.market_ingestion.clean_market_data(merged_market_df)

        merged_news_df = pd.concat([cached_news_df, new_news_df], ignore_index=True)
        merged_news_df = self.news_ingestion.clean_news_data(merged_news_df)

        upload_to_gcs(df=merged_market_df, bucket_name=bucket_name, destination_blob=market_blob)
        upload_to_gcs(df=merged_news_df, bucket_name=bucket_name, destination_blob=news_blob)
        _, gold_table = GoldWarehouse.append_gold_from_raw(days)
        logging.info(
            "Append complete. market_rows=%d news_rows=%d gold_table=%s",
            len(merged_market_df),
            len(merged_news_df),
            gold_table,
        )
        return DataIngestionArtifact(
            market_data_path=market_data_path,
            news_data_path=news_data_path,
        )


def build_effective_ingestion_config(
    tickers: list[str] | None = None,
    lookback_days: int | None = None,
) -> DataIngestionConfig:
    """Build runtime ingestion config by overlaying request overrides.

    Args:
        tickers: Optional ticker override list.
        lookback_days: Optional lookback override.

    Returns:
        DataIngestionConfig: Effective ingestion configuration used for execution.
    """
    from config.configuration import ConfigurationManager

    cfg = ConfigurationManager()
    ingestion_cfg = cfg.get_data_ingestion_config()
    return DataIngestionConfig(
        tickers=tickers if tickers else ingestion_cfg.tickers,
        lookback_days=lookback_days if lookback_days is not None else ingestion_cfg.lookback_days,
        interval=ingestion_cfg.interval,
        bigquery_project_id=ingestion_cfg.bigquery_project_id,
        raw_data_dir=ingestion_cfg.raw_data_dir,
        market_data_file=ingestion_cfg.market_data_file,
        news_data_file=ingestion_cfg.news_data_file,
    )


def ingest_data(request: Any):
    """Cloud-function-style ingestion entrypoint.

    Args:
        request: Request-like object exposing `args` and `get_json`.

    Returns:
        tuple[str, int, dict[str, str]]: JSON body string, status code, and headers.
    """
    try:
        payload = request.get_json(silent=True) if request else {}
        if payload is None:
            payload = {}
        args = request.args if request and getattr(request, "args", None) else {}

        tickers_raw = payload.get("tickers", args.get("tickers"))
        tickers = None
        if isinstance(tickers_raw, str):
            tickers = [ticker.strip() for ticker in tickers_raw.split(",") if ticker.strip()]
        elif isinstance(tickers_raw, list):
            tickers = [str(ticker).strip() for ticker in tickers_raw if str(ticker).strip()]

        lookback_days_raw = payload.get("lookback_days", args.get("lookback_days"))
        lookback_days = int(lookback_days_raw) if lookback_days_raw is not None else None

        has_append = "append" in payload or ("append" in args)
        append_raw = payload.get("append", args.get("append")) if has_append else None
        append_days = 1 if (has_append and (append_raw is None or str(append_raw).strip() == "")) else (
            int(append_raw) if has_append else None
        )

        if lookback_days is not None and has_append:
            raise ValueError("lookback_days and append cannot be used together.")

        effective_cfg = build_effective_ingestion_config(
            tickers=tickers,
            lookback_days=lookback_days,
        )
        ingestion = DataIngestion(effective_cfg)
        if has_append:
            artifact = ingestion.append_last_n_days(days=append_days)
        elif lookback_days is not None:
            artifact = ingestion.initiate_data_ingestion()
        else:
            artifact = ingestion.run_with_cache(refresh_from_api=False)

        response = {
            "status": "success",
            "market_data_path": artifact.market_data_path,
            "news_data_path": artifact.news_data_path,
            "tickers": effective_cfg.tickers,
        }
        if has_append:
            response["append"] = append_days
        if not has_append:
            response["lookback_days"] = effective_cfg.lookback_days
        return (json.dumps(response), 200, {"Content-Type": "application/json"})
    except Exception as error:
        logging.exception("Cloud ingestion failed: %s", error)
        response = {"status": "error", "message": str(error)}
        return (json.dumps(response), 500, {"Content-Type": "application/json"})


def gcp_data_ingestion(request: Any):
    # Backward-compatible Cloud Function alias.
    """Backward-compatible alias for ingestion cloud entrypoint.

    Args:
        request: Request-like object exposing `args` and `get_json`.

    Returns:
        tuple[str, int, dict[str, str]]: JSON body string, status code, and headers.
    """
    return ingest_data(request)
