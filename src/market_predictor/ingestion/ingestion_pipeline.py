import argparse
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

from market_predictor.entity.artifact_entity import DataIngestionArtifact
from market_predictor.entity.config_entity import DataIngestionConfig
from market_predictor.exception import CustomException
from market_predictor.ingestion.market_data_ingestion import MarketDataIngestion
from market_predictor.ingestion.news_ingestion import NewsIngestion
from market_predictor.logger import logging
from market_predictor.utils.common import ensure_dir
from market_predictor.warehousing.data_storage import GoldWarehouse


class DataIngestion:
    EASTERN_TZ = ZoneInfo("America/New_York")

    def __init__(self, config: DataIngestionConfig):
        self.config = config
        self.market_ingestion = MarketDataIngestion(config)
        self.news_ingestion = NewsIngestion(config)

    def get_news_data_path(self) -> str:
        return os.path.join(self.config.raw_data_dir, self.config.news_data_file)

    def _get_date_window(self) -> tuple[date, date]:
        if self.config.lookback_days < 1:
            raise ValueError("lookback_days must be >= 1")
        end_date = datetime.now(self.EASTERN_TZ).date()
        start_date = end_date - timedelta(days=self.config.lookback_days - 1)
        return start_date, end_date

    @staticmethod
    def _read_cached_csv(path: str, label: str) -> pd.DataFrame:
        if not os.path.exists(path):
            raise FileNotFoundError(f"{label} CSV not found: {path}")
        df = pd.read_csv(path)
        if "date" not in df.columns:
            raise ValueError(f"{label} CSV is missing required 'date' column: {path}")
        return df

    def _fetch_market_data_window(
        self,
        start_date: date | None = None,
        end_date: date | None = None,
    ) -> pd.DataFrame:
        if start_date is None or end_date is None:
            start_date, end_date = self._get_date_window()
        return self.market_ingestion.fetch_market_data_window(start_date, end_date)

    def _fetch_news_data(
        self,
        from_date: date | None = None,
        to_date: date | None = None,
    ) -> pd.DataFrame:
        if from_date is None or to_date is None:
            from_date, to_date = self._get_date_window()
        return self.news_ingestion.fetch_news_data_newsapi(from_date, to_date)

    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        try:
            logging.info(
                "Data ingestion started for %d tickers, lookback_days=%d, interval=%s",
                len(self.config.tickers),
                self.config.lookback_days,
                self.config.interval,
            )
            ensure_dir(self.config.raw_data_dir)
            start_date, end_date = self._get_date_window()

            market_df = self._fetch_market_data_window(start_date=start_date, end_date=end_date)
            news_df = self._fetch_news_data(from_date=start_date, to_date=end_date)

            market_data_path = os.path.join(self.config.raw_data_dir, self.config.market_data_file)
            news_data_path = self.get_news_data_path()

            market_df.to_csv(market_data_path, index=False)
            news_df.to_csv(news_data_path, index=False)
            _, gold_path, gold_table = GoldWarehouse.refresh_gold_from_raw(self.config.lookback_days)

            logging.info(
                "Market/news/gold refreshed. market_rows=%d, news_rows=%d, gold_path=%s, gold_table=%s",
                len(market_df),
                len(news_df),
                gold_path,
                gold_table,
            )
            return DataIngestionArtifact(
                market_data_path=market_data_path,
                news_data_path=news_data_path,
            )
        except Exception as error:
            raise CustomException(error, sys) from error

    def run_with_cache(self, refresh_from_api: bool = False) -> DataIngestionArtifact:
        market_data_path = os.path.join(self.config.raw_data_dir, self.config.market_data_file)
        news_data_path = self.get_news_data_path()
        ensure_dir(self.config.raw_data_dir)

        if not refresh_from_api and os.path.exists(market_data_path) and os.path.exists(news_data_path):
            start_date, end_date = self._get_date_window()
            try:
                cached_market_df = self._read_cached_csv(market_data_path, "Market")
                cached_news_df = self._read_cached_csv(news_data_path, "News")
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

            merged_market_df.to_csv(market_data_path, index=False)
            merged_news_df.to_csv(news_data_path, index=False)
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
        if days < 1:
            raise ValueError("append must be >= 1")

        market_data_path = os.path.join(self.config.raw_data_dir, self.config.market_data_file)
        news_data_path = self.get_news_data_path()
        ensure_dir(self.config.raw_data_dir)

        if not os.path.exists(market_data_path) or not os.path.exists(news_data_path):
            raise FileNotFoundError(
                "Append mode requires existing raw CSV files. "
                f"Missing market/news CSV under: {self.config.raw_data_dir}"
            )

        cached_market_df = self._read_cached_csv(market_data_path, "Market")
        cached_news_df = self._read_cached_csv(news_data_path, "News")

        end_date = datetime.now(self.EASTERN_TZ).date()
        start_date = end_date - timedelta(days=days - 1)
        logging.info("Appending last %d day(s). window_start=%s window_end=%s", days, start_date, end_date)

        new_market_df = self._fetch_market_data_window(start_date=start_date, end_date=end_date)
        new_news_df = self._fetch_news_data(from_date=start_date, to_date=end_date)

        merged_market_df = pd.concat([cached_market_df, new_market_df], ignore_index=True)
        merged_market_df = self.market_ingestion.clean_market_data(merged_market_df)

        merged_news_df = pd.concat([cached_news_df, new_news_df], ignore_index=True)
        merged_news_df = self.news_ingestion.clean_news_data(merged_news_df)

        merged_market_df.to_csv(market_data_path, index=False)
        merged_news_df.to_csv(news_data_path, index=False)
        _, gold_path, gold_table = GoldWarehouse.append_gold_from_raw(days)
        logging.info(
            "Append complete. market_rows=%d news_rows=%d gold_path=%s gold_table=%s",
            len(merged_market_df),
            len(merged_news_df),
            gold_path,
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


def _run_as_script() -> None:
    parser = argparse.ArgumentParser(description="Run data ingestion as a standalone script.")
    parser.add_argument("--tickers", nargs="+", help="Tickers list, e.g. --tickers MSFT AAPL GOOG AVGO UBER")
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--lookback_days",
        type=int,
        default=None,
        help="Force full API fetch for this many days.",
    )
    mode_group.add_argument(
        "--append",
        type=int,
        nargs="?",
        const=1,
        default=None,
        help="Append last n days to existing CSV files (default when omitted: 1).",
    )
    args = parser.parse_args()

    effective_cfg = build_effective_ingestion_config(
        tickers=args.tickers,
        lookback_days=args.lookback_days,
    )
    ingestion = DataIngestion(effective_cfg)
    if args.append is not None:
        artifact = ingestion.append_last_n_days(days=args.append)
    else:
        artifact = ingestion.initiate_data_ingestion()
    logging.info(
        "Standalone ingestion completed. market=%s news=%s",
        artifact.market_data_path,
        artifact.news_data_path,
    )


def gcp_data_ingestion(request: Any):
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
            "lookback_days": effective_cfg.lookback_days,
            "append": append_days if has_append else None,
            "tickers": effective_cfg.tickers,
        }
        return (json.dumps(response), 200, {"Content-Type": "application/json"})
    except Exception as error:
        logging.exception("Cloud ingestion failed: %s", error)
        response = {"status": "error", "message": str(error)}
        return (json.dumps(response), 500, {"Content-Type": "application/json"})


if __name__ == "__main__":
    _run_as_script()
