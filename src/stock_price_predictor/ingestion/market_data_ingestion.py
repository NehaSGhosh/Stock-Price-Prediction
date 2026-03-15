from datetime import date, timedelta

import pandas as pd
import yfinance as yf

from stock_price_predictor.entity.config_entity import DataIngestionConfig
from stock_price_predictor.logger import logging


class MarketDataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    @staticmethod
    def normalize_market_frame(data: pd.DataFrame, ticker: str) -> pd.DataFrame:
        frame = data.copy()
        if isinstance(frame.columns, pd.MultiIndex):
            frame.columns = frame.columns.get_level_values(0)

        frame = frame.reset_index()
        frame.columns = [str(col).strip().lower().replace(" ", "_") for col in frame.columns]

        if "date" not in frame.columns:
            if "datetime" in frame.columns:
                frame = frame.rename(columns={"datetime": "date"})
            else:
                first_col = frame.columns[0] if len(frame.columns) > 0 else None
                if first_col:
                    frame = frame.rename(columns={first_col: "date"})

        if "date" not in frame.columns:
            raise ValueError(f"Unable to identify date column for ticker {ticker}.")

        frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.date
        frame = frame.dropna(subset=["date"])
        required = ["open", "high", "low", "close", "adj_close", "volume"]
        available = [col for col in required if col in frame.columns]
        if not available:
            raise ValueError(f"No OHLCV columns found for ticker {ticker}.")
        frame = frame[["date"] + available]
        frame["ticker"] = ticker
        return frame

    @staticmethod
    def clean_market_data(market_df: pd.DataFrame) -> pd.DataFrame:
        cleaned_df = market_df.copy()
        numeric_cols = [col for col in ["open", "high", "low", "close", "adj_close", "volume"] if col in cleaned_df.columns]
        price_cols = [col for col in ["open", "high", "low", "close", "adj_close"] if col in cleaned_df.columns]

        missing_before = cleaned_df[numeric_cols].isna().sum().to_dict() if numeric_cols else {}
        rows_before = len(cleaned_df)

        cleaned_df["date"] = pd.to_datetime(cleaned_df["date"], errors="coerce").dt.date
        cleaned_df = cleaned_df.dropna(subset=["date", "ticker"])

        for col in numeric_cols:
            cleaned_df[col] = pd.to_numeric(cleaned_df[col], errors="coerce")

        cleaned_df = cleaned_df.sort_values(["ticker", "date"]).reset_index(drop=True)
        cleaned_df = cleaned_df.drop_duplicates(subset=["ticker", "date"], keep="last")

        if price_cols:
            cleaned_df[price_cols] = cleaned_df.groupby("ticker")[price_cols].transform(
                lambda series: series.ffill().bfill()
            )

        if "volume" in cleaned_df.columns:
            cleaned_df["volume"] = cleaned_df.groupby("ticker")["volume"].transform(
                lambda series: series.ffill().bfill()
            )
            cleaned_df["volume"] = cleaned_df["volume"].fillna(0)

        missing_after = cleaned_df[numeric_cols].isna().sum().to_dict() if numeric_cols else {}
        rows_after = len(cleaned_df)

        logging.info(
            "Market data cleaning done. rows_before=%d rows_after=%d missing_before=%s missing_after=%s",
            rows_before,
            rows_after,
            missing_before,
            missing_after,
        )
        return cleaned_df

    def fetch_market_data_window(
        self,
        start_date: date,
        end_date: date,
    ) -> pd.DataFrame:
        if len(self.config.tickers) != 5:
            raise ValueError("Exactly 5 tickers are required for this ingestion job.")

        logging.info(
            "Fetching market data window start=%s end=%s (lookback_days=%d)",
            start_date.isoformat(),
            end_date.isoformat(),
            self.config.lookback_days,
        )

        all_frames = []
        failed_tickers = []
        for ticker in self.config.tickers:
            try:
                logging.info("Fetching market data for ticker: %s", ticker)
                data = yf.download(
                    tickers=ticker,
                    start=start_date.isoformat(),
                    end=(end_date + timedelta(days=1)).isoformat(),
                    interval=self.config.interval,
                    progress=False,
                    auto_adjust=False,
                    group_by="column",
                )
                if data.empty:
                    logging.warning("No market rows returned for %s", ticker)
                    failed_tickers.append(ticker)
                    continue

                normalized = self.normalize_market_frame(data, ticker)
                all_frames.append(normalized)
            except Exception as error:
                logging.exception("Failed to fetch market data for %s. Error: %s", ticker, error)
                failed_tickers.append(ticker)

        if not all_frames:
            raise ValueError("Market data download failed for all configured tickers.")

        market_df = pd.concat(all_frames, ignore_index=True)
        market_df = self.clean_market_data(market_df)
        market_df = market_df[
            (market_df["date"] >= start_date) & (market_df["date"] <= end_date)
        ].reset_index(drop=True)
        logging.info(
            "Market data fetched for %d/%d tickers. Failed tickers: %s",
            len(self.config.tickers) - len(failed_tickers),
            len(self.config.tickers),
            failed_tickers,
        )
        return market_df

    def evaluate_market_cache(
        self,
        market_df: pd.DataFrame,
        start_date: date,
        end_date: date,
    ) -> tuple[str, date | None]:
        df = market_df.copy()
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
        df = df.dropna(subset=["date", "ticker"])
        if df.empty:
            return "reload", None

        append_from: date | None = None
        for ticker in self.config.tickers:
            ticker_df = df[df["ticker"] == ticker]
            if ticker_df.empty:
                logging.info("Cache check: ticker %s missing in market CSV.", ticker)
                return "reload", None

            ticker_min = ticker_df["date"].min()
            ticker_max = ticker_df["date"].max()
            if ticker_min > start_date:
                logging.info(
                    "Cache check: old market data missing for %s (min=%s > start=%s).",
                    ticker,
                    ticker_min,
                    start_date,
                )
                return "reload", None
            if ticker_max < end_date:
                candidate = ticker_max + timedelta(days=1)
                append_from = candidate if append_from is None else min(append_from, candidate)

        if append_from is not None:
            return "append", append_from
        return "complete", None
