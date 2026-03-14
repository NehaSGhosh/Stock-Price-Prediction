import os
from datetime import date, timedelta

import pandas as pd
from dotenv import load_dotenv
from newsapi import NewsApiClient

from market_predictor.entity.config_entity import DataIngestionConfig
from market_predictor.logger import logging


class NewsIngestion:
    TICKER_QUERY_HINTS = {
        "MSFT": "Microsoft",
        "AAPL": "Apple",
        "GOOG": "Google",
        "AVGO": "Broadcom",
        "UBER": "Uber",
    }

    def __init__(self, config: DataIngestionConfig):
        self.config = config

    @staticmethod
    def clean_news_data(news_df: pd.DataFrame) -> pd.DataFrame:
        cleaned_df = news_df.copy()
        rows_before = len(cleaned_df)

        cleaned_df["date"] = pd.to_datetime(cleaned_df["date"], errors="coerce", utc=True).dt.date
        cleaned_df = cleaned_df.dropna(subset=["date", "ticker", "headline"])
        cleaned_df = cleaned_df.sort_values(["ticker", "date"]).reset_index(drop=True)

        rows_after = len(cleaned_df)
        logging.info(
            "News data cleaning done. rows_before=%d rows_after=%d",
            rows_before,
            rows_after,
        )
        return cleaned_df

    def fetch_news_data_newsapi(
        self,
        from_date: date,
        to_date: date,
    ) -> pd.DataFrame:
        load_dotenv()
        news_api_key = os.getenv("NEWSAPI_KEY", "").strip().strip('"').strip("'")
        if not news_api_key:
            raise ValueError("NEWSAPI_KEY not set in environment/.env.")

        newsapi = NewsApiClient(api_key=news_api_key)
        tickers = self.config.tickers
        all_rows: list[dict] = []
        failed_tickers: list[str] = []

        logging.info(
            "Fetching NewsAPI data for tickers=%s from=%s to=%s",
            tickers,
            from_date,
            to_date,
        )
        for ticker in tickers:
            company = self.TICKER_QUERY_HINTS.get(ticker, ticker)
            query = f'"{company}" or "{ticker}"'
            try:
                payload = newsapi.get_everything(
                    q=query,
                    language="en",
                    from_param=from_date.isoformat(),
                    to=to_date.isoformat(),
                    sort_by='relevancy'
                )
                if payload.get("status") != "ok":
                    failed_tickers.append(ticker)
                    logging.warning(
                        "NewsAPI request failed for %s. status=%s message=%s",
                        ticker,
                        payload.get("status"),
                        payload.get("message", "")[:300],
                    )
                    continue
                logging.info(
                    "NewsAPI totalResults for %s: %s",
                    ticker,
                    payload.get("totalResults", 0),
                )
                articles = payload.get("articles", [])
                for article in articles:
                    all_rows.append(
                        {
                            "date": article.get("publishedAt"),
                            "ticker": ticker,
                            "headline": article.get("title"),
                            "url": article.get("url"),
                        }
                    )
            except Exception as error:
                failed_tickers.append(ticker)
                logging.exception(
                    "NewsAPI request exception for %s. Error: %s",
                    ticker,
                    error,
                )

        if not all_rows:
            raise ValueError(f"NewsAPI returned no rows for all tickers. failed={failed_tickers}")

        raw_df = pd.DataFrame(all_rows)
        raw_df["date"] = pd.to_datetime(raw_df["date"], errors="coerce", utc=True).dt.date
        raw_df = raw_df[
            (raw_df["date"] >= from_date) & (raw_df["date"] <= to_date)
        ].reset_index(drop=True)

        news_df = self.clean_news_data(raw_df)
        if news_df.empty:
            raise ValueError("NewsAPI returned rows but none in configured date window after cleaning.")

        logging.info(
            "NewsAPI news fetched successfully. rows=%d failed_tickers=%s",
            len(news_df),
            failed_tickers,
        )
        return news_df

    @staticmethod
    def evaluate_news_cache(
        news_df: pd.DataFrame,
        start_date: date,
        end_date: date,
    ) -> tuple[str, date | None]:
        df = news_df.copy()
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
        df = df.dropna(subset=["date"])
        if df.empty:
            return "reload", None

        min_date = df["date"].min()
        max_date = df["date"].max()
        if min_date > start_date:
            return "reload", None
        if max_date < end_date:
            return "append", max_date + timedelta(days=1)
        return "complete", None
