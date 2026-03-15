import os
import ast
import json
from typing import Optional

import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from stock_price_predictor.logger import logging
from stock_price_predictor.utils.common import ensure_dir, parse_gcs_uri, read_from_gcs, upload_to_gcs


class SentimentScoring:
    """
    Unified sentiment + feature engineering utility.
    """

    def __init__(
        self,
        text_column: str = "headline",
        output_column: str = "sentiment_compound",
    ):
        self.text_column = text_column
        self.output_column = output_column
        self.analyzer = SentimentIntensityAnalyzer()

    def score_headline(self, text: Optional[str]) -> float:
        if not isinstance(text, str) or not text.strip():
            abc = 0.0
        else:
            abc = float(self.analyzer.polarity_scores(text)["compound"])
        return abc

    def score_dataframe(self, news_df: pd.DataFrame) -> pd.DataFrame:
        if self.text_column not in news_df.columns:
            raise ValueError(f"Expected text column '{self.text_column}' in news dataframe.")

        scored_df = news_df.copy()
        scored_df[self.output_column] = scored_df[self.text_column].apply(self.score_headline)
        logging.info(
            "Sentiment scoring completed. rows=%d output_column=%s",
            len(scored_df),
            self.output_column,
        )
        return scored_df

    def score_csv(self, input_path: str, output_path: Optional[str] = None) -> pd.DataFrame:
        if input_path.startswith("gs://"):
            bucket_name, blob_name = parse_gcs_uri(input_path)
            news_df = read_from_gcs(bucket_name=bucket_name, blob_name=blob_name)
        else:
            news_df = pd.read_csv(input_path)
        scored_df = self.score_dataframe(news_df)

        if output_path:
            if output_path.startswith("gs://"):
                bucket_name, blob_name = parse_gcs_uri(output_path)
                upload_to_gcs(df=scored_df, bucket_name=bucket_name, destination_blob=blob_name)
            else:
                ensure_dir(os.path.dirname(output_path))
                scored_df.to_csv(output_path, index=False)
            logging.info("Scored news file saved at: %s", output_path)

        return scored_df

    def aggregate_daily_sentiment(self, scored_df: pd.DataFrame) -> pd.DataFrame:
        required_cols = {"date", "ticker", self.output_column}
        missing_cols = required_cols.difference(scored_df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns for aggregation: {sorted(missing_cols)}")

        daily_df = scored_df.copy()
        daily_df["date"] = pd.to_datetime(daily_df["date"], errors="coerce").dt.date
        daily_df = daily_df.dropna(subset=["date", "ticker", self.output_column])

        aggregated_df = (
            daily_df.groupby(["ticker", "date"], as_index=False)[self.output_column]
            .mean()
            .rename(columns={self.output_column: "daily_avg_sentiment"})
            .sort_values(["ticker", "date"])
            .reset_index(drop=True)
        )
        return aggregated_df

    def aggregate_daily_sentiment_csv(
        self,
        scored_input_path: str,
        output_path: Optional[str] = None,
    ) -> pd.DataFrame:
        if scored_input_path.startswith("gs://"):
            bucket_name, blob_name = parse_gcs_uri(scored_input_path)
            scored_df = read_from_gcs(bucket_name=bucket_name, blob_name=blob_name)
        else:
            scored_df = pd.read_csv(scored_input_path)
        aggregated_df = self.aggregate_daily_sentiment(scored_df)

        if output_path:
            if output_path.startswith("gs://"):
                bucket_name, blob_name = parse_gcs_uri(output_path)
                upload_to_gcs(df=aggregated_df, bucket_name=bucket_name, destination_blob=blob_name)
            else:
                ensure_dir(os.path.dirname(output_path))
                aggregated_df.to_csv(output_path, index=False)
            logging.info("Daily aggregated sentiment file saved at: %s", output_path)

        return aggregated_df

    @staticmethod
    def _to_headlines_list(value) -> list[str]:
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return []
        text = str(value).strip()
        if not text:
            return []
        # Try JSON first, then Python literal list format.
        for parser in (json.loads, ast.literal_eval):
            try:
                parsed = parser(text)
                if isinstance(parsed, list):
                    return [str(item).strip() for item in parsed if str(item).strip()]
            except Exception:
                continue
        return []

    @classmethod
    def create_gold_with_features_for_training(
        cls,
        gold_df: pd.DataFrame,
        output_path: str | None = None,
    ) -> pd.DataFrame:
        working_gold_df = gold_df.copy()
        if "date" in working_gold_df.columns:
            working_gold_df["date"] = pd.to_datetime(working_gold_df["date"], errors="coerce")
        working_gold_df = working_gold_df.dropna(subset=["date", "ticker"]).sort_values(["ticker", "date"])
        working_gold_df = working_gold_df.drop_duplicates(subset=["ticker", "date"], keep="last").reset_index(drop=True)
        if "headlines_list" not in working_gold_df.columns:
            working_gold_df["headlines_list"] = [[] for _ in range(len(working_gold_df))]

        sentiment_scorer = cls(text_column="headline", output_column="sentiment_compound")
        working_gold_df["headlines_list"] = working_gold_df["headlines_list"].apply(cls._to_headlines_list)
        working_gold_df["avg_sentiment_headlines"] = working_gold_df["headlines_list"].apply(
            lambda items: (
                sum(sentiment_scorer.score_headline(item) for item in items) / len(items)
                if items
                else 0.0
            )
        )

        feature_df = cls.build_rolling_features(working_gold_df)
        feature_df["date"] = pd.to_datetime(feature_df["date"], errors="coerce").dt.date
        feature_df = feature_df.dropna(subset=["date", "ticker"]).sort_values(["ticker", "date"])
        feature_df = feature_df.drop_duplicates(subset=["ticker", "date"], keep="last").reset_index(drop=True)
        if output_path:
            if output_path.startswith("gs://"):
                bucket_name, blob_name = parse_gcs_uri(output_path)
                upload_to_gcs(df=feature_df, bucket_name=bucket_name, destination_blob=blob_name)
            else:
                ensure_dir(os.path.dirname(output_path))
                feature_df.to_csv(output_path, index=False)
            logging.info("Gold with features CSV saved at: %s (rows=%d)", output_path, len(feature_df))
        return feature_df

    @classmethod
    def create_gold_with_features_for_training_pipeline(
        cls,
        gold_df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, str]:
        from config.configuration import ConfigurationManager

        cfg = ConfigurationManager()
        output_path = cfg.get_gold_with_features_path()
        feature_df = cls.create_gold_with_features_for_training(
            gold_df=gold_df,
            output_path=output_path,
        )
        return feature_df, output_path

    @staticmethod
    def build_rolling_features(stock_sentiment_df: pd.DataFrame) -> pd.DataFrame:
        df = stock_sentiment_df.copy()
        required = {"ticker", "date", "close", "volume"}
        missing = required.difference(df.columns)
        if missing:
            raise ValueError(f"Missing required columns for feature engineering: {sorted(missing)}")

        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date", "ticker"]).sort_values(["ticker", "date"]).reset_index(drop=True)

        for numeric_col in ["open", "high", "low", "close", "adj_close", "volume"]:
            if numeric_col in df.columns:
                df[numeric_col] = pd.to_numeric(df[numeric_col], errors="coerce")
        if "daily_avg_sentiment" in df.columns:
            df = df.drop(columns=["daily_avg_sentiment"])

        grouped_close = df.groupby("ticker")["close"]
        grouped_volume = df.groupby("ticker")["volume"]

        df["return_1d"] = grouped_close.pct_change(1, fill_method=None)
        df["return_3d"] = grouped_close.pct_change(3, fill_method=None)
        df["return_5d"] = grouped_close.pct_change(5, fill_method=None)
        df["close_lag_1"] = grouped_close.shift(1)
        df["close_lag_2"] = grouped_close.shift(2)
        df["close_lag_3"] = grouped_close.shift(3)
        df["close_to_lag_1"] = (df["close"] / df["close_lag_1"]) - 1.0
        df["ma_3"] = grouped_close.transform(lambda series: series.rolling(3).mean())
        df["ma_7"] = grouped_close.transform(lambda series: series.rolling(7).mean())
        df["ma_14"] = grouped_close.transform(lambda series: series.rolling(14).mean())
        df["price_vs_ma_7"] = (df["close"] / df["ma_7"]) - 1.0
        df["ma_ratio_3_7"] = (df["ma_3"] / df["ma_7"]) - 1.0
        df["volatility_5"] = grouped_close.transform(
            lambda series: series.pct_change(fill_method=None).rolling(5).std()
        )
        df["day_of_week"] = df["date"].dt.dayofweek

        df["volume_change_1d"] = grouped_volume.pct_change(1, fill_method=None)
        df["volume_ma_5"] = grouped_volume.transform(lambda series: series.rolling(5).mean())
        df["volume_vs_ma_5"] = (df["volume"] / df["volume_ma_5"]) - 1.0

        if {"high", "low"}.issubset(df.columns):
            df["intraday_range_pct"] = (df["high"] - df["low"]) / df["close"]

        df["next_close"] = grouped_close.shift(-1)
        df["target_up"] = (df["next_close"] > df["close"]).astype(int)

        df = df.replace([float("inf"), float("-inf")], pd.NA)
        excluded = {"date", "ticker", "next_close", "target_up"}
        feature_cols = [col for col in df.columns if col not in excluded]
        for col in feature_cols:
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df.groupby("ticker")[col].ffill()
                median_value = df[col].median()
                if pd.isna(median_value):
                    median_value = 0.0
                df[col] = df[col].fillna(median_value)

        df = df.dropna(subset=["next_close", "close"]).reset_index(drop=True)
        return df
