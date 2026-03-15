import os
import sys
from datetime import datetime, timedelta
from typing import Any

import pandas as pd

CURRENT_FILE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.abspath(os.path.join(CURRENT_FILE_DIR, "..", ".."))
PROJECT_ROOT_DIR = os.path.abspath(os.path.join(CURRENT_FILE_DIR, "..", "..", ".."))
if PROJECT_ROOT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_ROOT_DIR)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from config.configuration import ConfigurationManager
from stock_price_predictor.logger import logging
from stock_price_predictor.ml_pipeline.feature_engineering import SentimentScoring
from stock_price_predictor.ml_pipeline.model_trainer import DEFAULT_MODEL_PATH
from stock_price_predictor.utils.common import gcs_blob_exists, load_object, parse_gcs_uri, read_from_gcs
from stock_price_predictor.warehousing.data_storage import GoldWarehouse

class ModelPredictor:
    def __init__(self) -> None:
        """Initialize prediction service with runtime configuration."""
        self.config_manager = ConfigurationManager()
        self.target_column = self.config_manager.get_classification_target_column()
        self.ingestion_cfg = self.config_manager.get_data_ingestion_config()

    @staticmethod
    def _require_non_empty(value: str, field_name: str) -> str:
        """Validate required text input.

        Args:
            value: Raw user-provided value.
            field_name: Field name for error context.

        Returns:
            str: Trimmed non-empty value.
        """
        normalized = str(value).strip() if value is not None else ""
        if not normalized:
            raise ValueError(f"{field_name} is required and cannot be empty.")
        return normalized

    def _resolve_gold_with_features(self) -> pd.DataFrame:
        """Load and window the gold-with-features dataset from GCS.

        Returns:
            pd.DataFrame: Feature dataset filtered to configured lookback window.
        """
        gcs_path = self.config_manager.get_gold_with_features_path()
        end_date = datetime.now(GoldWarehouse.EASTERN_TZ).date()
        start_date = end_date - timedelta(days=self.ingestion_cfg.lookback_days - 1)
        if not gcs_path.startswith("gs://"):
            raise ValueError(f"gold_with_features path must be a GCS URI, got: {gcs_path}")
        bucket_name, blob_name = parse_gcs_uri(gcs_path)
        logging.info("Fetching gold_with_features CSV from GCS: %s", gcs_path)
        df = read_from_gcs(bucket_name=bucket_name, blob_name=blob_name)
        if "date" in df.columns:
            # Restrict prediction context to the configured lookback window.
            df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
            df = df[(df["date"] >= start_date) & (df["date"] <= end_date)].reset_index(drop=True)
        if df.empty:
            raise FileNotFoundError(
                f"No matching rows found in {gcs_path} for window {start_date} to {end_date}."
            )
        return df

    @staticmethod
    def _prepare_inference_features(feature_df: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
        """Transform latest ticker row into model-ready inference matrix.

        Args:
            feature_df: Source dataframe containing latest ticker row.
            feature_columns: Ordered feature columns expected by trained model.

        Returns:
            pd.DataFrame: Single-row inference dataframe aligned to training schema.
        """
        working_df = feature_df.copy()
        if "ticker" in working_df.columns:
            working_df = pd.get_dummies(working_df, columns=["ticker"], prefix="ticker", dtype=int)

        x = working_df.drop(
            columns=[
                "date",
                "headline",
                "headlines_list",
                "next_close",
                "next_day_price_change_pct",
                "target_up",
            ],
            errors="ignore",
        ).copy()
        x = x.replace([float("inf"), float("-inf")], pd.NA)
        for col in x.columns:
            if pd.api.types.is_numeric_dtype(x[col]):
                x[col] = x[col].fillna(x[col].median())
            else:
                x[col] = x[col].fillna(0)

        for col in feature_columns:
            # Align inference columns with the exact schema used during training.
            if col not in x.columns:
                x[col] = 0
        return x[feature_columns]

    def predict_from_ticker_headline(
        self,
        ticker: str,
        headline: str,
        model_output_path: str = DEFAULT_MODEL_PATH,
    ) -> dict[str, Any]:
        """Predict trend direction using latest ticker features and input headline.

        Args:
            ticker: Stock ticker symbol.
            headline: News headline used to compute sentiment override.
            model_output_path: Path to serialized model payload.

        Returns:
            dict[str, Any]: Prediction payload with source date, sentiment, class, and probability.
        """
        ticker = self._require_non_empty(ticker, "ticker").upper()
        headline = self._require_non_empty(headline, "headline")

        if model_output_path.startswith("gs://"):
            bucket_name, blob_name = model_output_path[5:].split("/", 1)
            if not gcs_blob_exists(bucket_name=bucket_name, blob_name=blob_name):
                raise FileNotFoundError(f"Model file not found: {model_output_path}")
        elif not os.path.exists(model_output_path):
            raise FileNotFoundError(f"Model file not found: {model_output_path}")

        gold_features_df = self._resolve_gold_with_features()
        if "ticker" not in gold_features_df.columns:
            raise ValueError("gold_with_features dataset is missing required 'ticker' column.")
        if "date" not in gold_features_df.columns:
            raise ValueError("gold_with_features dataset is missing required 'date' column.")

        ticker_rows = gold_features_df[
            gold_features_df["ticker"].astype(str).str.upper().str.strip() == ticker
        ].copy()
        if ticker_rows.empty:
            raise ValueError(f"Ticker {ticker} not found in gold_with_features dataset.")

        ticker_rows["date"] = pd.to_datetime(ticker_rows["date"], errors="coerce")
        ticker_rows = ticker_rows.dropna(subset=["date"]).sort_values("date")
        if ticker_rows.empty:
            raise ValueError(f"No valid dated rows found for ticker {ticker} in gold_with_features dataset.")

        latest_row = ticker_rows.tail(1).copy().reset_index(drop=True)
        sentiment = SentimentScoring().score_headline(headline)
        # Override sentiment-related inputs with the current request headline.
        latest_row["headlines_list"] = [[headline]]
        latest_row["avg_sentiment_headlines"] = float(sentiment)

        model_payload = load_object(model_output_path)
        if not isinstance(model_payload, dict):
            raise ValueError("Invalid model payload. Expected a dictionary with model and feature_columns.")
        model = model_payload.get("model")
        feature_columns = model_payload.get("feature_columns", [])
        if model is None or not feature_columns:
            raise ValueError("Model payload is missing model and/or feature_columns.")

        x_infer = self._prepare_inference_features(latest_row, feature_columns)
        predicted_class = int(model.predict(x_infer)[0])
        predicted_probability = None
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(x_infer)
            if probabilities is not None and len(probabilities) > 0:
                predicted_probability = float(probabilities[0][1]) if len(probabilities[0]) > 1 else float(probabilities[0][0])

        prediction_date = latest_row.loc[0, "date"]
        prediction_date_str = prediction_date.date().isoformat() if hasattr(prediction_date, "date") else str(prediction_date)
        result = {
            "ticker": ticker,
            "source_date": prediction_date_str,
            "headline": headline,
            "sentiment_score": float(sentiment),
            "predicted_target_up": predicted_class,
            "predicted_probability_up": predicted_probability,
        }
        logging.info(
            "Prediction request processed. ticker=%s date=%s sentiment_score=%.4f predicted_target_up=%d",
            ticker,
            prediction_date_str,
            sentiment,
            predicted_class,
        )
        return result


def fastapi_predict(
    ticker: str,
    headline: str,
    model_output_path: str = DEFAULT_MODEL_PATH,
) -> dict[str, Any]:
    """FastAPI-facing helper for ticker/headline prediction.

    Args:
        ticker: Stock ticker symbol.
        headline: News headline text.
        model_output_path: Optional model artifact path override.

    Returns:
        dict[str, Any]: Prediction payload generated by ModelPredictor.
    """
    predictor = ModelPredictor()
    return predictor.predict_from_ticker_headline(
        ticker=ticker,
        headline=headline,
        model_output_path=model_output_path,
    )
