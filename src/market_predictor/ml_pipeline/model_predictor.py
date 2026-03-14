import argparse
import json
import os
import sys
from datetime import datetime, timedelta
from typing import Any

import pandas as pd

if __package__ in {None, ""}:
    CURRENT_FILE_DIR = os.path.dirname(os.path.abspath(__file__))
    SRC_DIR = os.path.abspath(os.path.join(CURRENT_FILE_DIR, "..", ".."))
    PROJECT_ROOT_DIR = os.path.abspath(os.path.join(CURRENT_FILE_DIR, "..", "..", ".."))
    if PROJECT_ROOT_DIR not in sys.path:
        sys.path.insert(0, PROJECT_ROOT_DIR)
    if SRC_DIR not in sys.path:
        sys.path.insert(0, SRC_DIR)

from config.configuration import ConfigurationManager
from market_predictor.logger import logging
from market_predictor.ml_pipeline.feature_engineering import SentimentScoring
from market_predictor.ml_pipeline.model_trainer import DEFAULT_MODEL_PATH
from market_predictor.utils.common import load_object
from market_predictor.warehousing.data_storage import GoldWarehouse

try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel
except Exception:  # pragma: no cover - FastAPI may be optional at runtime.
    FastAPI = None
    HTTPException = None
    BaseModel = object


class PredictionRequest(BaseModel):
    ticker: str
    headline: str
    model_output_path: str | None = None


class ModelPredictor:
    def __init__(self) -> None:
        self.config_manager = ConfigurationManager()
        self.target_column = self.config_manager.get_classification_target_column()
        self.gold_with_features_path = self.config_manager.get_gold_with_features_path()
        self.ingestion_cfg = self.config_manager.get_data_ingestion_config()

    @staticmethod
    def _require_non_empty(value: str, field_name: str) -> str:
        normalized = str(value).strip() if value is not None else ""
        if not normalized:
            raise ValueError(f"{field_name} is required and cannot be empty.")
        return normalized

    def _resolve_gold_with_features(self) -> pd.DataFrame:
        path = self.gold_with_features_path
        if os.path.exists(path):
            df = pd.read_csv(path)
            if df.empty:
                raise ValueError(f"Local gold_with_features.csv is empty: {path}")
            return df

        warehouse = GoldWarehouse.from_config()
        end_date = datetime.now(GoldWarehouse.EASTERN_TZ).date()
        start_date = end_date - timedelta(days=self.ingestion_cfg.lookback_days - 1)
        logging.info(
            "Local gold_with_features.csv not found. Fetching BigQuery table %s.",
            warehouse.gold_with_features_table_id,
        )
        df = warehouse.fetch_table_from_bigquery(
            table_id=warehouse.gold_with_features_table_id,
            start_date=start_date.isoformat(),
            end_date=end_date.isoformat(),
        )
        if df.empty:
            raise FileNotFoundError(
                "gold_with_features.csv not found locally and no matching rows found in BigQuery table "
                f"{warehouse.gold_with_features_table_id} for window {start_date} to {end_date}."
            )
        GoldWarehouse.save_gold_csv(df, path)
        return df

    @staticmethod
    def _prepare_inference_features(feature_df: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
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
            if col not in x.columns:
                x[col] = 0
        return x[feature_columns]

    def predict_from_ticker_headline(
        self,
        ticker: str,
        headline: str,
        model_output_path: str = DEFAULT_MODEL_PATH,
    ) -> dict[str, Any]:
        ticker = self._require_non_empty(ticker, "ticker").upper()
        headline = self._require_non_empty(headline, "headline")

        if not os.path.exists(model_output_path):
            raise FileNotFoundError(f"Model file not found: {model_output_path}")

        gold_features_df = self._resolve_gold_with_features()
        if "ticker" not in gold_features_df.columns:
            raise ValueError("gold_with_features.csv is missing required 'ticker' column.")
        if "date" not in gold_features_df.columns:
            raise ValueError("gold_with_features.csv is missing required 'date' column.")

        ticker_rows = gold_features_df[
            gold_features_df["ticker"].astype(str).str.upper().str.strip() == ticker
        ].copy()
        if ticker_rows.empty:
            raise ValueError(f"Ticker {ticker} not found in gold_with_features.csv.")

        ticker_rows["date"] = pd.to_datetime(ticker_rows["date"], errors="coerce")
        ticker_rows = ticker_rows.dropna(subset=["date"]).sort_values("date")
        if ticker_rows.empty:
            raise ValueError(f"No valid dated rows found for ticker {ticker} in gold_with_features.csv.")

        latest_row = ticker_rows.tail(1).copy().reset_index(drop=True)
        sentiment = SentimentScoring().score_headline(headline)
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
            "date": prediction_date_str,
            "headline": headline,
            "avg_sentiment_headlines": float(sentiment),
            "predicted_target_up": predicted_class,
            "predicted_probability_up": predicted_probability,
            "model_path": model_output_path,
            "gold_with_features_path": self.gold_with_features_path,
        }
        logging.info(
            "Prediction request processed. ticker=%s date=%s avg_sentiment_headlines=%.4f predicted_target_up=%d",
            ticker,
            prediction_date_str,
            sentiment,
            predicted_class,
        )
        return result


def run_predict_script(
    ticker: str,
    headline: str,
    model_output_path: str = DEFAULT_MODEL_PATH,
) -> dict[str, Any]:
    predictor = ModelPredictor()
    result = predictor.predict_from_ticker_headline(
        ticker=ticker,
        headline=headline,
        model_output_path=model_output_path,
    )
    logging.info("Prediction completed. result=%s", result)
    return result


def fastapi_predict(
    ticker: str,
    headline: str,
    model_output_path: str = DEFAULT_MODEL_PATH,
) -> dict[str, Any]:
    predictor = ModelPredictor()
    return predictor.predict_from_ticker_headline(
        ticker=ticker,
        headline=headline,
        model_output_path=model_output_path,
    )


app = FastAPI(title="Market Predictor API") if FastAPI is not None else None


if app is not None:
    @app.post("/predict")
    def predict_endpoint(payload: PredictionRequest):
        try:
            return fastapi_predict(
                ticker=payload.ticker,
                headline=payload.headline,
                model_output_path=payload.model_output_path or DEFAULT_MODEL_PATH,
            )
        except Exception as error:
            raise HTTPException(status_code=400, detail=str(error)) from error


def _run_as_script() -> None:
    parser = argparse.ArgumentParser(description="Predict target_up from ticker and latest feature row.")
    parser.add_argument("--ticker", required=True, help="Ticker symbol to predict for, e.g. MSFT")
    parser.add_argument("--headline", required=True, help="Latest headline text for the ticker")
    parser.add_argument("--model_output_path", default=DEFAULT_MODEL_PATH, help="Path to trained model payload")
    args = parser.parse_args()

    result = run_predict_script(
        ticker=args.ticker,
        headline=args.headline,
        model_output_path=args.model_output_path,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    _run_as_script()
