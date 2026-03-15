import argparse
import json
import os
import sys

SRC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from stock_price_predictor.exception import CustomException
from stock_price_predictor.logger import logging
from config.configuration import ConfigurationManager
from stock_price_predictor.ingestion.ingestion_pipeline import ingest_data
from stock_price_predictor.ml_pipeline.model_predictor import fastapi_predict
from stock_price_predictor.ml_pipeline.model_trainer import DEFAULT_MODEL_PATH
from stock_price_predictor.ml_pipeline.training_pipeline import TrainingPipeline, train_model

_config_manager = ConfigurationManager()

try:
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.responses import HTMLResponse
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel
except Exception:  # pragma: no cover - FastAPI may be optional when running CLI only.
    FastAPI = None
    HTTPException = None
    Request = None
    HTMLResponse = None
    JSONResponse = None
    BaseModel = object


class PredictionRequest(BaseModel):
    """Request payload for trend prediction.

    Attributes:
        ticker: Stock ticker symbol (for example, GOOG).
        headline: News headline used for sentiment-adjusted inference.
        model_output_path: Optional model artifact path override.
    """
    ticker: str
    headline: str
    model_output_path: str | None = None


app = FastAPI(title="Market Predictor API") if FastAPI is not None else None

if app is not None:
    PREDICT_UI_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ui", "predict.html")

    def get_predict_tickers() -> list[str]:
        """Return configured tickers for prediction UI.

        Returns:
            list[str]: Tickers loaded from ingestion configuration.
        """
        return _config_manager.get_data_ingestion_config().tickers

    class _CloudRequestAdapter:
        # Adapts FastAPI request data to the Cloud Functions request interface.
        def __init__(self, args: dict[str, str], payload: dict):
            self.args = args
            self._payload = payload

        def get_json(self, silent: bool = True):
            _ = silent
            return self._payload

    @app.post("/predict")
    def predict_endpoint(payload: PredictionRequest):
        """Serve model inference for a ticker/headline pair.

        Args:
            payload: JSON body with `ticker`, `headline`, and optional `model_output_path`.

        Returns:
            dict: Prediction output with source date, sentiment, class, and probability.
        """
        try:
            return fastapi_predict(
                ticker=payload.ticker,
                headline=payload.headline,
                model_output_path=payload.model_output_path or DEFAULT_MODEL_PATH,
            )
        except Exception as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

    @app.get("/predict-trend")
    def predict_ui():
        """Serve the prediction UI HTML page with injected ticker options.

        Returns:
            HTMLResponse: Rendered HTML response for browser-based prediction.
        """
        if not os.path.exists(PREDICT_UI_PATH):
            raise HTTPException(status_code=404, detail="Predict UI file not found.")
        with open(PREDICT_UI_PATH, "r", encoding="utf-8") as ui_file:
            html_content = ui_file.read()
        html_content = html_content.replace("__PREDICT_TICKERS__", json.dumps(get_predict_tickers()))
        return HTMLResponse(content=html_content, media_type="text/html")

    @app.post("/ingest")
    async def ingest_endpoint(request: Request):
        """Run ingestion via FastAPI using cloud-function-compatible handler.

        Args:
            request: FastAPI request carrying query params (`lookback_days`/`append`) and optional JSON body.

        Returns:
            JSONResponse: Ingestion execution result and output paths.
        """
        try:
            payload = await request.json()
            if payload is None:
                payload = {}
        except Exception:
            # Keep behavior consistent with cloud function handlers for empty/non-JSON bodies.
            payload = {}
        adapter = _CloudRequestAdapter(args=dict(request.query_params), payload=payload)
        body, status_code, headers = ingest_data(adapter)
        return JSONResponse(content=json.loads(body), status_code=status_code, headers=headers)

    @app.post("/train")
    async def train_endpoint(request: Request):
        """Run model training via FastAPI using cloud-function-compatible handler.

        Args:
            request: FastAPI request with optional `refresh_from_api` and `model_output_path` in query/body.

        Returns:
            JSONResponse: Training execution result and artifact locations.
        """
        try:
            payload = await request.json()
            if payload is None:
                payload = {}
        except Exception:
            # Keep behavior consistent with cloud function handlers for empty/non-JSON bodies.
            payload = {}
        adapter = _CloudRequestAdapter(args=dict(request.query_params), payload=payload)
        body, status_code, headers = train_model(adapter)
        return JSONResponse(content=json.loads(body), status_code=status_code, headers=headers)


def run_train_pipeline(
    refresh_from_api: bool = False,
    model_output_path: str = DEFAULT_MODEL_PATH,
) -> None:
    """Execute training pipeline from CLI mode.

    Args:
        refresh_from_api: Reserved flag for API refresh behavior.
        model_output_path: Destination path for the trained model artifact.

    Returns:
        None
    """
    result = TrainingPipeline().run_model_training_pipeline(
        refresh_from_api=refresh_from_api,
        model_output_path=model_output_path,
    )
    logging.info("Train pipeline completed. outputs=%s", result)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Market prediction pipeline.")
    parser.add_argument("--mode", choices=["train"], default="train")
    parser.add_argument("--refresh_from_api", action="store_true", help="Fetch fresh market/news data even if CSV files exist.")
    parser.add_argument("--model_output_path", default=DEFAULT_MODEL_PATH)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    try:
        logging.info("Main execution started in %s mode.", args.mode)
        if args.mode == "train":
            run_train_pipeline(
                refresh_from_api=args.refresh_from_api,
                model_output_path=args.model_output_path,
            )
    except Exception as error:
        raise CustomException(error, sys) from error
