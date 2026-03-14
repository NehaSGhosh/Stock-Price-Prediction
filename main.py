import argparse
import json
import os
import sys

SRC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from market_predictor.exception import CustomException
from market_predictor.logger import logging
from config.configuration import ConfigurationManager
from market_predictor.ingestion.ingestion_pipeline import ingest_data
from market_predictor.ml_pipeline.model_predictor import fastapi_predict
from market_predictor.ml_pipeline.model_trainer import DEFAULT_MODEL_PATH
from market_predictor.ml_pipeline.training_pipeline import TrainingPipeline, train_model

_config_manager = ConfigurationManager()

try:
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel
except Exception:  # pragma: no cover - FastAPI may be optional when running CLI only.
    FastAPI = None
    HTTPException = None
    Request = None
    JSONResponse = None
    BaseModel = object


class PredictionRequest(BaseModel):
    ticker: str
    headline: str
    model_output_path: str | None = None


app = FastAPI(title="Market Predictor API") if FastAPI is not None else None

if app is not None:
    class _CloudRequestAdapter:
        def __init__(self, args: dict[str, str], payload: dict):
            self.args = args
            self._payload = payload

        def get_json(self, silent: bool = True):
            _ = silent
            return self._payload

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

    @app.post("/ingest_data_http")
    async def ingest_data_http_endpoint(request: Request):
        try:
            payload = await request.json()
            if payload is None:
                payload = {}
        except Exception:
            payload = {}
        adapter = _CloudRequestAdapter(args=dict(request.query_params), payload=payload)
        body, status_code, headers = ingest_data(adapter)
        return JSONResponse(content=json.loads(body), status_code=status_code, headers=headers)

    @app.post("/train_model_http")
    async def train_model_http_endpoint(request: Request):
        try:
            payload = await request.json()
            if payload is None:
                payload = {}
        except Exception:
            payload = {}
        adapter = _CloudRequestAdapter(args=dict(request.query_params), payload=payload)
        body, status_code, headers = train_model(adapter)
        return JSONResponse(content=json.loads(body), status_code=status_code, headers=headers)


def run_train_pipeline(
    refresh_from_api: bool = False,
    model_output_path: str = DEFAULT_MODEL_PATH,
) -> None:
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
