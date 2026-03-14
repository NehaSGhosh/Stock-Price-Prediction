import os
import sys
from typing import Any

SRC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from market_predictor.ingestion.ingestion_pipeline import ingest_data
from market_predictor.ml_pipeline.training_pipeline import train_model


def ingest_data_http(request: Any):
    return ingest_data(request)


def train_model_http(request: Any):
    return train_model(request)
