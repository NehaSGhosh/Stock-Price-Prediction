import argparse
import os
import sys

SRC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from market_predictor.exception import CustomException
from market_predictor.logger import logging
from config.configuration import ConfigurationManager
from market_predictor.ml_pipeline.model_trainer import DEFAULT_MODEL_PATH
from market_predictor.ml_pipeline.training_pipeline import TrainingPipeline

_config_manager = ConfigurationManager()


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
