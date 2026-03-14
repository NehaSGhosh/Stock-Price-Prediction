# Market Sentiment & Stock Price Predictor

Production-ready, modular pipeline for:

- Market price + news sentiment data ingestion
- Data preprocessing and sentiment analysis
- Feature engineering
- ML model training
- On-demand prediction from ticker + headline

## Project Structure

```text
market_prediction/
├── artifacts/                         # Runtime outputs
│   ├── raw/                           # Ingested market/news CSVs
│   ├── processed/                     # Gold dataset (gold.csv)
│   └── models/                        # Trained model artifacts
├── config/
│   ├── config.yaml                    # Central project configuration values
│   └── configuration.py               # Configuration manager
├── notebooks/                         # Experiment notebooks
├── src/
│   └── market_predictor/
│       ├── ingestion/                 # Market/news ingestion modules
│       │   ├── ingestion_pipeline.py
│       │   ├── market_data_ingestion.py
│       │   └── news_ingestion.py
│       ├── warehousing/               # Gold dataset creation + cloud warehouse integration
│       │   ├── data_processing.py
│       │   └── data_storage.py
│       ├── ml_pipeline/               # Training and on-demand prediction modules
│       │   ├── feature_engineering.py
│       │   ├── training_pipeline.py
│       │   ├── model_trainer.py
│       │   └── model_predictor.py
│       ├── entity/                    # Dataclasses for config/artifacts
│       ├── utils/                     # Shared utility functions
│       ├── exception.py
│       └── logger.py
├── tests/
├── main.py                            # CLI entrypoint
└── requirements.txt
```

## Quick Start

### Dependendency management

1. Create and activate a virtual environment.

```bash
python -m venv .venv
source .venv/bin/activate
```

1. Install dependencies:

```bash
pip install -r requirements.txt
```

1. Set BigQuery credentials in `.env`:

```env
GOOGLE_APPLICATION_CREDENTIALS=D:\projects_code\market_prediction\service-account.json
```

1. Gold output paths are configured in `config/config.yaml` under `data_processing`.

### Data Ingestion

Data ingestion can run either as a standalone script or a GCP Cloud Function.

#### Standalone script

```bash
python src/market_predictor/ingestion/ingestion_pipeline.py
```

   Force fresh API pull for a specific window:

```bash
python src/market_predictor/ingestion/ingestion_pipeline.py --lookback_days 29
```

   Append recent days to existing raw CSV files:

```bash
python src/market_predictor/ingestion/ingestion_pipeline.py --append 1
```

   Override tickers together with a forced API window:

```bash
python src/market_predictor/ingestion/ingestion_pipeline.py --lookback_days 29 --tickers MSFT AAPL GOOG AVGO UBER
```

   Available standalone ingestion parameters and defaults:

| Parameter | Type | Default | Notes |
| --- | --- | --- | --- |
| `--tickers` | `list[str]` | `["MSFT","AAPL","GOOG","AVGO","UBER"]` | Space-separated list, e.g. `--tickers MSFT AAPL GOOG AVGO UBER`. |
| `--lookback_days` | `int` | `29` | If set, forces API calls even when raw CSV files already exist. |
| `--append` | `int` | `1` | If set, appends last `n` days to existing raw CSV files; throws if missing. |

   Constraint:

- `--lookback_days` and `--append` cannot be used together.

#### Google Cloud Function (HTTP)

```bash
curl -X POST "https://<FUNCTION_URL>" \
  -H "Content-Type: application/json" \
  -d '{"lookback_days": 29, "tickers": ["MSFT","AAPL","GOOG","AVGO","UBER"]}'
```

Append mode request:

```bash
curl -X POST "https://<FUNCTION_URL>" \
  -H "Content-Type: application/json" \
  -d '{"append": 1, "tickers": ["MSFT","AAPL","GOOG","AVGO","UBER"]}'
```

### Data Warehousing and Model Training

```bash
python main.py --mode train
```

Training flow is:

- Check local `gold.csv` at `data_processing.gold_path`.
- If local `gold.csv` is missing, download Gold from BigQuery (`data_processing.bigquery_gold_table_id`) for the configured lookback window and save it locally.
- If Gold is missing in both local and BigQuery, raise an error.
- Build `gold_with_features.csv` at `data_processing.gold_with_features_path` using:
  - `avg_sentiment_headlines` (average sentiment of `headlines_list`)
  - existing feature engineering logic (rolling returns/averages, lags, volume features, etc.)
- Replace/update BigQuery table `data_processing.bigquery_gold_with_features_table_id` with this feature dataset.
- Train the model using `train_test_split` with `model_trainer.train_ratio`.
- Run predictions on test split and compute train/test classification metrics.
- Save metrics JSON at `model_trainer.train_test_metrics_path` and log the same metrics.

Metrics JSON structure:

```json
{
  "train_metrics": {
    "accuracy": 0.0,
    "precision": 0.0,
    "recall": 0.0,
    "f1": 0.0
  },
  "test_metrics": {
    "accuracy": 0.0,
    "precision": 0.0,
    "recall": 0.0,
    "f1": 0.0
  },
  "train_rows": 0,
  "test_rows": 0,
  "feature_count": 0,
  "train_ratio": 0.8
}
```

### On-demand Prediction (Script + FastAPI)

Prediction requires non-empty `ticker` and `headline`.

Flow:

- Load local `gold_with_features.csv` from `data_processing.gold_with_features_path`.
- If missing locally, download `gold_with_features` from BigQuery.
- If missing in both, raise error.
- Validate requested ticker exists in dataset.
- Select latest row (by `date`) for ticker.
- Replace `headlines_list` with input headline and set `avg_sentiment_headlines` from that headline sentiment.
- Run model inference and return prediction.

Standalone script:

```bash
python src/market_predictor/ml_pipeline/model_predictor.py --ticker GOOG --headline "Google announces strong quarterly earnings"
```

FastAPI endpoint (if running with uvicorn):

```bash
uvicorn src.market_predictor.ml_pipeline.model_predictor:app --reload
```

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"ticker":"MSFT","headline":"Microsoft beats earnings estimates"}'
```

## Notes

- Replace placeholder APIs in `ingestion/ingestion_pipeline.py` with your data vendors.
- Current sentiment analysis uses VADER for fast baseline modeling.
- Model training currently uses Random Forest; extend with advanced models as needed.
