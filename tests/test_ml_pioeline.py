import pandas as pd

from stock_price_predictor.ml_pipeline import model_predictor as model_predictor_module
from stock_price_predictor.ml_pipeline.model_predictor import ModelPredictor


def test_predict_output_omits_internal_paths(monkeypatch):
    predictor = ModelPredictor()
    gold_df = pd.DataFrame(
        [
            {"ticker": "GOOG", "date": "2026-03-10", "close": 100.0, "volume": 10.0, "target_up": 1},
        ]
    )

    class _FakeModel:
        def predict(self, _x):
            return [1]

        def predict_proba(self, _x):
            return [[0.2, 0.8]]

    monkeypatch.setattr(ModelPredictor, "_resolve_gold_with_features", lambda self: gold_df)
    monkeypatch.setattr(model_predictor_module, "gcs_blob_exists", lambda bucket_name, blob_name: True)
    monkeypatch.setattr(model_predictor_module, "load_object", lambda path: {"model": _FakeModel(), "feature_columns": ["close", "volume"]})
    monkeypatch.setattr(model_predictor_module.SentimentScoring, "score_headline", lambda self, text: 0.5)

    result = predictor.predict_from_ticker_headline(
        ticker="GOOG",
        headline="Google announces strong quarterly earnings",
        model_output_path="gs://bucket/model.joblib",
    )

    assert "source_date" in result
    assert "model_path" not in result
    assert "gold_with_features_path" not in result
